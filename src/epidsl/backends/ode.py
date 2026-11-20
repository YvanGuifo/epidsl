"""Deterministic backend: integrate the model's mean-field ODE system."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from epidsl.backends import BackendResult, compile_model, observe
from epidsl.expr import walk
from epidsl.ir import Call, EpiModel, Num, Semantics, SimulationConfig, Sym


def _step_thresholds(model: EpiModel, constants: dict[str, float]) -> list[float]:
    """Return every numeric threshold t0 found in step(t, t0) calls in flow rates.

    Only the common pattern ``step(t, <literal-or-constant>)`` is recognised.
    Thresholds embedded in more complex expressions (e.g. ``step(t, a+1)``)
    are silently skipped: the ODE will still run, just without the extra
    breakpoint hint.
    """
    thresholds: list[float] = []
    for flow in model.flows:
        for node in walk(flow.expr):
            if not (isinstance(node, Call) and node.func == "step" and len(node.args) == 2):
                continue
            t0_expr = node.args[1]
            if isinstance(t0_expr, Num):
                thresholds.append(float(t0_expr.value))
            elif isinstance(t0_expr, Sym) and t0_expr.name in constants:
                thresholds.append(float(constants[t0_expr.name]))
    return thresholds


class ODEBackend:
    """Integrates with LSODA, which switches between stiff and non-stiff methods."""

    def __init__(self, rtol: float = 1e-9, atol: float = 1e-9) -> None:
        self.rtol = rtol
        self.atol = atol

    def run(self, model: EpiModel, config: SimulationConfig) -> BackendResult:
        compiled = compile_model(model, Semantics.ODE)
        n = len(compiled.names)
        # One extra variable per tracked flow accumulates what it has moved.
        counter = {flow: n + i for i, flow in enumerate(compiled.tracked)}

        failure: list[Exception] = []

        def rhs(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
            state = y[:n]
            dy = np.zeros_like(y)
            for j, flow in enumerate(compiled.flows):
                try:
                    value = flow.rate(state, t)
                except (ArithmeticError, ValueError) as error:
                    # A rate left the representable range (overflow, division by zero,
                    # log of a negative number). Do not raise through the solver: some
                    # SciPy versions abort the interpreter. Remember the error and hand
                    # back NaN, which makes the solver stop.
                    failure.append(error)
                    return np.full_like(y, np.nan)
                if flow.source is not None:
                    dy[flow.source] -= value
                if flow.target is not None:
                    dy[flow.target] += value
                if j in counter:
                    dy[counter[j]] = value
            return dy

        user_grid = config.grid()
        constants = {k.name: k.value for k in model.constants}
        thresholds = [
            t0
            for t0 in _step_thresholds(model, constants)
            if float(user_grid[0]) < t0 < float(user_grid[-1])
        ]

        # Inject threshold breakpoints into t_eval so LSODA crosses the step
        # discontinuity at the exact right moment.  After solving, we extract
        # only the user-requested points so the output shape is unchanged.
        if thresholds:
            solve_grid = np.sort(np.unique(np.concatenate([user_grid, thresholds])))
        else:
            solve_grid = user_grid

        y0 = np.concatenate([compiled.initial, np.zeros(len(compiled.tracked))])
        solution = solve_ivp(
            rhs,
            (float(solve_grid[0]), float(solve_grid[-1])),
            y0,
            method="LSODA",
            t_eval=solve_grid,
            rtol=self.rtol,
            atol=self.atol,
        )
        if failure:
            raise RuntimeError(
                f"ODE solver failed for model {model.name}: "
                f"a flow rate could not be evaluated ({failure[0]})"
            ) from failure[0]
        if not solution.success:
            raise RuntimeError(f"ODE solver failed for model {model.name}: {solution.message}")

        # The solver interpolates output points; report the initial state exactly.
        solution.y[:, 0] = y0

        # If extra breakpoints were added, trim back to the user-requested grid.
        # user_grid values are an exact subset of solve_grid, so searchsorted is exact.
        if thresholds:
            idx = np.searchsorted(solution.t, user_grid)
            t_out = solution.t[idx]
            y_out = solution.y[:, idx]
        else:
            t_out = solution.t
            y_out = solution.y

        states = y_out[:n].T
        cumulative = y_out[n:].T

        return BackendResult(
            t=t_out,
            y={name: y_out[i] for i, name in enumerate(compiled.names)},
            metadata={"backend": "ODE", "method": "LSODA", "nfev": int(solution.nfev)},
            observed=observe(model, compiled, t_out, states, cumulative),
        )
