"""Deterministic backend: integrate the model's mean-field ODE system."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from epidsl.backends import BackendResult
from epidsl.checks import errors
from epidsl.expr import compile_expr
from epidsl.ir import EpiModel, Semantics, SimulationConfig


class ODEBackend:
    """Integrates with LSODA, which switches between stiff and non-stiff methods."""

    def __init__(self, rtol: float = 1e-9, atol: float = 1e-9) -> None:
        self.rtol = rtol
        self.atol = atol

    def run(self, model: EpiModel, config: SimulationConfig) -> BackendResult:
        problems = errors(model)
        if problems:
            details = "; ".join(str(p) for p in problems)
            raise ValueError(f"model {model.name} has errors: {details}")
        if Semantics.ODE not in model.semantics:
            declared = ", ".join(s.value for s in model.semantics)
            raise ValueError(
                f"model {model.name} does not declare ODE semantics (declared: {declared})"
            )

        names = [c.name for c in model.compartments]
        index = {name: i for i, name in enumerate(names)}
        constants = {k.name: k.value for k in model.constants}
        flows = [
            (
                None if f.source is None else index[f.source],
                None if f.target is None else index[f.target],
                compile_expr(f.expr, index, constants),
            )
            for f in model.flows
        ]

        def rhs(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
            dy = np.zeros_like(y)
            for source, target, rate in flows:
                value = rate(y, t)
                if source is not None:
                    dy[source] -= value
                if target is not None:
                    dy[target] += value
            return dy

        t_eval = np.linspace(config.t_start, config.t_end, config.n_points)
        y0 = np.array([c.initial for c in model.compartments], dtype=np.float64)
        try:
            solution = solve_ivp(
                rhs,
                (config.t_start, config.t_end),
                y0,
                method="LSODA",
                t_eval=t_eval,
                rtol=self.rtol,
                atol=self.atol,
            )
        except (ArithmeticError, ValueError) as error:
            # A rate left the representable range (overflow, division by zero, log of a
            # negative number): the trajectory has diverged or the model is ill-posed.
            raise RuntimeError(
                f"ODE solver failed for model {model.name}: "
                f"a flow rate could not be evaluated ({error})"
            ) from error
        if not solution.success:
            raise RuntimeError(f"ODE solver failed for model {model.name}: {solution.message}")

        # The solver interpolates output points; report the initial state exactly.
        solution.y[:, 0] = y0

        return BackendResult(
            t=solution.t,
            y={name: solution.y[i] for i, name in enumerate(names)},
            metadata={"backend": "ODE", "method": "LSODA", "nfev": int(solution.nfev)},
        )
