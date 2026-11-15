"""Deterministic backend: integrate the model's mean-field ODE system."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from epidsl.backends import BackendResult, compile_model
from epidsl.ir import EpiModel, Semantics, SimulationConfig


class ODEBackend:
    """Integrates with LSODA, which switches between stiff and non-stiff methods."""

    def __init__(self, rtol: float = 1e-9, atol: float = 1e-9) -> None:
        self.rtol = rtol
        self.atol = atol

    def run(self, model: EpiModel, config: SimulationConfig) -> BackendResult:
        compiled = compile_model(model, Semantics.ODE)

        def rhs(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
            dy = np.zeros_like(y)
            for flow in compiled.flows:
                value = flow.rate(y, t)
                if flow.source is not None:
                    dy[flow.source] -= value
                if flow.target is not None:
                    dy[flow.target] += value
            return dy

        t_eval = np.linspace(config.t_start, config.t_end, config.n_points)
        try:
            solution = solve_ivp(
                rhs,
                (config.t_start, config.t_end),
                compiled.initial,
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
        solution.y[:, 0] = compiled.initial

        return BackendResult(
            t=solution.t,
            y={name: solution.y[i] for i, name in enumerate(compiled.names)},
            metadata={"backend": "ODE", "method": "LSODA", "nfev": int(solution.nfev)},
        )
