"""Approximate stochastic backend: explicit tau-leaping with adaptive steps.

Same jump process as the Gillespie backend, but a step of length ``tau`` fires each
flow a Poisson number of times at once, with rates frozen at the start of the step.
When leaping would not save work (few individuals involved), single exact Gillespie
steps are taken instead.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from epidsl.backends import BackendResult, compile_model
from epidsl.backends._stochastic import (
    choose_flow,
    evaluate_propensities,
    require_time_independent_rates,
    require_whole_initial_values,
)
from epidsl.ir import EpiModel, Semantics, SimulationConfig

#: A leap is rejected and halved at most this many times before the run fails.
_MAX_HALVINGS = 40


class TauLeapBackend:
    """Simulates one trajectory per run; ``config.seed`` makes it reproducible.

    ``epsilon`` is the accuracy parameter: a step is chosen so that, in expectation,
    no compartment changes by more than this fraction of its size (or by one
    individual, if larger) and no more than this fraction of a compartment leaves it.
    Smaller is more accurate and slower.

    A leap shorter than ``exact_threshold`` mean inter-event times is replaced by one
    exact step. ``max_steps`` bounds the number of steps of either kind.
    """

    def __init__(
        self,
        epsilon: float = 0.03,
        exact_threshold: float = 10.0,
        max_steps: int = 10_000_000,
    ) -> None:
        if not 0 < epsilon < 1:
            raise ValueError("epsilon must be strictly between 0 and 1")
        self.epsilon = epsilon
        self.exact_threshold = exact_threshold
        self.max_steps = max_steps

    def run(self, model: EpiModel, config: SimulationConfig) -> BackendResult:
        compiled = compile_model(model, Semantics.TAU_LEAP)
        require_whole_initial_values(model, "TauLeap")
        require_time_independent_rates(model, "TauLeap")

        n_compartments, n_flows = len(compiled.names), len(compiled.flows)
        change = np.zeros((n_compartments, n_flows), dtype=np.float64)
        outflow = np.zeros((n_compartments, n_flows), dtype=np.float64)
        for j, flow in enumerate(compiled.flows):
            if flow.source is not None:
                change[flow.source, j] -= 1
                outflow[flow.source, j] = 1
            if flow.target is not None:
                change[flow.target, j] += 1

        rng = np.random.default_rng(config.seed)
        grid = np.linspace(config.t_start, config.t_end, config.n_points)
        output = np.empty((config.n_points, n_compartments), dtype=np.float64)
        state = compiled.initial.copy()
        propensities = np.zeros(n_flows, dtype=np.float64)
        t = config.t_start
        recorded = 0
        leaps = 0
        exact_steps = 0

        while recorded < config.n_points:
            if t >= grid[recorded]:
                output[recorded] = state
                recorded += 1
                continue

            evaluate_propensities(compiled, state, t, propensities, model.name, "TauLeap")
            total = float(propensities.sum())
            if total == 0:
                # Nothing can happen any more: the state is final.
                output[recorded:] = state
                break
            if leaps + exact_steps == self.max_steps:
                raise RuntimeError(
                    f"TauLeap simulation of model {model.name} needs more than "
                    f"{self.max_steps} steps (reached t = {t:g})"
                )

            horizon = float(grid[recorded] - t)
            tau = self._step(state, propensities, change, outflow)

            if tau < self.exact_threshold / total:
                wait = rng.exponential(1.0 / total)
                if wait >= horizon:
                    t = float(grid[recorded])  # no jump before the next output time
                    continue
                state += change[:, choose_flow(propensities, total, rng)]
                t += wait
                exact_steps += 1
                continue

            tau = min(tau, horizon)
            reaches_grid = tau == horizon
            for _ in range(_MAX_HALVINGS):
                candidate = state + change @ rng.poisson(propensities * tau)
                if candidate.min() >= 0:
                    break
                tau /= 2  # the leap emptied a compartment: retry with a shorter one
                reaches_grid = False
            else:
                raise RuntimeError(
                    f"TauLeap simulation failed for model {model.name}: "
                    f"no step keeps every compartment non-negative at t = {t:g}"
                )
            state = candidate
            t = float(grid[recorded]) if reaches_grid else t + tau
            leaps += 1

        return BackendResult(
            t=grid,
            y={name: output[:, i].copy() for i, name in enumerate(compiled.names)},
            metadata={
                "backend": "TauLeap",
                "leaps": leaps,
                "exact_steps": exact_steps,
                "seed": config.seed,
            },
        )

    def _step(
        self,
        state: NDArray[np.float64],
        propensities: NDArray[np.float64],
        change: NDArray[np.float64],
        outflow: NDArray[np.float64],
    ) -> float:
        """Largest step that keeps every compartment's expected change small."""
        allowed = np.maximum(self.epsilon * state, 1.0)
        drift = np.abs(change @ propensities)
        spread = (change**2) @ propensities
        leaving = outflow @ propensities

        bounds = [np.inf]
        if (moving := drift > 0).any():
            bounds.append(float((allowed[moving] / drift[moving]).min()))
        if (noisy := spread > 0).any():
            bounds.append(float((allowed[noisy] ** 2 / spread[noisy]).min()))
        if (draining := leaving > 0).any():
            bounds.append(float((self.epsilon * state[draining] / leaving[draining]).min()))
        return min(bounds)
