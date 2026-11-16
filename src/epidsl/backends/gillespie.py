"""Exact stochastic backend: Gillespie's direct method.

Compartments hold whole numbers of individuals. Each flow is a jump that moves one
individual, firing at a rate (propensity) equal to the flow's expression.

When a rate names ``t``, the waiting time to the next jump is found by integrating
the total rate — exact up to the precision of the quadrature — instead of drawing it
from an exponential law with a fixed rate. Models without ``t`` take the usual path.
"""

from __future__ import annotations

import math

import numpy as np

from epidsl.backends import BackendResult, compile_model, observe
from epidsl.backends._stochastic import (
    TimedJumps,
    choose_flow,
    evaluate_propensities,
    require_whole_initial_values,
    uses_time,
)
from epidsl.ir import EpiModel, Semantics, SimulationConfig


class GillespieBackend:
    """Simulates one trajectory per run; ``config.seed`` makes it reproducible.

    ``max_events`` bounds the number of jumps, so that a model whose rates grow
    without limit fails instead of running forever.
    """

    def __init__(self, max_events: int = 10_000_000) -> None:
        self.max_events = max_events

    def run(self, model: EpiModel, config: SimulationConfig) -> BackendResult:
        compiled = compile_model(model, Semantics.GILLESPIE)
        require_whole_initial_values(model, "Gillespie")

        rng = np.random.default_rng(config.seed)
        grid = config.grid()
        output = np.empty((len(grid), len(compiled.names)), dtype=np.float64)
        counter = {flow: i for i, flow in enumerate(compiled.tracked)}
        moved = np.zeros(len(compiled.tracked), dtype=np.float64)
        moved_output = np.empty((len(grid), len(compiled.tracked)), dtype=np.float64)
        state = compiled.initial.copy()
        propensities = np.zeros(len(compiled.flows), dtype=np.float64)
        t = float(grid[0])
        recorded = 0
        events = 0

        jumps = TimedJumps(compiled, model.name, "Gillespie") if uses_time(model) else None
        # With rates that depend on time, the next jump falls where the integral of the
        # total rate reaches a unit-exponential draw; `owed` is what is left of it.
        owed = rng.exponential(1.0) if jumps is not None else 0.0
        while jumps is not None and recorded < len(grid):
            ahead = float(grid[recorded])
            if t >= ahead:
                output[recorded] = state
                moved_output[recorded] = moved
                recorded += 1
                continue
            mass = jumps.mass(state, t, ahead)
            if mass < owed:
                owed -= mass  # no jump before the next output time
                t = ahead
                continue
            if events == self.max_events:
                raise RuntimeError(
                    f"Gillespie simulation of model {model.name} needs more than "
                    f"{self.max_events} events (reached t = {t:g})"
                )
            t = jumps.reach(state, t, ahead, owed)
            owed = rng.exponential(1.0)
            evaluate_propensities(compiled, state, t, propensities, model.name, "Gillespie")
            total = float(propensities.sum())
            if total <= 0:
                continue  # the rate vanishes exactly here: nothing to choose from
            chosen = choose_flow(propensities, total, rng)
            fired = compiled.flows[chosen]
            if chosen in counter:
                moved[counter[chosen]] += 1
            if fired.source is not None:
                state[fired.source] -= 1
            if fired.target is not None:
                state[fired.target] += 1
            events += 1

        while jumps is None:
            evaluate_propensities(compiled, state, t, propensities, model.name, "Gillespie")
            total = float(propensities.sum())
            t_next = t + rng.exponential(1.0 / total) if total > 0 else math.inf

            # The state is constant until the next jump: fill the grid points before it.
            while recorded < len(grid) and grid[recorded] < t_next:
                output[recorded] = state
                moved_output[recorded] = moved
                recorded += 1
            if recorded == len(grid):
                break

            if events == self.max_events:
                raise RuntimeError(
                    f"Gillespie simulation of model {model.name} needs more than "
                    f"{self.max_events} events (reached t = {t:g})"
                )
            chosen = choose_flow(propensities, total, rng)
            fired = compiled.flows[chosen]
            if chosen in counter:
                moved[counter[chosen]] += 1
            if fired.source is not None:
                state[fired.source] -= 1
            if fired.target is not None:
                state[fired.target] += 1
            t = t_next
            events += 1

        return BackendResult(
            t=grid,
            y={name: output[:, i].copy() for i, name in enumerate(compiled.names)},
            metadata={"backend": "Gillespie", "events": events, "seed": config.seed},
            observed=observe(model, compiled, grid, output, moved_output),
        )
