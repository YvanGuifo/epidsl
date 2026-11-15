"""Exact stochastic backend: Gillespie's direct method.

Compartments hold whole numbers of individuals. Each flow is a jump that moves one
individual, firing at a rate (propensity) equal to the flow's expression.
"""

from __future__ import annotations

import math

import numpy as np

from epidsl.backends import BackendResult, compile_model
from epidsl.expr import walk
from epidsl.ir import EpiModel, Semantics, SimulationConfig, Sym


class GillespieBackend:
    """Simulates one trajectory per run; ``config.seed`` makes it reproducible.

    ``max_events`` bounds the number of jumps, so that a model whose rates grow
    without limit fails instead of running forever.
    """

    def __init__(self, max_events: int = 10_000_000) -> None:
        self.max_events = max_events

    def run(self, model: EpiModel, config: SimulationConfig) -> BackendResult:
        compiled = compile_model(model, Semantics.GILLESPIE)
        for compartment in model.compartments:
            if compartment.initial != math.floor(compartment.initial):
                raise ValueError(
                    f"model {model.name}: initial value of '{compartment.name}' "
                    f"({compartment.initial:g}) must be a whole number for Gillespie semantics"
                )
        for declared in model.flows:
            if any(isinstance(node, Sym) and node.name == "t" for node in walk(declared.expr)):
                # The direct method assumes rates are constant between two jumps.
                raise ValueError(
                    f"model {model.name}: time-dependent rates are not supported "
                    "by the Gillespie backend"
                )

        rng = np.random.default_rng(config.seed)
        grid = np.linspace(config.t_start, config.t_end, config.n_points)
        output = np.empty((config.n_points, len(compiled.names)), dtype=np.float64)
        state = compiled.initial.copy()
        propensities = np.zeros(len(compiled.flows), dtype=np.float64)
        t = config.t_start
        recorded = 0
        events = 0

        while True:
            for i, flow in enumerate(compiled.flows):
                if flow.source is not None and state[flow.source] <= 0:
                    # Nobody left to move, whatever the expression evaluates to.
                    propensities[i] = 0.0
                    continue
                try:
                    value = flow.rate(state, t)
                except (ArithmeticError, ValueError) as error:
                    raise RuntimeError(
                        f"Gillespie simulation failed for model {model.name}: "
                        f"a flow rate could not be evaluated ({error})"
                    ) from error
                if not value >= 0:
                    raise RuntimeError(
                        f"Gillespie simulation failed for model {model.name}: "
                        f"a flow rate is negative or undefined ({value:g}) at t = {t:g}"
                    )
                propensities[i] = value

            total = float(propensities.sum())
            t_next = t + rng.exponential(1.0 / total) if total > 0 else math.inf

            # The state is constant until the next jump: fill the grid points before it.
            while recorded < config.n_points and grid[recorded] < t_next:
                output[recorded] = state
                recorded += 1
            if recorded == config.n_points:
                break

            if events == self.max_events:
                raise RuntimeError(
                    f"Gillespie simulation of model {model.name} needs more than "
                    f"{self.max_events} events (reached t = {t:g})"
                )
            threshold = rng.random() * total
            chosen = int(np.searchsorted(np.cumsum(propensities), threshold, side="right"))
            chosen = min(chosen, len(compiled.flows) - 1)
            while propensities[chosen] == 0.0:  # guard against rounding at the upper end
                chosen -= 1
            fired = compiled.flows[chosen]
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
        )
