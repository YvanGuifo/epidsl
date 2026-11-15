"""Simulation backends. Each turns an EpiModel into trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from epidsl.checks import errors
from epidsl.expr import Evaluator, compile_expr
from epidsl.ir import EpiModel, Semantics, SimulationConfig


@dataclass(frozen=True, slots=True)
class BackendResult:
    """Trajectories on a common time grid, keyed by compartment name."""

    t: NDArray[np.float64]
    y: dict[str, NDArray[np.float64]]
    metadata: dict[str, object] = field(default_factory=dict)


class Backend(Protocol):
    """Anything that can simulate a model."""

    def run(self, model: EpiModel, config: SimulationConfig) -> BackendResult: ...


@dataclass(frozen=True, slots=True)
class CompiledFlow:
    """A flow with compartment positions resolved and its rate ready to evaluate."""

    source: int | None
    target: int | None
    rate: Evaluator


@dataclass(frozen=True, slots=True)
class CompiledModel:
    names: list[str]
    initial: NDArray[np.float64]
    flows: list[CompiledFlow]


def compile_model(model: EpiModel, semantics: Semantics) -> CompiledModel:
    """Validate ``model`` for a backend and resolve it against a state vector.

    Raises ValueError if the model has static errors or does not declare ``semantics``.
    """
    problems = errors(model)
    if problems:
        details = "; ".join(str(p) for p in problems)
        raise ValueError(f"model {model.name} has errors: {details}")
    if semantics not in model.semantics:
        declared = ", ".join(s.value for s in model.semantics)
        raise ValueError(
            f"model {model.name} does not declare {semantics.value} semantics "
            f"(declared: {declared})"
        )

    names = [c.name for c in model.compartments]
    index = {name: i for i, name in enumerate(names)}
    constants = {k.name: k.value for k in model.constants}
    return CompiledModel(
        names=names,
        initial=np.array([c.initial for c in model.compartments], dtype=np.float64),
        flows=[
            CompiledFlow(
                None if f.source is None else index[f.source],
                None if f.target is None else index[f.target],
                compile_expr(f.expr, index, constants),
            )
            for f in model.flows
        ],
    )
