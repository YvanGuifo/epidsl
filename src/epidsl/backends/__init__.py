"""Simulation backends. Each turns an EpiModel into trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from epidsl.checks import errors
from epidsl.expr import Evaluator, compile_expr, walk
from epidsl.ir import EpiModel, Incidence, NoiseDistribution, Semantics, SimulationConfig


@dataclass(frozen=True, slots=True)
class BackendResult:
    """Trajectories on a common time grid.

    ``y`` holds the compartments and ``observed`` the model's observables, both
    keyed by name.
    """

    t: NDArray[np.float64]
    y: dict[str, NDArray[np.float64]]
    metadata: dict[str, object] = field(default_factory=dict)
    observed: dict[str, NDArray[np.float64]] = field(default_factory=dict)


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
    #: Indices of the flows whose cumulative count the observables need.
    tracked: list[int]


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
    counted = {pair for node in _incidences(model) for pair in node.pairs}
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
        tracked=[j for j, f in enumerate(model.flows) if (f.source, f.target) in counted],
    )


def _incidences(model: EpiModel) -> list[Incidence]:
    return [
        node
        for observable in model.observables
        for node in walk(observable.expr)
        if isinstance(node, Incidence)
    ]


def observe(
    model: EpiModel,
    compiled: CompiledModel,
    t: NDArray[np.float64],
    states: NDArray[np.float64],
    cumulative: NDArray[np.float64],
    rng: np.random.Generator | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Evaluate the model's observables at every output time.

    ``states[k]`` is the state at ``t[k]`` and ``cumulative[k, i]`` the number of
    individuals moved so far by flow ``compiled.tracked[i]``. An incidence term is
    the count over the interval that ends at the output time, and zero at the first.

    When ``rng`` is provided and an observable declares a noise distribution, the
    returned values are drawn from that distribution (mean = deterministic value).
    Without ``rng`` the deterministic mean is always returned.
    """
    if not model.observables:
        return {}
    moved = np.diff(cumulative, axis=0, prepend=cumulative[:1])
    column = {flow: i for i, flow in enumerate(compiled.tracked)}
    series: dict[Incidence, NDArray[np.float64]] = {}
    for node in _incidences(model):
        columns = [
            column[j] for j, f in enumerate(model.flows) if (f.source, f.target) in node.pairs
        ]
        series[node] = moved[:, columns].sum(axis=1)

    index = {name: i for i, name in enumerate(compiled.names)}
    constants = {k.name: k.value for k in model.constants}
    current: dict[Incidence, float] = {}
    observed: dict[str, NDArray[np.float64]] = {}
    for observable in model.observables:
        evaluate = compile_expr(observable.expr, index, constants, current.__getitem__)
        values = np.empty(len(t), dtype=np.float64)
        for k in range(len(t)):
            for node, counts in series.items():
                current[node] = float(counts[k])
            try:
                values[k] = evaluate(states[k], float(t[k]))
            except (ArithmeticError, ValueError) as error:
                raise RuntimeError(
                    f"observable '{observable.name}' could not be evaluated "
                    f"at t = {t[k]:g} ({error})"
                ) from error
        if rng is not None and observable.noise is not None:
            values = _apply_noise(observable.noise, values, rng)
        if observable.delay > 0 and len(t) > 1:
            dt = float(t[-1] - t[0]) / (len(t) - 1)
            steps = int(round(observable.delay / dt))
            if steps >= len(values):
                values = np.zeros_like(values)
            elif steps > 0:
                shifted = np.empty_like(values)
                shifted[:steps] = 0.0
                shifted[steps:] = values[:-steps]
                values = shifted
        observed[observable.name] = values
    return observed


def _apply_noise(
    noise: NoiseDistribution,
    values: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Draw one sample per time point from ``noise`` centred on ``values``."""
    mu = np.maximum(values, 0.0)
    if noise.dist == "poisson":
        return rng.poisson(mu).astype(np.float64)
    elif noise.dist == "normal":
        sd = noise.params[0]
        return rng.normal(mu, sd)
    else:  # negbinomial
        phi = noise.params[0]
        # Parametrisation: mean = mu, variance = mu + mu^2 / phi
        # scipy / numpy NB uses (n=phi, p=phi/(phi+mu))
        p = np.where(mu > 0, phi / (phi + mu), 1.0)
        return rng.negative_binomial(phi, p).astype(np.float64)
