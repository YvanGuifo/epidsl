"""Typed intermediate representation shared by the parser, checks and backends.

Source positions (``line``/``column``, 1-based) are carried for diagnostics only
and never take part in equality.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from itertools import pairwise
from typing import Literal

import numpy as np
from numpy.typing import NDArray


def _pos() -> int:
    return field(default=0, compare=False)


@dataclass(frozen=True, slots=True)
class Num:
    value: float


@dataclass(frozen=True, slots=True)
class Sym:
    name: str
    line: int = _pos()
    column: int = _pos()


@dataclass(frozen=True, slots=True)
class Neg:
    operand: Expr


@dataclass(frozen=True, slots=True)
class BinOp:
    op: Literal["+", "-", "*", "/", "^"]
    left: Expr
    right: Expr


@dataclass(frozen=True, slots=True)
class Call:
    func: str
    args: tuple[Expr, ...]
    line: int = _pos()
    column: int = _pos()


@dataclass(frozen=True, slots=True)
class Incidence:
    """Number of individuals who moved along some flows since the previous output time.

    ``pairs`` lists the (source, target) endpoints of the flows counted; None stands
    for outside the population. Only meaningful in an observable.
    """

    pairs: tuple[tuple[str | None, str | None], ...]
    line: int = _pos()
    column: int = _pos()


Expr = Num | Sym | Neg | BinOp | Call | Incidence


class Semantics(Enum):
    ODE = "ODE"
    GILLESPIE = "Gillespie"
    TAU_LEAP = "TauLeap"
    ABM = "ABM"


@dataclass(frozen=True, slots=True)
class Compartment:
    name: str
    initial: float = 0.0
    line: int = _pos()
    column: int = _pos()


@dataclass(frozen=True, slots=True)
class Prior:
    """Prior distribution of a parameter, used by calibration.

    ``uniform(a, b)`` and ``loguniform(a, b)`` take bounds; ``normal(a, b)`` takes a
    mean and a standard deviation.
    """

    dist: Literal["uniform", "loguniform", "normal"]
    a: float
    b: float

    def problem(self) -> str | None:
        """Why the distribution is not well defined, or None if it is."""
        if self.dist == "normal":
            return None if self.b > 0 else "standard deviation must be positive"
        if self.dist == "loguniform" and self.a <= 0:
            return "bounds must be positive"
        return None if self.a < self.b else "lower bound must be below the upper bound"

    def pdf(self, x: float) -> float:
        """Probability density at ``x``."""
        match self.dist:
            case "uniform":
                return 1.0 / (self.b - self.a) if self.a <= x <= self.b else 0.0
            case "loguniform":
                inside = self.a <= x <= self.b
                return 1.0 / (x * math.log(self.b / self.a)) if inside else 0.0
            case "normal":
                z = (x - self.a) / self.b
                return math.exp(-0.5 * z * z) / (self.b * math.sqrt(2.0 * math.pi))

    def sample(self, rng: np.random.Generator) -> float:
        """Draw one value."""
        match self.dist:
            case "uniform":
                return float(rng.uniform(self.a, self.b))
            case "loguniform":
                return float(math.exp(rng.uniform(math.log(self.a), math.log(self.b))))
            case "normal":
                return float(rng.normal(self.a, self.b))


@dataclass(frozen=True, slots=True)
class NoiseDistribution:
    """Noise distribution of an observable, used by calibration.

    ``poisson()`` takes no parameters; ``normal(sd)`` takes a standard deviation;
    ``negbinomial(phi)`` takes a dispersion parameter.
    """

    dist: Literal["poisson", "normal", "negbinomial"]
    params: tuple[float, ...]

    def problem(self) -> str | None:
        """Why the distribution is not well defined, or None if it is."""
        match self.dist:
            case "poisson":
                return None if not self.params else "poisson() takes no parameters"
            case "normal":
                if len(self.params) != 1:
                    return "normal() takes exactly one parameter (sd)"
                return None if self.params[0] > 0 else "standard deviation must be positive"
            case "negbinomial":
                if len(self.params) != 1:
                    return "negbinomial() takes exactly one parameter (phi)"
                return None if self.params[0] > 0 else "dispersion must be positive"


@dataclass(frozen=True, slots=True)
class Constant:
    """A named constant. ``parameter`` marks it as calibratable; ``rate`` is fixed."""

    name: str
    kind: Literal["parameter", "rate"]
    value: float
    unit: str | None = None
    line: int = _pos()
    column: int = _pos()
    prior: Prior | None = None


@dataclass(frozen=True, slots=True)
class Flow:
    """Movement at rate ``expr``; ``source``/``target`` of None is an inflow/outflow."""

    source: str | None
    target: str | None
    expr: Expr
    line: int = _pos()
    column: int = _pos()


@dataclass(frozen=True, slots=True)
class Observable:
    """A quantity derived from the state, as it would be measured (``observe``)."""

    name: str
    expr: Expr
    line: int = _pos()
    column: int = _pos()
    noise: NoiseDistribution | None = None


@dataclass(frozen=True, slots=True)
class EpiModel:
    name: str
    compartments: tuple[Compartment, ...]
    constants: tuple[Constant, ...]
    flows: tuple[Flow, ...]
    semantics: tuple[Semantics, ...] = (Semantics.ODE,)
    observables: tuple[Observable, ...] = ()
    #: For a composed model: each component's name and compartments, in order. Its
    #: compartments are then the product of these, in the same order.
    parts: tuple[tuple[str, tuple[str, ...]], ...] = ()
    line: int = _pos()
    column: int = _pos()


@dataclass(frozen=True, slots=True)
class Ref:
    """A reference by name to a model or to one of its compartments."""

    name: str
    line: int = _pos()
    column: int = _pos()


@dataclass(frozen=True, slots=True)
class Sync:
    """A synchronised transition: every ``(source, target)`` move fires at once."""

    moves: tuple[tuple[Ref, Ref], ...]
    expr: Expr
    line: int = _pos()
    column: int = _pos()


@dataclass(frozen=True, slots=True)
class Stratified:
    """A constant of one composed model given one value per state of another (``by``).

    ``by`` names that model once; naming it twice declares a matrix, whose values are
    then keyed by an ordered pair of states: the one a flow leaves, then the one it meets.
    """

    name: str
    kind: Literal["parameter", "rate"]
    by: tuple[Ref, ...]
    values: tuple[tuple[tuple[Ref, ...], float], ...]
    unit: str | None = None
    line: int = _pos()
    column: int = _pos()
    #: Prior given to each generated parameter; None inherits the original's.
    prior: Prior | None = None


@dataclass(frozen=True, slots=True)
class Entry:
    """The state of ``model`` that newcomers from another model's inflow enter."""

    model: Ref
    state: Ref
    line: int = _pos()
    column: int = _pos()


@dataclass(frozen=True, slots=True)
class Migrate:
    """Cross-patch migration: named compartments move according to a rate matrix.

    ``rate`` must name a matrix-valued constant (``by Model, Model``) declared in
    the same compose block.  Every listed compartment gets one outflow to each other
    patch, weighted by the corresponding matrix entry.
    """

    compartments: tuple[str, ...]
    rate: str
    line: int = _pos()
    column: int = _pos()


@dataclass(frozen=True, slots=True)
class Composition:
    """A ``compose`` block, before it is flattened into an EpiModel."""

    name: str
    models: tuple[Ref, ...]
    syncs: tuple[Sync, ...]
    stratified: tuple[Stratified, ...] = ()
    entries: tuple[Entry, ...] = ()
    observables: tuple[Observable, ...] = ()
    #: Constants declared in the block itself, added to those of the models.
    constants: tuple[Constant, ...] = ()
    migrates: tuple[Migrate, ...] = ()
    line: int = _pos()
    column: int = _pos()


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """When to simulate and report.

    By default the output times are ``n_points`` evenly spaced from ``t_start`` to
    ``t_end``. ``SimulationConfig.at(times)`` reports at the given times instead.
    """

    t_end: float
    n_points: int = 201
    t_start: float = 0.0
    #: Seed of the random generator for stochastic backends; None draws a fresh one.
    seed: int | None = None
    #: Explicit output times; set through ``SimulationConfig.at``.
    times: tuple[float, ...] | None = None

    @classmethod
    def at(cls, times: Sequence[float], seed: int | None = None) -> SimulationConfig:
        """Report at ``times``, which must increase; the simulation starts at the first."""
        values = tuple(float(t) for t in times)
        if len(values) < 2:
            raise ValueError("at least two output times are needed")
        if any(later <= earlier for earlier, later in pairwise(values)):
            raise ValueError("output times must be increasing")
        return cls(
            t_end=values[-1], n_points=len(values), t_start=values[0], seed=seed, times=values
        )

    def grid(self) -> NDArray[np.float64]:
        """The output times."""
        if self.times is not None:
            return np.array(self.times, dtype=np.float64)
        return np.linspace(self.t_start, self.t_end, self.n_points)
