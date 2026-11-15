"""Typed intermediate representation shared by the parser, checks and backends.

Source positions (``line``/``column``, 1-based) are carried for diagnostics only
and never take part in equality.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np


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


Expr = Num | Sym | Neg | BinOp | Call


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
class EpiModel:
    name: str
    compartments: tuple[Compartment, ...]
    constants: tuple[Constant, ...]
    flows: tuple[Flow, ...]
    semantics: tuple[Semantics, ...] = (Semantics.ODE,)
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
    """A constant of one composed model given one value per state of another (``by``)."""

    name: str
    kind: Literal["parameter", "rate"]
    by: Ref
    values: tuple[tuple[Ref, float], ...]
    unit: str | None = None
    line: int = _pos()
    column: int = _pos()


@dataclass(frozen=True, slots=True)
class Entry:
    """The state of ``model`` that newcomers from another model's inflow enter."""

    model: Ref
    state: Ref
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
    line: int = _pos()
    column: int = _pos()


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    t_end: float
    n_points: int = 201
    t_start: float = 0.0
    #: Seed of the random generator for stochastic backends; None draws a fresh one.
    seed: int | None = None
