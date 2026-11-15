"""Typed intermediate representation shared by the parser, checks and backends.

Source positions (``line``/``column``, 1-based) are carried for diagnostics only
and never take part in equality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


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
class Constant:
    """A named constant. ``parameter`` marks it as calibratable; ``rate`` is fixed."""

    name: str
    kind: Literal["parameter", "rate"]
    value: float
    unit: str | None = None
    line: int = _pos()
    column: int = _pos()


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
class Composition:
    """A ``compose`` block, before it is flattened into an EpiModel."""

    name: str
    models: tuple[Ref, ...]
    syncs: tuple[Sync, ...]
    line: int = _pos()
    column: int = _pos()


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    t_end: float
    n_points: int = 201
    t_start: float = 0.0
