"""Flatten a ``compose`` block into an ordinary model.

Each composed model is read as an automaton whose states are its compartments.
The composition is their product, in the manner of a stochastic automata network:

* **States.** One compartment per combination of local states, named by joining
  the local names with ``_`` (``S_young``). The initial joint distribution is the
  independent one: the product of the marginals divided by ``N`` for each extra model.
* **Local transitions.** A flow of one model is copied into every combination of the
  other models' states (a Kronecker sum).
* **Synchronised transitions.** A ``sync`` rule moves several automata at once.
* **Rates.** In a rate expression, the name of the flow's source denotes the specific
  source compartment; any other state name, or ``_``-joined group of state names,
  denotes the marginal total over the models it does not mention.

The result is a plain EpiModel, so every backend supports composition unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from itertools import combinations, product
from typing import NoReturn

from epidsl.checks import errors
from epidsl.errors import CompositionError
from epidsl.ir import (
    BinOp,
    Call,
    Compartment,
    Composition,
    Constant,
    EpiModel,
    Expr,
    Flow,
    Neg,
    Ref,
    Semantics,
    Sym,
    Sync,
)

__all__ = ["CompositionError", "flatten"]

#: A cell of the product: one local state name per composed model.
Cell = tuple[str, ...]


def _fail(message: str, node: Composition | Ref) -> NoReturn:
    raise CompositionError(message, node.line, node.column)


def _name(states: tuple[str, ...]) -> str:
    return "_".join(states)


def flatten(composition: Composition, env: Mapping[str, EpiModel]) -> EpiModel:
    """Return the product model described by ``composition``.

    ``env`` holds the models defined before it, by name.
    """
    for ref in composition.models:
        if ref.name not in env:
            _fail(f"unknown model '{ref.name}'", ref)
    if len(composition.models) < 2:
        _fail(f"compose {composition.name} needs at least two models", composition)
    parts = [env[ref.name] for ref in composition.models]

    for ref, part in zip(composition.models, parts, strict=True):
        problems = errors(part)
        if problems:
            _fail(f"model '{part.name}' has errors: {problems[0].message}", ref)
        for flow in part.flows:
            if flow.source is None:
                _fail(
                    f"model '{part.name}' has an inflow, which cannot be composed yet: "
                    "the stratum that newcomers enter is unspecified",
                    ref,
                )

    owner = _owners(composition, parts)
    constants = _constants(composition, parts)
    total = _common_total(composition, parts)
    semantics = _semantics(composition, parts)

    cells: list[Cell] = list(product(*([c.name for c in part.compartments] for part in parts)))
    marginals = _marginals(composition, cells, len(parts), set(constants))

    initial = {c.name: c.initial for part in parts for c in part.compartments}
    compartments = tuple(
        Compartment(
            _name(cell),
            math.prod(initial[state] for state in cell) / total ** (len(parts) - 1),
            composition.line,
            composition.column,
        )
        for cell in cells
    )

    flows: list[Flow] = []
    for i, part in enumerate(parts):
        for flow in part.flows:
            assert flow.source is not None
            moves = {i: (flow.source, flow.target)}
            flows.extend(_lift(moves, flow.source, flow, cells, marginals))
    for sync in composition.syncs:
        sync_moves: dict[int, tuple[str, str | None]] = {}
        for source, target in sync.moves:
            for ref in (source, target):
                if ref.name not in owner:
                    _fail(f"'{ref.name}' is not a compartment of a composed model", ref)
            if owner[source.name] != owner[target.name]:
                _fail(
                    f"'{source.name}' and '{target.name}' belong to different models; "
                    "a move stays within one model",
                    source,
                )
            if source.name == target.name:
                _fail(f"move '{source.name} -> {target.name}' stays in the same state", source)
            if owner[source.name] in sync_moves:
                _fail("a sync rule takes at most one move per model", source)
            sync_moves[owner[source.name]] = (source.name, target.name)
        key = _name(tuple(sync_moves[i][0] for i in sorted(sync_moves)))
        flows.extend(_lift(sync_moves, key, sync, cells, marginals))

    return EpiModel(
        name=composition.name,
        compartments=compartments,
        constants=tuple(constants.values()),
        flows=tuple(flows),
        semantics=semantics,
        line=composition.line,
        column=composition.column,
    )


def _owners(composition: Composition, parts: list[EpiModel]) -> dict[str, int]:
    """Map each local state name to the index of the model that declares it."""
    owner: dict[str, int] = {}
    for i, part in enumerate(parts):
        for compartment in part.compartments:
            if compartment.name in owner:
                other = parts[owner[compartment.name]].name
                _fail(
                    f"compartment '{compartment.name}' is declared by both "
                    f"'{other}' and '{part.name}'",
                    composition.models[i],
                )
            owner[compartment.name] = i
    return owner


def _constants(composition: Composition, parts: list[EpiModel]) -> dict[str, Constant]:
    constants: dict[str, Constant] = {}
    declared_by: dict[str, str] = {}
    for i, part in enumerate(parts):
        for constant in part.constants:
            if constant.name in constants:
                _fail(
                    f"constant '{constant.name}' is declared by both "
                    f"'{declared_by[constant.name]}' and '{part.name}'",
                    composition.models[i],
                )
            constants[constant.name] = constant
            declared_by[constant.name] = part.name
    return constants


def _common_total(composition: Composition, parts: list[EpiModel]) -> float:
    totals = [sum(c.initial for c in part.compartments) for part in parts]
    if not all(math.isclose(t, totals[0], rel_tol=1e-9) for t in totals) or totals[0] <= 0:
        listing = ", ".join(f"{part.name}: {t:g}" for part, t in zip(parts, totals, strict=True))
        _fail(
            "composed models must have the same total population, and it must be positive "
            f"({listing})",
            composition,
        )
    return totals[0]


def _semantics(composition: Composition, parts: list[EpiModel]) -> tuple[Semantics, ...]:
    common = tuple(s for s in parts[0].semantics if all(s in part.semantics for part in parts))
    if not common:
        _fail("the composed models have no semantics in common", composition)
    return common


def _marginals(
    composition: Composition, cells: list[Cell], n_parts: int, taken: set[str]
) -> dict[str, list[str]]:
    """Map every name that denotes a group of cells to the names of those cells.

    A name is the ``_``-join of the local states of some of the models, in model
    order; it denotes all cells that agree with it on those models.
    """
    members: dict[str, list[str]] = {}
    meaning: dict[str, tuple[tuple[int, ...], tuple[str, ...]]] = {}
    subsets = [s for size in range(1, n_parts + 1) for s in combinations(range(n_parts), size)]
    for cell in cells:
        for subset in subsets:
            states = tuple(cell[i] for i in subset)
            name = _name(states)
            if meaning.setdefault(name, (subset, states)) != (subset, states) or name in taken:
                _fail(
                    f"the name '{name}' is ambiguous in compose {composition.name}; "
                    "rename a compartment or constant",
                    composition,
                )
            members.setdefault(name, []).append(_name(cell))
    return members


def _lift(
    moves: Mapping[int, tuple[str, str | None]],
    source_name: str,
    origin: Flow | Sync,
    cells: list[Cell],
    marginals: Mapping[str, list[str]],
) -> list[Flow]:
    """Instantiate a transition in every cell whose states match the moves' sources."""
    lifted: list[Flow] = []
    for cell in cells:
        if any(cell[i] != source for i, (source, _) in moves.items()):
            continue
        if any(target is None for _, target in moves.values()):
            destination = None
        else:
            destination = _name(
                tuple(moves[i][1] or state if i in moves else state for i, state in enumerate(cell))
            )
        lifted.append(
            Flow(
                _name(cell),
                destination,
                _rewrite(origin.expr, source_name, _name(cell), marginals),
                origin.line,
                origin.column,
            )
        )
    return lifted


def _rewrite(expr: Expr, source_name: str, cell: str, marginals: Mapping[str, list[str]]) -> Expr:
    """Replace the source's name by its cell and every other group name by a sum of cells."""
    match expr:
        case Sym(name, line, column) if name == source_name:
            return Sym(cell, line, column)
        case Sym(name, line, column) if name in marginals:
            terms: list[Expr] = [Sym(member, line, column) for member in marginals[name]]
            total = terms[0]
            for term in terms[1:]:
                total = BinOp("+", total, term)
            return total
        case Neg(operand):
            return Neg(_rewrite(operand, source_name, cell, marginals))
        case BinOp(op, left, right):
            return BinOp(
                op,
                _rewrite(left, source_name, cell, marginals),
                _rewrite(right, source_name, cell, marginals),
            )
        case Call(func, args, line, column):
            return Call(
                func, tuple(_rewrite(a, source_name, cell, marginals) for a in args), line, column
            )
        case _:
            return expr
