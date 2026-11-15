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
* **Stratified constants.** ``rate beta by Age = {young: .., old: ..}`` replaces the
  constant ``beta`` by one constant per state of ``Age`` (``beta_young``, ...). A
  transition uses the one of the cell it leaves (or enters, for an inflow): this is
  a rate of one automaton that depends on the state of another.
* **Inflows.** Newcomers of one model's inflow enter the other models in the state
  given by ``enter Model at state``.
* **Observables.** Each model's observables are kept. A state name becomes the
  marginal total, and an incidence counts every transition of the product in which
  that model makes the move, synchronised transitions included.

The result is a plain EpiModel, so every backend supports composition unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from itertools import combinations, product
from typing import NoReturn, Protocol

from epidsl.checks import errors
from epidsl.errors import CompositionError
from epidsl.expr import walk
from epidsl.ir import (
    BinOp,
    Call,
    Compartment,
    Composition,
    Constant,
    EpiModel,
    Expr,
    Flow,
    Incidence,
    Neg,
    Observable,
    Semantics,
    Sym,
    Sync,
)

__all__ = ["CompositionError", "flatten"]

#: A cell of the product: one local state name per composed model.
Cell = tuple[str, ...]


class _Positioned(Protocol):
    @property
    def line(self) -> int: ...
    @property
    def column(self) -> int: ...


def _fail(message: str, node: _Positioned) -> NoReturn:
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

    owner = _owners(composition, parts)
    index = {ref.name: i for i, ref in enumerate(composition.models)}
    entry = _entries(composition, parts, index)
    for i, (ref, part) in enumerate(zip(composition.models, parts, strict=True)):
        if any(flow.source is None for flow in part.flows):
            for j, other in enumerate(parts):
                if j != i and j not in entry:
                    _fail(
                        f"model '{part.name}' has an inflow: say which state of '{other.name}' "
                        f"newcomers enter with 'enter {other.name} at <compartment>'",
                        ref,
                    )

    constants, strata = _stratify(composition, parts, index, _constants(composition, parts))
    total = _common_total(composition, parts)
    semantics = _semantics(composition, parts)

    cells: list[Cell] = list(product(*([c.name for c in part.compartments] for part in parts)))
    observed_names = {o.name for part in parts for o in part.observables}
    marginals = _marginals(composition, cells, len(parts), set(constants) | observed_names)

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
            if flow.source is None:
                assert flow.target is not None
                cell = tuple(flow.target if j == i else entry[j] for j in range(len(parts)))
                expr = _rewrite(flow.expr, None, "", marginals, _renames(cell, strata))
                flows.append(Flow(None, _name(cell), expr, flow.line, flow.column))
                continue
            moves = {i: (flow.source, flow.target)}
            flows.extend(_lift(moves, flow.source, flow, cells, marginals, strata))
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
        flows.extend(_lift(sync_moves, key, sync, cells, marginals, strata))

    observables: list[Observable] = []
    declared_by: dict[str, str] = {}
    for i, part in enumerate(parts):
        for observable in part.observables:
            if observable.name in declared_by:
                _fail(
                    f"observable '{observable.name}' is declared by both "
                    f"'{declared_by[observable.name]}' and '{part.name}'",
                    composition.models[i],
                )
            declared_by[observable.name] = part.name
            for node in walk(observable.expr):
                if isinstance(node, Sym) and node.name in strata:
                    _fail(
                        f"observable '{observable.name}' uses the stratified constant "
                        f"'{node.name}', which has no single value",
                        composition.models[i],
                    )
            lifted = _lift_observable(observable.expr, i, cells, marginals)
            observables.append(
                Observable(observable.name, lifted, observable.line, observable.column)
            )

    return EpiModel(
        name=composition.name,
        compartments=compartments,
        constants=tuple(constants.values()),
        flows=tuple(flows),
        semantics=semantics,
        observables=tuple(observables),
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


def _entries(
    composition: Composition, parts: list[EpiModel], index: Mapping[str, int]
) -> dict[int, str]:
    """Map a model's index to the state newcomers enter, from the ``enter`` statements."""
    entry: dict[int, str] = {}
    for declared in composition.entries:
        model, state = declared.model, declared.state
        if model.name not in index:
            _fail(f"'{model.name}' is not one of the composed models", model)
        i = index[model.name]
        if i in entry:
            _fail(f"model '{model.name}' has more than one entry state", model)
        if state.name not in {c.name for c in parts[i].compartments}:
            _fail(f"'{state.name}' is not a compartment of model '{model.name}'", state)
        entry[i] = state.name
    return entry


#: For each stratified constant: the index of the model it varies by, and the name
#: of the constant that replaces it in each state of that model.
Strata = dict[str, tuple[int, dict[str, str]]]


def _stratify(
    composition: Composition,
    parts: list[EpiModel],
    index: Mapping[str, int],
    constants: dict[str, Constant],
) -> tuple[dict[str, Constant], Strata]:
    """Replace each stratified constant by one constant per state of its ``by`` model."""
    strata: Strata = {}
    for declared in composition.stratified:
        name = declared.name
        if name in strata:
            _fail(f"constant '{name}' is stratified more than once", declared)
        if name not in constants:
            _fail(f"no composed model declares a constant '{name}'", declared)
        if declared.by.name not in index:
            _fail(f"'{declared.by.name}' is not one of the composed models", declared.by)
        by = index[declared.by.name]
        states = [c.name for c in parts[by].compartments]

        given: dict[str, float] = {}
        for state, value in declared.values:
            if state.name not in states:
                _fail(f"'{state.name}' is not a compartment of model '{declared.by.name}'", state)
            if state.name in given:
                _fail(f"'{state.name}' is given more than once", state)
            given[state.name] = value
        missing = [f"'{local}'" for local in states if local not in given]
        if missing:
            _fail(f"constant '{name}' has no value for {', '.join(missing)}", declared)

        unit = declared.unit if declared.unit is not None else constants[name].unit
        generated = {local: f"{name}_{local}" for local in states}
        replaced: dict[str, Constant] = {}
        for existing, constant in constants.items():
            if existing != name:
                replaced[existing] = constant
                continue
            for local in states:
                replaced[generated[local]] = Constant(
                    generated[local],
                    declared.kind,
                    given[local],
                    unit,
                    declared.line,
                    declared.column,
                )
        for new_name in generated.values():
            if new_name in constants:
                _fail(
                    f"the name '{new_name}' is ambiguous in compose {composition.name}; "
                    "rename a compartment or constant",
                    declared,
                )
        constants = replaced
        strata[name] = (by, generated)
    return constants, strata


def _renames(cell: Cell, strata: Strata) -> dict[str, str]:
    """The stratified constants as seen from ``cell``: original name -> its replacement."""
    return {name: generated[cell[by]] for name, (by, generated) in strata.items()}


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
    strata: Strata,
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
                _rewrite(origin.expr, source_name, _name(cell), marginals, _renames(cell, strata)),
                origin.line,
                origin.column,
            )
        )
    return lifted


def _lift_observable(
    expr: Expr, part: int, cells: list[Cell], marginals: Mapping[str, list[str]]
) -> Expr:
    """Rewrite an observable of model ``part`` for the product.

    State names become marginal totals. An incidence from ``a`` to ``b`` counts every
    transition whose cell goes from state ``a`` to state ``b`` in that model.
    """
    match expr:
        case Incidence(pairs, line, column):
            lifted: list[tuple[str | None, str | None]] = []
            for source, target in pairs:
                sources = (
                    [None] if source is None else [_name(c) for c in cells if c[part] == source]
                )
                targets = (
                    [None] if target is None else [_name(c) for c in cells if c[part] == target]
                )
                lifted.extend((a, b) for a in sources for b in targets)
            return Incidence(tuple(lifted), line, column)
        case Neg(operand):
            return Neg(_lift_observable(operand, part, cells, marginals))
        case BinOp(op, left, right):
            return BinOp(
                op,
                _lift_observable(left, part, cells, marginals),
                _lift_observable(right, part, cells, marginals),
            )
        case Call(func, args, line, column):
            lifted_args = tuple(_lift_observable(a, part, cells, marginals) for a in args)
            return Call(func, lifted_args, line, column)
        case _:
            return _rewrite(expr, None, "", marginals, {})


def _rewrite(
    expr: Expr,
    source_name: str | None,
    cell: str,
    marginals: Mapping[str, list[str]],
    renames: Mapping[str, str],
) -> Expr:
    """Rewrite a rate for one cell.

    The source's name becomes its cell, any other group name a sum of cells, and a
    stratified constant its replacement for this cell.
    """
    match expr:
        case Sym(name, line, column) if name == source_name:
            return Sym(cell, line, column)
        case Sym(name, line, column) if name in renames:
            return Sym(renames[name], line, column)
        case Sym(name, line, column) if name in marginals:
            terms: list[Expr] = [Sym(member, line, column) for member in marginals[name]]
            total = terms[0]
            for term in terms[1:]:
                total = BinOp("+", total, term)
            return total
        case Neg(operand):
            return Neg(_rewrite(operand, source_name, cell, marginals, renames))
        case BinOp(op, left, right):
            return BinOp(
                op,
                _rewrite(left, source_name, cell, marginals, renames),
                _rewrite(right, source_name, cell, marginals, renames),
            )
        case Call(func, args, line, column):
            rewritten = tuple(_rewrite(a, source_name, cell, marginals, renames) for a in args)
            return Call(func, rewritten, line, column)
        case _:
            return expr
