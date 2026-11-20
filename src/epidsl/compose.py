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
* **Contact matrices.** ``by Age, Age`` gives one constant per ordered pair of states
  (``beta_young_old``). A rate that uses one is the sum, over the states of ``Age``, of
  that rate with the matrix on the pair (the cell's own state, the state met) and every
  group of compartments restricted to the cells in the state met.
* **Inflows.** Newcomers of one model's inflow enter the other models in the state
  given by ``enter Model at state``.
* **Observables.** Each model's observables are kept. A state name becomes the
  marginal total, and an incidence counts every transition of the product in which
  that model makes the move, synchronised transitions included. The block may add
  its own, written with cell names and group names, and its own constants.

The result is a plain EpiModel, so every backend supports composition unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
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
    Num,
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

    owner = _owners(composition, parts)
    index = {ref.name: i for i, ref in enumerate(composition.models)}
    entry = _entries(composition, parts, index)
    # Names declared as stratified constants in the compose block may be referenced
    # in component flows before stratification resolves them — allow them past C3.
    stratified_names = frozenset(s.name for s in composition.stratified)
    for i, (ref, part) in enumerate(zip(composition.models, parts, strict=True)):
        problems = errors(part, stratified_names)
        if problems:
            _fail(f"model '{part.name}' has errors: {problems[0].message}", ref)
        if any(flow.source is None for flow in part.flows):
            for j, other in enumerate(parts):
                if j != i and j not in entry:
                    _fail(
                        f"model '{part.name}' has an inflow: say which state of '{other.name}' "
                        f"newcomers enter with 'enter {other.name} at <compartment>'",
                        ref,
                    )

    migrate_rates = {m.rate for m in composition.migrates}
    constants, strata = _stratify(composition, parts, index, _constants(composition, parts), migrate_rates)
    for constant in composition.constants:
        if constant.name in constants or constant.name in strata:
            _fail(
                f"the name '{constant.name}' is already used in compose {composition.name}",
                constant,
            )
        constants[constant.name] = constant
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
                rate = _rate_for(flow.expr, None, cell, cells, marginals, strata, flow)
                flows.append(Flow(None, _name(cell), rate, flow.line, flow.column))
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

    flows.extend(_migration_flows(composition, parts, cells, strata, owner))

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
            states = _states_of(i, cells)
            lifted = _lift_observable(observable.expr, states, marginals)
            observables.append(
                Observable(observable.name, lifted, observable.line, observable.column)
            )

    for observable in composition.observables:
        name = observable.name
        if name in declared_by or name in constants or name in marginals or name in strata:
            _fail(f"the name '{name}' is already used in compose {composition.name}", observable)
        declared_by[name] = composition.name
        for node in walk(observable.expr):
            if isinstance(node, Sym) and node.name in strata:
                replacements = ", ".join(strata[node.name].generated.values())
                _fail(
                    f"observable '{name}' uses the stratified constant '{node.name}'; "
                    f"name one of {replacements}",
                    node,
                )
            if isinstance(node, Incidence):
                for endpoint in (e for pair in node.pairs for e in pair):
                    if endpoint is not None and endpoint not in marginals:
                        _fail(
                            f"'{endpoint}' is not a compartment or group of compartments "
                            f"of compose {composition.name}",
                            node,
                        )
        lifted = _lift_observable(observable.expr, marginals, marginals)
        observables.append(Observable(name, lifted, observable.line, observable.column))

    return EpiModel(
        name=composition.name,
        compartments=compartments,
        constants=tuple(constants.values()),
        flows=tuple(flows),
        semantics=semantics,
        observables=tuple(observables),
        parts=tuple((part.name, tuple(c.name for c in part.compartments)) for part in parts),
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


@dataclass(frozen=True, slots=True)
class Stratum:
    """A stratified constant: the model it varies by, and the constants replacing it.

    ``generated`` is keyed by the state of that model, or — for a matrix — by the
    ordered pair (the state a flow leaves, the state it meets).
    """

    by: int
    by_name: str
    generated: dict[tuple[str, ...], str]
    matrix: bool


#: Every stratified constant of a composition, by its original name.
Strata = dict[str, Stratum]


def _stratify(
    composition: Composition,
    parts: list[EpiModel],
    index: Mapping[str, int],
    constants: dict[str, Constant],
    migrate_rates: set[str] = frozenset(),
) -> tuple[dict[str, Constant], Strata]:
    """Replace each stratified constant by one constant per state, or per ordered pair."""
    strata: Strata = {}
    for declared in composition.stratified:
        name = declared.name
        if name in strata:
            _fail(f"constant '{name}' is stratified more than once", declared)
        for ref in declared.by:
            if ref.name not in index:
                _fail(f"'{ref.name}' is not one of the composed models", ref)
        matrix = len(declared.by) > 1
        by_name = declared.by[0].name
        if matrix and declared.by[1].name != by_name:
            _fail(
                f"a matrix is indexed by one model twice; write 'by {by_name}, {by_name}'",
                declared.by[1],
            )
        by = index[by_name]
        states = [c.name for c in parts[by].compartments]
        keys: list[tuple[str, ...]] = (
            [(a, b) for a in states for b in states] if matrix else [(a,) for a in states]
        )

        given: dict[tuple[str, ...], float] = {}
        for refs, value in declared.values:
            if len(refs) != len(keys[0]):
                _fail(
                    f"a value of '{name}' needs a pair of states, as 'young, old: 0.2'"
                    if matrix
                    else f"'{name}' takes one state per value; declare it "
                    f"'by {by_name}, {by_name}' to give it a pair",
                    refs[0],
                )
            for ref in refs:
                if ref.name not in states:
                    _fail(f"'{ref.name}' is not a compartment of model '{by_name}'", ref)
            key = tuple(ref.name for ref in refs)
            if key in given:
                _fail(f"'{', '.join(key)}' is given more than once", refs[0])
            given[key] = value
        if matrix and name in migrate_rates:
            # Sparse migration matrix: omitted pairs default to 0 (no edge).
            for key in keys:
                given.setdefault(key, 0.0)
        else:
            missing = [f"'{', '.join(key)}'" for key in keys if key not in given]
            if missing:
                _fail(f"constant '{name}' has no value for {', '.join(missing)}", declared)

        unit = declared.unit if declared.unit is not None else (
            constants[name].unit if name in constants else None
        )
        prior = declared.prior
        if prior is None and declared.kind == "parameter" and name in constants:
            prior = constants[name].prior
        generated = {key: "_".join((name, *key)) for key in keys}
        for new_name in generated.values():
            if new_name in constants:
                _fail(
                    f"the name '{new_name}' is ambiguous in compose {composition.name}; "
                    "rename a compartment or constant",
                    declared,
                )
        if name not in constants:
            # New constant declared inline in the compose block.
            # For migration matrices, skip the diagonal: migrate only generates
            # off-diagonal flows, so diagonal constants would trigger C7.
            # Contact matrices need the diagonal (self-transmission within a group).
            for key in keys:
                if matrix and name in migrate_rates and len(key) == 2 and key[0] == key[1]:
                    continue
                constants[generated[key]] = Constant(
                    generated[key],
                    declared.kind,
                    given[key],
                    unit,
                    declared.line,
                    declared.column,
                    prior,
                )
        else:
            replaced: dict[str, Constant] = {}
            for existing, constant in constants.items():
                if existing != name:
                    replaced[existing] = constant
                    continue
                for key in keys:
                    replaced[generated[key]] = Constant(
                        generated[key],
                        declared.kind,
                        given[key],
                        unit,
                        declared.line,
                        declared.column,
                        prior,
                    )
            constants = replaced
        strata[name] = Stratum(by, by_name, generated, matrix)
    return constants, strata


#: Stands in for a name that is not stratified, so a lookup needs no second test.
_PLAIN = Stratum(0, "", {}, False)


def _renames(cell: Cell, strata: Strata, met: str | None = None) -> dict[str, str]:
    """The stratified constants as seen from ``cell``, meeting a cell in state ``met``.

    A matrix is left out when no state is met, since it then stands for no one value.
    """
    names: dict[str, str] = {}
    for name, stratum in strata.items():
        if not stratum.matrix:
            names[name] = stratum.generated[(cell[stratum.by],)]
        elif met is not None:
            names[name] = stratum.generated[(cell[stratum.by], met)]
    return names


def _symbols(expr: Expr) -> set[str]:
    return {node.name for node in walk(expr) if isinstance(node, Sym)}


def _rate_for(
    expr: Expr,
    source_name: str | None,
    cell: Cell,
    cells: list[Cell],
    marginals: Mapping[str, list[str]],
    strata: Strata,
    origin: _Positioned,
) -> Expr:
    """Rewrite a rate for ``cell``, summing over the state met when it uses a matrix."""
    matrices = sorted(name for name in _symbols(expr) if strata.get(name, _PLAIN).matrix)
    if not matrices:
        return _rewrite(expr, source_name, _name(cell), marginals, _renames(cell, strata))
    models = {strata[name].by for name in matrices}
    if len(models) > 1:
        _fail(
            f"the rate uses matrices of two different models ({', '.join(matrices)}); "
            "a rate is summed over one model only",
            origin,
        )
    by = models.pop()
    if not any(name in marginals and name != source_name for name in _symbols(expr)):
        _fail(
            f"the rate uses the matrix '{matrices[0]}' but names no group of compartments; "
            f"a matrix is summed over the state of '{strata[matrices[0]].by_name}' it meets",
            origin,
        )

    state_of = {_name(c): c[by] for c in cells}
    terms = [
        _rewrite(
            expr,
            source_name,
            _name(cell),
            {
                name: [member for member in members if state_of[member] == met]
                for name, members in marginals.items()
            },
            _renames(cell, strata, met),
        )
        for met in dict.fromkeys(c[by] for c in cells)
    ]
    total = terms[0]
    for term in terms[1:]:
        total = BinOp("+", total, term)
    return total


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
                _rate_for(origin.expr, source_name, cell, cells, marginals, strata, origin),
                origin.line,
                origin.column,
            )
        )
    return lifted


def _states_of(part: int, cells: list[Cell]) -> dict[str, list[str]]:
    """Map each state of model ``part`` to the names of the cells in that state."""
    states: dict[str, list[str]] = {}
    for cell in cells:
        states.setdefault(cell[part], []).append(_name(cell))
    return states


def _lift_observable(
    expr: Expr, endpoints: Mapping[str, list[str]], marginals: Mapping[str, list[str]]
) -> Expr:
    """Rewrite an observable for the product.

    Group names become totals. An incidence from ``a`` to ``b`` counts every
    transition from a cell of ``a`` to a cell of ``b``, ``endpoints`` giving the cells
    that each name stands for.
    """
    match expr:
        case Incidence(pairs, line, column):
            lifted: list[tuple[str | None, str | None]] = []
            for source, target in pairs:
                sources = [None] if source is None else endpoints[source]
                targets = [None] if target is None else endpoints[target]
                lifted.extend((a, b) for a in sources for b in targets)
            return Incidence(tuple(lifted), line, column)
        case Neg(operand):
            return Neg(_lift_observable(operand, endpoints, marginals))
        case BinOp(op, left, right):
            return BinOp(
                op,
                _lift_observable(left, endpoints, marginals),
                _lift_observable(right, endpoints, marginals),
            )
        case Call(func, args, line, column):
            lifted_args = tuple(_lift_observable(a, endpoints, marginals) for a in args)
            return Call(func, lifted_args, line, column)
        case _:
            return _rewrite(expr, None, "", marginals, {})


def _migration_flows(
    composition: Composition,
    parts: list[EpiModel],
    cells: list[Cell],
    strata: Strata,
    owner: dict[str, int],
) -> list[Flow]:
    """Generate cross-patch flows from every ``migrate`` statement."""
    result: list[Flow] = []
    for migrate in composition.migrates:
        rate_name = migrate.rate
        if rate_name not in strata:
            _fail(
                f"'migrate ... : {rate_name}' — '{rate_name}' is not a matrix constant "
                f"in compose {composition.name}; declare it with "
                f"'parameter {rate_name} by <Model>, <Model> = ...'",
                migrate,
            )
        stratum = strata[rate_name]
        if not stratum.matrix:
            _fail(
                f"'migrate ... : {rate_name}' — '{rate_name}' is not a matrix constant; "
                f"declare it with 'by {stratum.by_name}, {stratum.by_name}'",
                migrate,
            )
        by = stratum.by
        patch_states = [c.name for c in parts[by].compartments]

        for comp_name in migrate.compartments:
            if comp_name not in owner:
                _fail(
                    f"'migrate' lists '{comp_name}', which is not a compartment "
                    f"of any composed model in compose {composition.name}",
                    migrate,
                )
            if owner[comp_name] == by:
                _fail(
                    f"'migrate' lists '{comp_name}', which belongs to model "
                    f"'{parts[by].name}' — the same model that '{rate_name}' indexes; "
                    f"list compartments of a different model",
                    migrate,
                )

        for src_patch in patch_states:
            for dst_patch in patch_states:
                if src_patch == dst_patch:
                    continue
                rate_const = stratum.generated[(src_patch, dst_patch)]
                for comp_name in migrate.compartments:
                    m = owner[comp_name]
                    for cell in cells:
                        if cell[m] != comp_name or cell[by] != src_patch:
                            continue
                        dst_cell = tuple(
                            dst_patch if i == by else s for i, s in enumerate(cell)
                        )
                        source_cell_name = _name(cell)
                        dest_cell_name = _name(dst_cell)
                        result.append(
                            Flow(
                                source_cell_name,
                                dest_cell_name,
                                BinOp("*", Sym(rate_const), Sym(source_cell_name)),
                                migrate.line,
                                migrate.column,
                            )
                        )
    return result


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
            if not marginals[name]:
                return Num(0.0)  # no cell of this group is in the state being met
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
