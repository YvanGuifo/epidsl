"""Parse EpiDSL source text into the typed IR."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any, Literal

from lark import Lark, Token, Transformer, v_args
from lark.exceptions import UnexpectedCharacters, UnexpectedInput, UnexpectedToken, VisitError

from epidsl.compose import flatten
from epidsl.errors import EpiSyntaxError
from epidsl.ir import (
    BinOp,
    Call,
    Compartment,
    Composition,
    Constant,
    Entry,
    EpiModel,
    Expr,
    Flow,
    Incidence,
    Neg,
    Num,
    Observable,
    Prior,
    Ref,
    Semantics,
    Stratified,
    Sym,
    Sync,
)

__all__ = ["EpiSyntaxError", "parse"]

_GRAMMAR = Path(__file__).with_name("grammar.lark")


@cache
def _lark() -> Lark:
    return Lark(_GRAMMAR.read_text(), parser="lalr", maybe_placeholders=True)


def _at(token: Token) -> dict[str, int]:
    return {"line": token.line or 0, "column": token.column or 0}


@v_args(inline=True)
class _ToIR(Transformer[Token, tuple[EpiModel | Composition, ...]]):
    def start(self, *items: EpiModel | Composition) -> tuple[EpiModel | Composition, ...]:
        return items

    def compose(self, name: Token, *statements: Any) -> Composition:
        models: tuple[Ref, ...] | None = None
        syncs: list[Sync] = []
        stratified: list[Stratified] = []
        entries: list[Entry] = []
        observables: list[Observable] = []
        constants: list[Constant] = []
        for item in statements:
            if isinstance(item, Sync):
                syncs.append(item)
            elif isinstance(item, Observable):
                observables.append(item)
            elif isinstance(item, Constant):
                constants.append(item)
            elif isinstance(item, Stratified):
                stratified.append(item)
            elif isinstance(item, Entry):
                entries.append(item)
            elif models is not None:
                raise EpiSyntaxError(f"compose {name} declares models twice", **_at(name))
            else:
                models = item
        return Composition(
            str(name),
            models or (),
            tuple(syncs),
            tuple(stratified),
            tuple(entries),
            tuple(observables),
            tuple(constants),
            **_at(name),
        )

    def stratified_parameter(self, name: Token, by: tuple[Ref, ...], *rest: Any) -> Stratified:
        *values, prior, unit = rest
        return _stratified("parameter", name, by, values, unit, prior)

    def stratified_rate(self, name: Token, by: tuple[Ref, ...], *rest: Any) -> Stratified:
        *values, unit = rest
        return _stratified("rate", name, by, values, unit, None)

    def by_models(self, *names: Token) -> tuple[Ref, ...]:
        return tuple(Ref(str(name), **_at(name)) for name in names)

    def stratum(self, *parts: Any) -> tuple[tuple[Ref, ...], float]:
        *states, value = parts
        return tuple(Ref(str(state), **_at(state)) for state in states), value

    def enter(self, model: Token, state: Token) -> Entry:
        return Entry(Ref(str(model), **_at(model)), Ref(str(state), **_at(state)), **_at(model))

    def models(self, *names: Token) -> tuple[Ref, ...]:
        return tuple(Ref(str(name), **_at(name)) for name in names)

    def sync(self, *parts: Any) -> Sync:
        *moves, expr = parts
        first: Ref = moves[0][0]
        return Sync(tuple(moves), expr, first.line, first.column)

    def move(self, source: Token, arrow: Token, target: Token) -> tuple[Ref, Ref]:
        return Ref(str(source), **_at(source)), Ref(str(target), **_at(target))

    def model(self, name: Token, *statements: Any) -> EpiModel:
        compartments: list[Compartment] = []
        constants: list[Constant] = []
        flows: list[Flow] = []
        observables: list[Observable] = []
        semantics: tuple[Semantics, ...] | None = None
        for item in statements:
            if isinstance(item, list):
                compartments.extend(item)
            elif isinstance(item, Constant):
                constants.append(item)
            elif isinstance(item, Flow):
                flows.append(item)
            elif isinstance(item, Observable):
                observables.append(item)
            elif semantics is not None:
                raise EpiSyntaxError(f"model {name} declares semantics twice", **_at(name))
            else:
                semantics = item
        return EpiModel(
            name=str(name),
            compartments=tuple(compartments),
            constants=tuple(constants),
            flows=tuple(flows),
            semantics=semantics or (Semantics.ODE,),
            observables=tuple(observables),
            line=name.line or 0,
            column=name.column or 0,
        )

    def compartments(self, *items: Compartment) -> list[Compartment]:
        return list(items)

    def compartment(self, name: Token, initial: float | None) -> Compartment:
        return Compartment(str(name), 0.0 if initial is None else initial, **_at(name))

    def parameter(
        self, name: Token, value: float, prior: Prior | None, unit: Token | None
    ) -> Constant:
        return Constant(
            str(name), "parameter", value, _unit(unit), name.line or 0, name.column or 0, prior
        )

    def prior(self, dist: Token, a: float, b: float) -> Prior:
        if dist not in ("uniform", "loguniform", "normal"):
            raise EpiSyntaxError(
                f"unknown distribution '{dist}' (expected uniform, loguniform or normal)",
                **_at(dist),
            )
        return Prior(str(dist), a, b)  # type: ignore[arg-type]

    def rate(self, name: Token, value: float, unit: Token | None) -> Constant:
        return Constant(str(name), "rate", value, _unit(unit), name.line or 0, name.column or 0)

    def flow(self, source: Token | None, arrow: Token, target: Token | None, expr: Expr) -> Flow:
        return Flow(
            None if source is None else str(source),
            None if target is None else str(target),
            expr,
            **_at(source or arrow),
        )

    def observe(self, name: Token, expr: Expr) -> Observable:
        return Observable(str(name), expr, **_at(name))

    def incidence(self, source: Token | None, arrow: Token, target: Token | None) -> Incidence:
        pair = (None if source is None else str(source), None if target is None else str(target))
        return Incidence((pair,), **_at(source or arrow))

    def semantics(self, *names: Token) -> tuple[Semantics, ...]:
        values: list[Semantics] = []
        for name in names:
            try:
                values.append(Semantics(str(name)))
            except ValueError:
                known = ", ".join(s.value for s in Semantics)
                raise EpiSyntaxError(
                    f"unknown semantics '{name}' (expected one of: {known})", **_at(name)
                ) from None
        return tuple(values)

    def positive(self, number: Token) -> float:
        return float(number)

    def negative(self, number: Token) -> float:
        return -float(number)

    def number(self, number: Token) -> Num:
        return Num(float(number))

    def symbol(self, name: Token) -> Sym:
        return Sym(str(name), **_at(name))

    def call(self, name: Token, *args: Expr | None) -> Call:
        return Call(str(name), tuple(a for a in args if a is not None), **_at(name))

    def neg(self, operand: Expr) -> Neg:
        return Neg(operand)

    def add(self, left: Expr, right: Expr) -> BinOp:
        return BinOp("+", left, right)

    def sub(self, left: Expr, right: Expr) -> BinOp:
        return BinOp("-", left, right)

    def mul(self, left: Expr, right: Expr) -> BinOp:
        return BinOp("*", left, right)

    def div(self, left: Expr, right: Expr) -> BinOp:
        return BinOp("/", left, right)

    def pow(self, left: Expr, right: Expr) -> BinOp:
        return BinOp("^", left, right)


def _stratified(
    kind: Literal["parameter", "rate"],
    name: Token,
    by: tuple[Ref, ...],
    values: list[Any],
    unit: Token | None,
    prior: Prior | None,
) -> Stratified:
    return Stratified(
        str(name),
        kind,
        by,
        tuple(values),
        _unit(unit),
        name.line or 0,
        name.column or 0,
        prior,
    )


def _unit(token: Token | None) -> str | None:
    return None if token is None else str(token)[1:-1].strip()


def _describe(error: UnexpectedInput) -> str:
    if isinstance(error, UnexpectedToken):
        if error.token.type == "$END":
            return "unexpected end of input"
        return f"unexpected '{error.token}'"
    if isinstance(error, UnexpectedCharacters):
        return f"unexpected character '{error.char}'"
    return "unexpected end of input"


def parse(source: str) -> tuple[EpiModel, ...]:
    """Parse ``source`` and return its models in source order.

    ``compose`` blocks are flattened into ordinary models. Raises EpiSyntaxError on
    malformed input and CompositionError on a composition that cannot be built.
    """
    try:
        tree = _lark().parse(source)
        items = _ToIR().transform(tree)
    except UnexpectedInput as error:
        line = error.line if error.line and error.line > 0 else source.count("\n") + 1
        column = error.column if error.column and error.column > 0 else 1
        raise EpiSyntaxError(_describe(error), line, column) from None
    except VisitError as error:
        if isinstance(error.orig_exc, EpiSyntaxError):
            raise error.orig_exc from None
        raise

    models: dict[str, EpiModel] = {}
    for item in items:
        if item.name in models:
            raise EpiSyntaxError(f"'{item.name}' is defined more than once", item.line, item.column)
        models[item.name] = flatten(item, models) if isinstance(item, Composition) else item
    return tuple(models.values())
