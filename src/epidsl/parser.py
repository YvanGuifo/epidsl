"""Parse EpiDSL source text into the typed IR."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

from lark import Lark, Token, Transformer, v_args
from lark.exceptions import UnexpectedCharacters, UnexpectedInput, UnexpectedToken, VisitError

from epidsl.ir import (
    BinOp,
    Call,
    Compartment,
    Constant,
    EpiModel,
    Expr,
    Flow,
    Neg,
    Num,
    Semantics,
    Sym,
)

_GRAMMAR = Path(__file__).with_name("grammar.lark")


class EpiSyntaxError(Exception):
    def __init__(self, message: str, line: int, column: int) -> None:
        super().__init__(f"{line}:{column}: syntax error: {message}")
        self.message = message
        self.line = line
        self.column = column


@cache
def _lark() -> Lark:
    return Lark(_GRAMMAR.read_text(), parser="lalr", maybe_placeholders=True)


def _at(token: Token) -> dict[str, int]:
    return {"line": token.line or 0, "column": token.column or 0}


@v_args(inline=True)
class _ToIR(Transformer[Token, tuple[EpiModel, ...]]):
    def start(self, *models: EpiModel) -> tuple[EpiModel, ...]:
        return models

    def model(self, name: Token, *statements: Any) -> EpiModel:
        compartments: list[Compartment] = []
        constants: list[Constant] = []
        flows: list[Flow] = []
        semantics: tuple[Semantics, ...] | None = None
        for item in statements:
            if isinstance(item, list):
                compartments.extend(item)
            elif isinstance(item, Constant):
                constants.append(item)
            elif isinstance(item, Flow):
                flows.append(item)
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
            **_at(name),
        )

    def compartments(self, *items: Compartment) -> list[Compartment]:
        return list(items)

    def compartment(self, name: Token, initial: float | None) -> Compartment:
        return Compartment(str(name), 0.0 if initial is None else initial, **_at(name))

    def parameter(self, name: Token, value: float, unit: Token | None) -> Constant:
        return Constant(str(name), "parameter", value, _unit(unit), **_at(name))

    def rate(self, name: Token, value: float, unit: Token | None) -> Constant:
        return Constant(str(name), "rate", value, _unit(unit), **_at(name))

    def flow(self, source: Token | None, arrow: Token, target: Token | None, expr: Expr) -> Flow:
        return Flow(
            None if source is None else str(source),
            None if target is None else str(target),
            expr,
            **_at(source or arrow),
        )

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
    """Parse ``source`` and return its models, raising EpiSyntaxError on bad input."""
    try:
        tree = _lark().parse(source)
        return _ToIR().transform(tree)
    except UnexpectedInput as error:
        line = error.line if error.line and error.line > 0 else source.count("\n") + 1
        column = error.column if error.column and error.column > 0 else 1
        raise EpiSyntaxError(_describe(error), line, column) from None
    except VisitError as error:
        if isinstance(error.orig_exc, EpiSyntaxError):
            raise error.orig_exc from None
        raise
