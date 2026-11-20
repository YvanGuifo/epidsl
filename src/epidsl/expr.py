"""Evaluation of rate expressions against a state vector."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping

import numpy as np
from numpy.typing import NDArray

from epidsl.ir import BinOp, Call, Expr, Incidence, Neg, Num, Sym

State = NDArray[np.float64]
Evaluator = Callable[[State, float], float]

#: Built-in functions: name -> (arity, implementation).
def _step(t: float, t0: float) -> float:
    """Heaviside step: 0 for t < t0, 1 for t >= t0."""
    return 1.0 if t >= t0 else 0.0


FUNCTIONS: Mapping[str, tuple[int, Callable[..., float]]] = {
    "exp": (1, math.exp),
    "log": (1, math.log),
    "sqrt": (1, math.sqrt),
    "sin": (1, math.sin),
    "cos": (1, math.cos),
    "abs": (1, abs),
    "min": (2, min),
    "max": (2, max),
    "step": (2, _step),
}

#: Built-in symbols: total population and current time.
BUILTIN_SYMBOLS = frozenset({"N", "t"})

RESERVED = BUILTIN_SYMBOLS | frozenset(FUNCTIONS)


def walk(expr: Expr) -> Iterator[Expr]:
    """Yield ``expr`` and every sub-expression, depth-first."""
    yield expr
    match expr:
        case Neg(operand):
            yield from walk(operand)
        case BinOp(_, left, right):
            yield from walk(left)
            yield from walk(right)
        case Call(_, args):
            for arg in args:
                yield from walk(arg)
        case _:
            pass


def compile_expr(
    expr: Expr,
    index: Mapping[str, int],
    constants: Mapping[str, float],
    incidence: Callable[[Incidence], float] | None = None,
) -> Evaluator:
    """Turn ``expr`` into a function of (state, time).

    ``index`` maps compartment names to positions in the state vector. ``incidence``
    gives the current value of an incidence term; it is needed for observables only.
    The expression is assumed to have passed the static checks.
    """
    match expr:
        case Num(value):
            return lambda y, t: value
        case Sym("N"):
            return lambda y, t: float(y.sum())
        case Sym("t"):
            return lambda y, t: t
        case Sym(name) if name in index:
            i = index[name]
            return lambda y, t: float(y[i])
        case Sym(name):
            constant = constants[name]
            return lambda y, t: constant
        case Neg(operand):
            inner = compile_expr(operand, index, constants, incidence)
            return lambda y, t: -inner(y, t)
        case BinOp(op, left, right):
            lhs = compile_expr(left, index, constants, incidence)
            rhs = compile_expr(right, index, constants, incidence)
            match op:
                case "+":
                    return lambda y, t: lhs(y, t) + rhs(y, t)
                case "-":
                    return lambda y, t: lhs(y, t) - rhs(y, t)
                case "*":
                    return lambda y, t: lhs(y, t) * rhs(y, t)
                case "/":
                    return lambda y, t: lhs(y, t) / rhs(y, t)
                case "^":
                    return lambda y, t: lhs(y, t) ** rhs(y, t)
        case Incidence() if incidence is not None:
            lookup, node = incidence, expr
            return lambda y, t: lookup(node)
        case Call(func, args):
            _, implementation = FUNCTIONS[func]
            compiled = [compile_expr(arg, index, constants, incidence) for arg in args]
            return lambda y, t: float(implementation(*(arg(y, t) for arg in compiled)))
    raise TypeError(f"not an expression: {expr!r}")


_PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2, "^": 4}
_NEGATION = 3


def _precedence(expr: Expr) -> int:
    if isinstance(expr, BinOp):
        return _PRECEDENCE[expr.op]
    return _NEGATION if isinstance(expr, Neg) else 5


def format_expr(expr: Expr) -> str:
    """Write ``expr`` back as source text, with only the parentheses it needs."""

    def wrapped(child: Expr, needs: bool) -> str:
        text = format_expr(child)
        return f"({text})" if needs else text

    match expr:
        case Num(value):
            return f"{value:g}"
        case Sym(name):
            return name
        case Neg(operand):
            return "-" + wrapped(operand, _precedence(operand) < _NEGATION)
        case BinOp(op, left, right):
            mine = _PRECEDENCE[op]
            theirs = _precedence(left), _precedence(right)
            if op == "^":
                left_needs = theirs[0] <= mine
                right_needs = theirs[1] < mine and not isinstance(right, Neg)
            else:
                left_needs = theirs[0] < mine
                right_needs = theirs[1] < mine or (theirs[1] == mine and op in "-/")
            return f"{wrapped(left, left_needs)} {op} {wrapped(right, right_needs)}"
        case Call(func, args):
            return f"{func}({', '.join(format_expr(arg) for arg in args)})"
        case Incidence(pairs):
            moves = ", ".join(" ".join(filter(None, (a, "->", b))) for a, b in pairs)
            return f"incidence({moves})"
    raise TypeError(f"not an expression: {expr!r}")
