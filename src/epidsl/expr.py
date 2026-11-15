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
FUNCTIONS: Mapping[str, tuple[int, Callable[..., float]]] = {
    "exp": (1, math.exp),
    "log": (1, math.log),
    "sqrt": (1, math.sqrt),
    "sin": (1, math.sin),
    "cos": (1, math.cos),
    "abs": (1, abs),
    "min": (2, min),
    "max": (2, max),
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
