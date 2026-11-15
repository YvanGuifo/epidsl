"""Static checks on a parsed model.

C1  duplicate declaration                      error
C2  flow endpoint is not a compartment         error
C3  undefined symbol / bad function call       error
C4  degenerate flow (same or no endpoints)     error
C5  rate or parameter without a unit           warning
C6  negative or non-finite declared value      error
C7  unused compartment, rate or parameter      warning
C8  declaration shadows a reserved name        error
C11 ill-defined prior, or value outside it     error
C12 incidence in a flow rate, or of no flow    error

(C9 and C10 are reserved for mass conservation and undeclared semantics.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from epidsl.expr import BUILTIN_SYMBOLS, FUNCTIONS, RESERVED, walk
from epidsl.ir import Call, Compartment, Constant, EpiModel, Expr, Incidence, Observable, Sym


class Severity(Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class Diagnostic:
    code: str
    severity: Severity
    message: str
    line: int
    column: int

    def __str__(self) -> str:
        return f"{self.line}:{self.column}: {self.severity.value} {self.code}: {self.message}"


def check(model: EpiModel) -> list[Diagnostic]:
    """Return every diagnostic for ``model``; an empty list means it is clean."""
    found: list[Diagnostic] = []

    def error(code: str, message: str, node: object) -> None:
        found.append(_diagnostic(code, Severity.ERROR, message, node))

    def warning(code: str, message: str, node: object) -> None:
        found.append(_diagnostic(code, Severity.WARNING, message, node))

    compartments = {c.name for c in model.compartments}
    constants = {k.name for k in model.constants}

    seen: set[str] = set()
    declarations: tuple[Compartment | Constant | Observable, ...] = (
        *model.compartments,
        *model.constants,
        *model.observables,
    )
    for decl in declarations:
        if decl.name in seen:
            error("C1", f"'{decl.name}' is declared more than once", decl)
        seen.add(decl.name)
        if decl.name in RESERVED:
            error("C8", f"'{decl.name}' is a reserved name", decl)

    for compartment in model.compartments:
        if not math.isfinite(compartment.initial) or compartment.initial < 0:
            error(
                "C6",
                f"initial value of '{compartment.name}' must be finite and non-negative",
                compartment,
            )

    for constant in model.constants:
        if not math.isfinite(constant.value):
            error("C6", f"value of '{constant.name}' must be finite", constant)
        if constant.unit is None:
            warning("C5", f"{constant.kind} '{constant.name}' has no unit", constant)
        if constant.prior is not None:
            problem = constant.prior.problem()
            if problem is not None:
                error("C11", f"prior of '{constant.name}': {problem}", constant)
            elif constant.prior.pdf(constant.value) == 0.0:
                error(
                    "C11",
                    f"prior of '{constant.name}': value {constant.value:g} lies outside its prior",
                    constant,
                )

    used: set[str] = set()
    endpoints = {(flow.source, flow.target) for flow in model.flows}

    def check_expr(expr: Expr, in_observable: bool) -> None:
        for node in walk(expr):
            if isinstance(node, Sym):
                used.add(node.name)
                if node.name not in compartments | constants | BUILTIN_SYMBOLS:
                    error("C3", f"undefined symbol '{node.name}'", node)
            elif isinstance(node, Call):
                if node.func not in FUNCTIONS:
                    error("C3", f"unknown function '{node.func}'", node)
                elif len(node.args) != FUNCTIONS[node.func][0]:
                    arity = FUNCTIONS[node.func][0]
                    error(
                        "C3",
                        f"function '{node.func}' takes {arity} argument(s), got {len(node.args)}",
                        node,
                    )
            elif isinstance(node, Incidence):
                if not in_observable:
                    error("C12", "incidence can only be used in an observable", node)
                    continue
                unknown = [
                    name
                    for pair in node.pairs
                    for name in pair
                    if name and name not in compartments
                ]
                for name in unknown:
                    error("C2", f"incidence endpoint '{name}' is not a compartment", node)
                if not unknown and not any(pair in endpoints for pair in node.pairs):
                    source, target = node.pairs[0]
                    if source is None:
                        where = f"into {target}"
                    elif target is None:
                        where = f"out of {source}"
                    else:
                        where = f"from {source} to {target}"
                    error("C12", f"no flow goes {where}", node)

    for flow in model.flows:
        for endpoint in (flow.source, flow.target):
            if endpoint is None:
                continue
            used.add(endpoint)
            if endpoint not in compartments:
                error("C2", f"flow endpoint '{endpoint}' is not a compartment", flow)
        if flow.source == flow.target:
            what = (
                "neither a source nor a target"
                if flow.source is None
                else "one compartment at both ends"
            )
            error("C4", f"flow has {what}", flow)

        check_expr(flow.expr, in_observable=False)

    for observable in model.observables:
        check_expr(observable.expr, in_observable=True)

    for compartment in model.compartments:
        if compartment.name not in used:
            warning("C7", f"compartment '{compartment.name}' is never used", compartment)
    for constant in model.constants:
        if constant.name not in used:
            warning("C7", f"{constant.kind} '{constant.name}' is never used", constant)

    return found


def errors(model: EpiModel) -> list[Diagnostic]:
    return [d for d in check(model) if d.severity is Severity.ERROR]


def _diagnostic(code: str, severity: Severity, message: str, node: object) -> Diagnostic:
    return Diagnostic(code, severity, message, getattr(node, "line", 0), getattr(node, "column", 0))
