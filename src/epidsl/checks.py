"""Static checks C1-C8 on a parsed model.

C1  duplicate declaration                      error
C2  flow endpoint is not a compartment         error
C3  undefined symbol / bad function call       error
C4  degenerate flow (same or no endpoints)     error
C5  rate or parameter without a unit           warning
C6  negative or non-finite declared value      error
C7  unused compartment, rate or parameter      warning
C8  declaration shadows a reserved name        error
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from epidsl.expr import BUILTIN_SYMBOLS, FUNCTIONS, RESERVED, walk
from epidsl.ir import Call, Compartment, Constant, EpiModel, Sym


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
    declarations: tuple[Compartment | Constant, ...] = (*model.compartments, *model.constants)
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

    used: set[str] = set()
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

        for node in walk(flow.expr):
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
