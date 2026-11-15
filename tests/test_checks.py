from epidsl.checks import Diagnostic, Severity, check
from epidsl.parser import parse


def diagnostics(body: str) -> list[Diagnostic]:
    (model,) = parse(f"model M {{ {body} }}")
    return check(model)


def codes(body: str, severity: Severity | None = None) -> list[str]:
    return [d.code for d in diagnostics(body) if severity is None or d.severity is severity]


CLEAN = "compartment S = 9, I = 1  rate b = 0.1 [1/day]  flow S -> I : b * S * I / N"


def test_clean_model_has_no_diagnostics() -> None:
    assert diagnostics(CLEAN) == []


def test_c1_duplicate_declaration() -> None:
    assert "C1" in codes("compartment S, S")
    assert "C1" in codes("compartment S  rate S = 1 [1/day]  flow S -> : S")


def test_c2_flow_endpoint_is_not_a_compartment() -> None:
    assert codes("compartment S  flow S -> X : S", Severity.ERROR) == ["C2"]
    assert "C2" in codes("compartment S  rate g = 1 [1/day]  flow S -> g : g * S")


def test_c3_undefined_symbol_and_unknown_function() -> None:
    assert codes("compartment S  flow S -> : nope * S", Severity.ERROR) == ["C3"]
    assert codes("compartment S  flow S -> : frob(S)", Severity.ERROR) == ["C3"]
    assert codes("compartment S  flow S -> : exp(S, S)", Severity.ERROR) == ["C3"]


def test_c4_degenerate_flow() -> None:
    assert codes("compartment S  flow S -> S : S", Severity.ERROR) == ["C4"]


def test_c5_missing_unit_is_a_warning() -> None:
    found = diagnostics("compartment S  rate g = 1  flow S -> : g * S")
    assert [(d.code, d.severity) for d in found] == [("C5", Severity.WARNING)]


def test_c6_negative_initial_value() -> None:
    assert codes("compartment S = -1  flow S -> : S", Severity.ERROR) == ["C6"]


def test_c7_unused_declarations_are_warnings() -> None:
    found = diagnostics(
        "compartment S, Z  rate g = 1 [1/day]  rate h = 2 [1/day]  flow S -> : g * S"
    )
    assert [(d.code, d.severity) for d in found] == [
        ("C7", Severity.WARNING),
        ("C7", Severity.WARNING),
    ]
    assert "Z" in found[0].message and "h" in found[1].message


def test_c8_reserved_name() -> None:
    assert "C8" in codes("compartment N")
    assert "C8" in codes("compartment S  rate t = 1 [1/day]  flow S -> : S")
    assert "C8" in codes("compartment exp")


def test_diagnostic_carries_source_position() -> None:
    (model,) = parse("model M {\n  compartment S\n  flow S -> : nope\n}")
    (diag,) = [d for d in check(model) if d.code == "C3"]
    assert (diag.line, diag.column) == (3, 15)
