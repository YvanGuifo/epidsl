"""C9 (mass conservation) and C10 (undeclared semantics) static checks."""


from epidsl.checks import Diagnostic, Severity, check, check_semantics
from epidsl.ir import EpiModel, Semantics
from epidsl.parser import parse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SIR = """
model SIR {
    semantics: [ODE, Gillespie, TauLeap]
    compartment S = 990, I = 10, R = 0
    rate beta  = 0.3 [1/day]
    rate gamma = 0.1 [1/day]
    flow S -> I : beta * S * I / N
    flow I -> R : gamma * I
}
"""

_OPEN_INFLOW = """
model M {
    semantics: [ODE]
    compartment S = 990, I = 10, R = 0
    rate beta  = 0.3 [1/day]
    rate gamma = 0.1 [1/day]
    rate mu    = 0.01 [1/day]
    flow -> S  : mu * N
    flow S -> I : beta * S * I / N
    flow I -> R : gamma * I
}
"""

_OPEN_OUTFLOW = """
model M {
    semantics: [ODE]
    compartment S = 990, I = 10, R = 0
    rate beta  = 0.3 [1/day]
    rate gamma = 0.1 [1/day]
    rate mu    = 0.01 [1/day]
    flow S -> I : beta * S * I / N
    flow I -> R : gamma * I
    flow S ->   : mu * S
}
"""

_DEMOGRAPHIC = """
model M {
    semantics: [ODE]
    compartment S = 990, I = 10, R = 0
    rate beta  = 0.3 [1/day]
    rate gamma = 0.1 [1/day]
    rate mu    = 0.01 [1/day]
    flow -> S  : mu * N
    flow S -> I : beta * S * I / N
    flow I -> R : gamma * I
    flow S ->   : mu * S
    flow I ->   : mu * I
    flow R ->   : mu * R
}
"""

_ODE_ONLY = """
model M {
    semantics: [ODE]
    compartment S = 990, I = 10, R = 0
    rate beta  = 0.3 [1/day]
    rate gamma = 0.1 [1/day]
    flow S -> I : beta * S * I / N
    flow I -> R : gamma * I
}
"""


def _model(source: str, name: str = "M") -> EpiModel:
    return next(m for m in parse(source) if m.name == name)


def _codes(diagnostics: list[Diagnostic]) -> list[str]:
    return [d.code for d in diagnostics]


def _warnings(diagnostics: list[Diagnostic]) -> list[Diagnostic]:
    return [d for d in diagnostics if d.severity is Severity.WARNING]


# ---------------------------------------------------------------------------
# C9 — mass conservation
# ---------------------------------------------------------------------------


def test_c9_closed_sir_is_clean() -> None:
    diags = check(_model(_SIR, "SIR"))
    assert "C9" not in _codes(diags)


def test_c9_demographic_model_is_clean() -> None:
    """Balanced inflows + outflows → no C9."""
    diags = check(_model(_DEMOGRAPHIC))
    assert "C9" not in _codes(diags)


def test_c9_warns_inflows_without_outflows() -> None:
    diags = check(_model(_OPEN_INFLOW))
    c9 = [d for d in diags if d.code == "C9"]
    assert len(c9) == 1
    assert c9[0].severity is Severity.WARNING
    assert "inflow" in c9[0].message.lower()


def test_c9_warns_outflows_without_inflows() -> None:
    diags = check(_model(_OPEN_OUTFLOW))
    c9 = [d for d in diags if d.code == "C9"]
    assert len(c9) == 1
    assert c9[0].severity is Severity.WARNING
    assert "outflow" in c9[0].message.lower()


def test_c9_has_a_source_position() -> None:
    """C9 must carry a non-zero source position."""
    diags = check(_model(_OPEN_INFLOW))
    c9 = next(d for d in diags if d.code == "C9")
    assert c9.line > 0


def test_c9_strict_mode_turns_warning_into_error() -> None:
    """In strict mode the CLI treats warnings as errors; C9 must surface."""
    diags = check(_model(_OPEN_INFLOW))
    c9 = [d for d in diags if d.code == "C9"]
    assert any(d.severity is Severity.WARNING for d in c9)


# ---------------------------------------------------------------------------
# C10 — undeclared semantics
# ---------------------------------------------------------------------------


def test_c10_no_error_when_semantics_declared() -> None:
    diag = check_semantics(_model(_SIR, "SIR"), Semantics.GILLESPIE)
    assert diag is None


def test_c10_error_when_semantics_not_declared() -> None:
    diag = check_semantics(_model(_ODE_ONLY), Semantics.GILLESPIE)
    assert diag is not None
    assert diag.code == "C10"
    assert diag.severity is Severity.ERROR
    assert "Gillespie" in diag.message
    assert "ODE" in diag.message


def test_c10_error_mentions_model_name() -> None:
    diag = check_semantics(_model(_ODE_ONLY), Semantics.TAU_LEAP)
    assert diag is not None
    assert "M" in diag.message


def test_c10_has_source_position() -> None:
    diag = check_semantics(_model(_ODE_ONLY), Semantics.GILLESPIE)
    assert diag is not None
    assert diag.line > 0


def test_c10_all_declared_semantics_pass() -> None:
    m = _model(_SIR, "SIR")
    for sem in (Semantics.ODE, Semantics.GILLESPIE, Semantics.TAU_LEAP):
        assert check_semantics(m, sem) is None


def test_c10_tau_leap_error_when_not_declared() -> None:
    diag = check_semantics(_model(_ODE_ONLY), Semantics.TAU_LEAP)
    assert diag is not None
    assert diag.code == "C10"
