"""Tests for the LSP server pure functions."""

from __future__ import annotations

from lsprotocol import types as lsp

from epidsl.lsp import _check_document, _get_completions, _get_definition

_SIR = """\
model SIR {
    semantics: [ODE, Gillespie, TauLeap]
    compartment S = 990, I = 10, R = 0
    rate beta = 0.3 [1/day]
    rate gamma = 0.1 [1/day]
    flow S -> I : beta * S * I / N
    flow I -> R : gamma * I
}
"""

_SIMPLE = "model M {\n  compartment S = 10, I = 0\n  flow S -> I : S\n}"


# ---------------------------------------------------------------------------
# _check_document
# ---------------------------------------------------------------------------


def test_clean_model_has_no_diagnostics() -> None:
    assert _check_document(_SIR) == []


def test_error_emits_error_diagnostic() -> None:
    diags = _check_document("model M { compartment S  flow S -> X : S }")
    assert any(d.code == "C2" and d.severity is lsp.DiagnosticSeverity.Error for d in diags)


def test_warning_emits_warning_diagnostic() -> None:
    # C5: rate without unit
    src = "model M { compartment S = 9, I = 1  rate g = 1  flow S -> I : g * S }"
    diags = _check_document(src)
    assert any(d.severity is lsp.DiagnosticSeverity.Warning for d in diags)


def test_parse_error_emits_diagnostic() -> None:
    diags = _check_document("this is not valid EpiDSL")
    assert len(diags) >= 1
    assert diags[0].severity is lsp.DiagnosticSeverity.Error


def test_diagnostic_positions_are_zero_based() -> None:
    # C3: 'nope' undefined — in source line 2 (1-based) = 1 (0-based)
    src = "model M {\n  compartment S\n  flow S -> : nope * S\n}"
    diags = _check_document(src)
    c3 = [d for d in diags if d.code == "C3"]
    assert c3
    assert c3[0].range.start.line == 2  # line 3 (1-based) → 2 (0-based)


def test_diagnostic_source_is_epidsl() -> None:
    diags = _check_document("model M { compartment S  flow S -> X : S }")
    assert all(d.source == "epidsl" for d in diags)


def test_multiple_errors_all_reported() -> None:
    # C2 on X, C2 on Y
    diags = _check_document("model M { compartment S  flow S -> X : S  flow S -> Y : S }")
    c2 = [d for d in diags if d.code == "C2"]
    assert len(c2) == 2


# ---------------------------------------------------------------------------
# _get_completions
# ---------------------------------------------------------------------------


def test_completions_include_keywords() -> None:
    items = _get_completions(_SIR, 0, 0)
    labels = {i.label for i in items}
    assert {"model", "compartment", "flow", "rate", "semantics"} <= labels


def test_completions_include_declared_names() -> None:
    items = _get_completions(_SIR, 0, 0)
    labels = {i.label for i in items}
    assert {"S", "I", "R", "beta", "gamma"} <= labels


def test_completions_include_builtin_symbols() -> None:
    items = _get_completions(_SIR, 0, 0)
    labels = {i.label for i in items}
    assert {"N", "t"} <= labels


def test_completions_include_math_functions() -> None:
    items = _get_completions(_SIR, 0, 0)
    labels = {i.label for i in items}
    assert "exp" in labels


def test_completions_on_broken_file_still_return_keywords() -> None:
    items = _get_completions("not valid", 0, 0)
    labels = {i.label for i in items}
    assert "model" in labels


def test_completion_kinds_are_set() -> None:
    items = _get_completions(_SIR, 0, 0)
    kw = next(i for i in items if i.label == "model")
    fn = next(i for i in items if i.label == "exp")
    sym = next(i for i in items if i.label == "N")
    assert kw.kind is lsp.CompletionItemKind.Keyword
    assert fn.kind is lsp.CompletionItemKind.Function
    assert sym.kind is lsp.CompletionItemKind.Variable


# ---------------------------------------------------------------------------
# _get_definition
# ---------------------------------------------------------------------------

_URI = "file:///test.epidsl"


def test_definition_finds_compartment() -> None:
    # _SIMPLE: line 2 (0-based) = "  flow S -> I : S"
    # first 'S' is at char 7
    flow_line = "  flow S -> I : S"
    s_char = flow_line.index("S")
    loc = _get_definition(_SIMPLE, 2, s_char, _URI)
    assert loc is not None
    # compartment S is on line 1 (0-based)
    assert loc.range.start.line == 1


def test_definition_finds_rate() -> None:
    src = "model M {\n  compartment S = 9, I = 1\n  rate beta = 0.3 [1/day]\n  flow S -> I : beta * S\n}"
    # 'beta' on flow line (line 3, 0-based)
    flow_line = "  flow S -> I : beta * S"
    beta_char = flow_line.index("beta")
    loc = _get_definition(src, 3, beta_char, _URI)
    assert loc is not None
    assert loc.range.start.line == 2  # rate beta is on line 3 (1-based) = 2 (0-based)


def test_definition_returns_none_for_undeclared_symbol() -> None:
    # X is not declared as a compartment
    broken = "model M {\n  compartment S = 10\n  flow S -> X : S\n}"
    # "  flow S -> X : S", 'X' is at char 12
    loc = _get_definition(broken, 2, 12, _URI)
    assert loc is None


def test_definition_returns_none_at_empty_position() -> None:
    loc = _get_definition(_SIMPLE, 2, 0, _URI)  # leading spaces
    assert loc is None


def test_definition_returns_none_on_parse_error() -> None:
    loc = _get_definition("not valid EpiDSL", 0, 4, _URI)
    assert loc is None


def test_definition_location_has_correct_uri() -> None:
    flow_line = "  flow S -> I : S"
    s_char = flow_line.index("S")
    loc = _get_definition(_SIMPLE, 2, s_char, _URI)
    assert loc is not None
    assert loc.uri == _URI
