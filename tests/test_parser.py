from pathlib import Path

import pytest

from epidsl.ir import BinOp, Semantics, Sym
from epidsl.parser import EpiSyntaxError, parse

EXAMPLES = Path(__file__).parent.parent / "examples"

SIR = (EXAMPLES / "sir.epidsl").read_text()


def test_parses_sir_model() -> None:
    (model,) = parse(SIR)

    assert model.name == "SIR"
    assert model.semantics == (Semantics.ODE, Semantics.GILLESPIE)
    assert [(c.name, c.initial) for c in model.compartments] == [
        ("S", 990.0),
        ("I", 10.0),
        ("R", 0.0),
    ]
    assert [(k.name, k.kind, k.value, k.unit) for k in model.constants] == [
        ("beta", "parameter", 0.3, "1/day"),
        ("gamma", "rate", 0.1, "1/day"),
    ]
    assert [(f.source, f.target) for f in model.flows] == [("S", "I"), ("I", "R")]


def test_expression_precedence_and_associativity() -> None:
    (model,) = parse("model M { compartment A, B  flow A -> B : 1 + 2 * A / B - 3 ^ 2 ^ A }")
    expr = model.flows[0].expr

    # ((1 + ((2 * A) / B)) - (3 ^ (2 ^ A)))
    assert isinstance(expr, BinOp) and expr.op == "-"
    assert isinstance(expr.left, BinOp) and expr.left.op == "+"
    product = expr.left.right
    assert isinstance(product, BinOp) and product.op == "/"
    assert isinstance(product.left, BinOp) and product.left.op == "*"
    power = expr.right
    assert isinstance(power, BinOp) and power.op == "^"
    assert isinstance(power.right, BinOp) and power.right.op == "^"


def test_semantics_defaults_to_ode_and_accepts_list() -> None:
    (default,) = parse("model M { compartment A }")
    assert default.semantics == (Semantics.ODE,)

    (listed,) = parse("model M { semantics: [ODE, Gillespie, TauLeap] compartment A }")
    assert listed.semantics == (Semantics.ODE, Semantics.GILLESPIE, Semantics.TAU_LEAP)


def test_inflow_and_outflow_have_no_source_or_target() -> None:
    (model,) = parse("model M { compartment A  flow -> A : 1  flow A -> : A }")
    assert (model.flows[0].source, model.flows[0].target) == (None, "A")
    assert (model.flows[1].source, model.flows[1].target) == ("A", None)


def test_comments_are_ignored() -> None:
    (model,) = parse("/* header\n spans lines */ model M { -- note\n compartment A -- trailing\n }")
    assert [c.name for c in model.compartments] == ["A"]


def test_identifier_starting_with_keyword_is_a_name() -> None:
    (model,) = parse("model M { compartment flowing, rated  flow flowing -> rated : flowing }")
    assert [c.name for c in model.compartments] == ["flowing", "rated"]
    expr = model.flows[0].expr
    assert expr == Sym("flowing")
    assert isinstance(expr, Sym) and (expr.line, expr.column) == (1, 63)


def test_multiple_models_in_one_file() -> None:
    models = parse("model A { compartment X } model B { compartment Y }")
    assert [m.name for m in models] == ["A", "B"]


def test_syntax_error_reports_position() -> None:
    with pytest.raises(EpiSyntaxError) as excinfo:
        parse("model M {\n  compartment S\n  flow S => I : 1\n}")
    assert excinfo.value.line == 3
    assert "3:" in str(excinfo.value)


def test_unknown_semantics_is_a_syntax_error() -> None:
    with pytest.raises(EpiSyntaxError):
        parse("model M { semantics: [Magic] compartment A }")
