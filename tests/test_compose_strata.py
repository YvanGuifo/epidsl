"""Stratified constants (``rate x by Model = {...}``) and entry states (``enter``)."""

import numpy as np
import pytest

from epidsl.backends import BackendResult
from epidsl.backends.ode import ODEBackend
from epidsl.checks import check
from epidsl.compose import CompositionError
from epidsl.ir import EpiModel, Expr, SimulationConfig
from epidsl.parser import parse

SIR = """
model SIR {
    compartment S = 990, I = 10, R = 0
    parameter beta = 0.3 [1/day]
    rate gamma = 0.1 [1/day]
    flow S -> I : beta * S * I / N
    flow I -> R : gamma * I
}
"""
AGE = """
model Age {
    compartment young = 300, old = 700
    rate aging = 0.002 [1/day]
    flow young -> old : aging * young
}
"""
STATIC_AGE = "model Age { compartment young = 300, old = 700 }\n"
SIRD = """
model SIRD {
    compartment S = 990, I = 10, R = 0
    rate beta = 0.5 [1/day]
    rate gamma = 0.1 [1/day]
    rate mu = 0.01 [1/day]
    flow -> S : mu * N
    flow S -> I : beta * S * I / N
    flow I -> R : gamma * I
    flow S -> : mu * S
    flow I -> : mu * I
    flow R -> : mu * R
}
"""
AGEING = """
model Age {
    compartment young = 300, old = 700
    rate aging = 0.02 [1/day]
    flow young -> old : aging * young
}
"""


def model(source: str, name: str = "M") -> EpiModel:
    return next(m for m in parse(source) if m.name == name)


def expr(text: str) -> Expr:
    (m,) = parse(f"model _ {{ flow -> X : {text} }}")
    return m.flows[0].expr


def simulate(m: EpiModel, t_end: float = 120.0) -> BackendResult:
    return ODEBackend().run(m, SimulationConfig(t_end=t_end, n_points=61))


def compose(body: str, parts: str = SIR + AGE) -> str:
    return parts + f"compose M {{ models: [SIR, Age] {body} }}"


# --- stratified constants -------------------------------------------------------------


def test_constant_is_replaced_by_one_constant_per_stratum() -> None:
    m = model(compose("rate beta by Age = { young: 0.5, old: 0.2 } [1/week]"))

    assert [(k.name, k.kind, k.value, k.unit) for k in m.constants] == [
        ("beta_young", "rate", 0.5, "1/week"),
        ("beta_old", "rate", 0.2, "1/week"),
        ("gamma", "rate", 0.1, "1/day"),
        ("aging", "rate", 0.002, "1/day"),
    ]
    flows = [(f.source, f.target, f.expr) for f in m.flows]
    assert flows[0] == ("S_young", "I_young", expr("beta_young * S_young * (I_young + I_old) / N"))
    assert flows[1] == ("S_old", "I_old", expr("beta_old * S_old * (I_young + I_old) / N"))
    assert flows[2] == ("I_young", "R_young", expr("gamma * I_young"))
    assert check(m) == []


def test_kind_is_taken_from_the_declaration_and_unit_is_inherited() -> None:
    m = model(compose("parameter gamma by Age = { old: 0.1, young: 0.2 }"))
    assert [(k.name, k.kind, k.value, k.unit) for k in m.constants][1:3] == [
        ("gamma_young", "parameter", 0.2, "1/day"),
        ("gamma_old", "parameter", 0.1, "1/day"),
    ]


def test_equal_values_change_nothing() -> None:
    plain = simulate(model(compose("")))
    stratified = simulate(model(compose("rate beta by Age = { young: 0.3, old: 0.3 }")))
    for name in plain.y:
        np.testing.assert_allclose(stratified.y[name], plain.y[name], rtol=1e-7, atol=1e-7)


def test_susceptibility_by_age_obeys_the_power_law_invariant() -> None:
    # dS_a/dt = -beta_a * S_a * I / N for each age group, with no ageing, gives
    # S_young(t) / S_young(0) = (S_old(t) / S_old(0)) ^ (beta_young / beta_old).
    source = compose("rate beta by Age = { young: 0.5, old: 0.2 }", SIR + STATIC_AGE)
    result = simulate(model(source), t_end=200.0)

    young = result.y["S_young"] / result.y["S_young"][0]
    old = result.y["S_old"] / result.y["S_old"][0]
    np.testing.assert_allclose(young, old ** (0.5 / 0.2), rtol=1e-5)
    assert young[-1] < 0.5 * old[-1]


def test_constant_of_any_model_can_depend_on_any_other_model() -> None:
    m = model(compose("rate aging by SIR = { S: 0.01, I: 0, R: 0.03 }"))
    ageing = [(f.source, f.target, f.expr) for f in m.flows[-3:]]
    assert ageing == [
        ("S_young", "S_old", expr("aging_S * S_young")),
        ("I_young", "I_old", expr("aging_I * I_young")),
        ("R_young", "R_old", expr("aging_R * R_young")),
    ]


def test_sync_rate_uses_the_stratum_of_its_source() -> None:
    source = compose(
        "rate beta by Age = { young: 0.5, old: 0.2 }  sync S -> I, young -> old : beta * S_young"
    )
    sync = model(source).flows[-1]
    assert (sync.source, sync.target) == ("S_young", "I_old")
    assert sync.expr == expr("beta_young * S_young")


# --- entry states ---------------------------------------------------------------------

DEMOGRAPHY = SIRD + AGEING + "compose M { models: [SIRD, Age]  enter Age at young }"


def test_inflow_enters_the_declared_state() -> None:
    m = model(DEMOGRAPHY)
    inflows = [f for f in m.flows if f.source is None]

    assert [(f.target, f.expr) for f in inflows] == [("S_young", expr("mu * N"))]
    assert len(m.flows) == 1 + 5 * 2 + 1 * 3
    assert check(m) == []


def test_births_enter_young_and_marginals_are_preserved() -> None:
    alone = simulate(model(DEMOGRAPHY, "SIRD"), t_end=2000.0)
    joint = simulate(model(DEMOGRAPHY), t_end=2000.0)

    for status in ("S", "I", "R"):
        marginal = joint.y[f"{status}_young"] + joint.y[f"{status}_old"]
        np.testing.assert_allclose(marginal, alone.y[status], rtol=1e-6, atol=1e-6)
    # young' = mu * N - (aging + mu) * young, so young* = mu * N / (aging + mu).
    young = sum(joint.y[f"{status}_young"] for status in ("S", "I", "R"))
    assert young[-1] == pytest.approx(0.01 * 1000 / (0.02 + 0.01), rel=1e-5)
    assert joint.y["R_young"][-1] > 0


def test_inflow_rate_uses_the_stratum_it_enters() -> None:
    m = model(
        SIRD
        + AGEING
        + "compose M { models: [SIRD, Age]  enter Age at old "
        + "rate mu by Age = { young: 0, old: 0.03 } }"
    )
    (inflow,) = [f for f in m.flows if f.source is None]
    assert (inflow.target, inflow.expr) == ("S_old", expr("mu_old * N"))


def test_inflow_of_a_later_model_enters_the_earlier_one() -> None:
    source = (
        "model Imm { compartment a = 1000  rate k = 0.005 [1/day] "
        "flow -> a : 5  flow a -> : k * a }" + SIR
    )
    m = model(source + "compose M { models: [SIR, Imm]  enter SIR at S }")
    (inflow,) = [f for f in m.flows if f.source is None]
    assert inflow.target == "S_a"


def test_every_other_model_needs_an_entry_state() -> None:
    sex = "model Sex { compartment f = 500, m = 500 }\n"
    source = SIRD + AGEING + sex + "compose M { models: [SIRD, Age, Sex]  enter Age at young {} }"

    with pytest.raises(CompositionError, match="enter Sex at"):
        parse(source.replace("{}", ""))
    m = model(source.replace("{}", "enter Sex at f"))
    assert [f.target for f in m.flows if f.source is None] == ["S_young_f"]


# --- refusals -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        ("rate nope by Age = { young: 1, old: 1 }", "no composed model declares a constant 'nope'"),
        ("rate beta by Nope = { young: 1, old: 1 }", "'Nope' is not one of the composed models"),
        ("rate beta by Age = { young: 1 }", "constant 'beta' has no value for 'old'"),
        (
            "rate beta by Age = { young: 1, old: 1, S: 1 }",
            "'S' is not a compartment of model 'Age'",
        ),
        ("rate beta by Age = { young: 1, young: 2, old: 1 }", "'young' is given more than once"),
        (
            "rate beta by Age = { young: 1, old: 1 }  rate beta by Age = { young: 1, old: 1 }",
            "'beta' is stratified more than once",
        ),
        ("enter Nope at young", "'Nope' is not one of the composed models"),
        ("enter Age at S", "'S' is not a compartment of model 'Age'"),
        ("enter Age at young  enter Age at old", "'Age' has more than one entry state"),
    ],
)
def test_refusals(body: str, fragment: str) -> None:
    with pytest.raises(CompositionError, match=fragment):
        parse(compose(body))


def test_generated_name_must_be_free() -> None:
    parts = SIR.replace("rate gamma", "rate beta_old = 1 [1/day]\n    rate gamma") + AGE
    with pytest.raises(CompositionError, match="'beta_old' is ambiguous"):
        parse(compose("rate beta by Age = { young: 1, old: 1 }", parts))


def test_inflow_without_entry_state_explains_the_fix() -> None:
    source = SIRD + AGEING + "compose M {\n  models: [SIRD, Age]\n}"
    with pytest.raises(CompositionError, match="enter Age at <compartment>") as excinfo:
        parse(source)
    line = source.splitlines()[excinfo.value.line - 1]
    assert line[excinfo.value.column - 1 :].startswith("SIRD, Age]")


def test_refusal_reports_position() -> None:
    source = compose("\n  rate beta by Age = { young: 1, oops: 2 }")
    with pytest.raises(CompositionError) as excinfo:
        parse(source)
    line = source.splitlines()[excinfo.value.line - 1]
    assert line[excinfo.value.column - 1 :].startswith("oops")
