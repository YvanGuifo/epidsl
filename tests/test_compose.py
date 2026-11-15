import numpy as np
import pytest

from epidsl.backends.ode import ODEBackend
from epidsl.checks import check
from epidsl.compose import CompositionError
from epidsl.ir import EpiModel, Expr, Semantics, SimulationConfig
from epidsl.parser import EpiSyntaxError, parse

SIR = """
model SIR {
    compartment S = 990, I = 10, R = 0
    rate beta = 0.3 [1/day]
    rate gamma = 0.1 [1/day]
    flow S -> I : beta * S * I / N
    flow I -> R : gamma * I
}
"""
AGE = """
model Age {
    compartment young = 300, old = 700
    rate aging = 0.02 [1/day]
    flow young -> old : aging * young
}
"""
SEX = """
model Sex {
    compartment f = 500, m = 500
    rate swap = 0.01 [1/day]
    flow f -> m : swap * f
}
"""
AGE_SIR = SIR + AGE + "compose AgeSIR { models: [SIR, Age] }"


def model(source: str, name: str) -> EpiModel:
    return next(m for m in parse(source) if m.name == name)


def expr(text: str) -> Expr:
    (m,) = parse(f"model _ {{ flow -> X : {text} }}")
    return m.flows[0].expr


def simulate(m: EpiModel, t_end: float = 120.0):
    return ODEBackend().run(m, SimulationConfig(t_end=t_end, n_points=61))


def test_product_compartments_and_initial_values() -> None:
    composed = model(AGE_SIR, "AgeSIR")

    assert [c.name for c in composed.compartments] == [
        "S_young",
        "S_old",
        "I_young",
        "I_old",
        "R_young",
        "R_old",
    ]
    # Independent joint distribution: marginal * marginal / N.
    assert [c.initial for c in composed.compartments] == pytest.approx(
        [297.0, 693.0, 3.0, 7.0, 0.0, 0.0]
    )
    assert [k.name for k in composed.constants] == ["beta", "gamma", "aging"]


def test_local_flows_are_lifted_to_every_stratum() -> None:
    composed = model(AGE_SIR, "AgeSIR")
    flows = [(f.source, f.target, f.expr) for f in composed.flows]

    assert len(flows) == 2 * 2 + 1 * 3
    # The source symbol becomes the stratum's own cell; other names are marginal totals.
    assert flows[0] == ("S_young", "I_young", expr("beta * S_young * (I_young + I_old) / N"))
    assert flows[1] == ("S_old", "I_old", expr("beta * S_old * (I_young + I_old) / N"))
    assert flows[2] == ("I_young", "R_young", expr("gamma * I_young"))
    assert flows[4] == ("S_young", "S_old", expr("aging * S_young"))
    assert flows[6] == ("R_young", "R_old", expr("aging * R_young"))


def test_composed_model_passes_static_checks() -> None:
    assert check(model(AGE_SIR, "AgeSIR")) == []


def test_marginals_match_components_and_joint_is_a_product() -> None:
    sir = simulate(model(AGE_SIR, "SIR"))
    age = simulate(model(AGE_SIR, "Age"))
    joint = simulate(model(AGE_SIR, "AgeSIR"))

    for status in ("S", "I", "R"):
        marginal = joint.y[f"{status}_young"] + joint.y[f"{status}_old"]
        np.testing.assert_allclose(marginal, sir.y[status], rtol=1e-6, atol=1e-6)
    for group in ("young", "old"):
        marginal = sum(joint.y[f"{status}_{group}"] for status in ("S", "I", "R"))
        np.testing.assert_allclose(marginal, age.y[group], rtol=1e-6, atol=1e-6)
    # The automata are independent, so the joint stays a product of its marginals.
    np.testing.assert_allclose(
        joint.y["I_old"], sir.y["I"] * age.y["old"] / 1000.0, rtol=1e-6, atol=1e-6
    )


def test_three_way_composition_equals_nested_composition() -> None:
    source = (
        SIR
        + AGE
        + SEX
        + "compose AgeSIR { models: [SIR, Age] }"
        + "compose Direct { models: [SIR, Age, Sex] }"
        + "compose Nested { models: [AgeSIR, Sex] }"
    )
    direct, nested = model(source, "Direct"), model(source, "Nested")

    assert [c.name for c in direct.compartments] == [c.name for c in nested.compartments]
    assert len(direct.compartments) == 12
    assert direct.compartments[0].name == "S_young_f"
    assert len(direct.flows) == len(nested.flows) == 2 * 4 + 1 * 6 + 1 * 6

    a, b = simulate(direct), simulate(nested)
    for name in a.y:
        np.testing.assert_allclose(a.y[name], b.y[name], rtol=1e-6, atol=1e-6)


HOSPITAL = """
model Disease {
    compartment S = 990, I = 10, R = 0
    rate beta = 0.3 [1/day]
    rate gamma = 0.1 [1/day]
    rate severe = 0.2 [1]
    flow S -> I : (1 - severe) * beta * S * I / N
    flow I -> R : gamma * I
}
model Place {
    compartment home = 1000, hospital = 0
    rate discharge = 0.05 [1/day]
    flow hospital -> home : discharge * hospital
}
compose Hospital {
    models: [Disease, Place]
    sync S -> I, home -> hospital : severe * beta * S_home * I / N
}
"""


def test_sync_moves_both_automata_at_once() -> None:
    composed = model(HOSPITAL, "Hospital")
    sync = composed.flows[-1]

    assert (sync.source, sync.target) == ("S_home", "I_hospital")
    assert sync.expr == expr("severe * beta * S_home * (I_home + I_hospital) / N")
    assert len(composed.flows) == 2 * 2 + 1 * 3 + 1
    assert check(composed) == []


def test_sync_restores_full_force_of_infection_at_home() -> None:
    # Mild (local) plus severe (sync) infections of S_home add up to the plain SIR rate,
    # so the disease marginal differs from SIR only through the few susceptibles in hospital.
    joint = simulate(model(HOSPITAL, "Hospital"))
    total = sum(joint.y.values())
    np.testing.assert_allclose(total, 1000.0, rtol=1e-8)
    assert joint.y["I_hospital"].max() > 1.0
    assert joint.y["S_hospital"].max() == pytest.approx(0.0, abs=1e-6)


def test_sync_is_lifted_over_remaining_models() -> None:
    source = (
        HOSPITAL
        + SEX
        + "compose Three { models: [Disease, Place, Sex] "
        + ("sync S -> I, home -> hospital : severe * beta * S_home * I / N }")
    )
    syncs = model(source, "Three").flows[-2:]

    assert [(f.source, f.target) for f in syncs] == [
        ("S_home_f", "I_hospital_f"),
        ("S_home_m", "I_hospital_m"),
    ]
    infected = "(I_home_f + I_home_m + I_hospital_f + I_hospital_m)"
    assert syncs[0].expr == expr(f"severe * beta * S_home_f * {infected} / N")


def test_semantics_is_the_intersection() -> None:
    source = (
        "model A { semantics: [ODE, Gillespie, TauLeap] compartment a = 1, b  flow a -> b : a }"
        "model B { semantics: [Gillespie, ODE] compartment c = 1, d  flow c -> d : c }"
        "compose AB { models: [A, B] }"
    )
    assert model(source, "AB").semantics == (Semantics.ODE, Semantics.GILLESPIE)


A = "model A { compartment a = 4, b  rate k = 1 [1/day]  flow a -> b : k * a }\n"
B = "model B { compartment c = 4, d  rate j = 1 [1/day]  flow c -> d : j * c }\n"


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        (A + "compose X { models: [A, Missing] }", "unknown model 'Missing'"),
        (A + "compose X { models: [A] }", "at least two models"),
        (A + B + "compose X { }", "at least two models"),
        (A + "compose X { models: [A, A] }", "'a' is declared by both"),
        (
            A + "model B { compartment c = 5, d  flow c -> d : c }\ncompose X { models: [A, B] }",
            "same total population",
        ),
        (
            A + "model B { compartment c, d  flow c -> d : c }\ncompose X { models: [A, B] }",
            "same total population",
        ),
        (
            A + "model B { compartment c = 4, d  rate k = 2 [1/day]  flow c -> d : k * c }\n"
            "compose X { models: [A, B] }",
            "constant 'k' is declared by both",
        ),
        (
            A + "model B { compartment c = 4  flow -> c : 1 }\ncompose X { models: [A, B] }",
            "inflow",
        ),
        (
            A
            + "model B { compartment c = 4, d  flow c -> d : nope }\ncompose X { models: [A, B] }",
            "model 'B' has errors",
        ),
        (
            "model A { semantics: [ODE] compartment a = 1, b  flow a -> b : a }"
            "model B { semantics: [Gillespie] compartment c = 1, d  flow c -> d : c }"
            "compose X { models: [A, B] }",
            "no semantics in common",
        ),
        (A + B + "compose X { models: [A, B]  sync a -> b, c -> zz : 1 }", "'zz'"),
        (A + B + "compose X { models: [A, B]  sync a -> b, b -> a : 1 }", "one move per model"),
        (A + B + "compose X { models: [A, B]  sync a -> a, c -> d : 1 }", "same state"),
        (A + B + "compose X { models: [A, B]  sync a -> d, c -> d : 1 }", "different models"),
        (
            "model A { compartment a = 1, b_c  flow a -> b_c : a }"
            "model B { compartment c = 1, d  flow c -> d : c }"
            "model C { compartment b = 1, e  flow b -> e : b }"
            "compose X { models: [A, C, B] }",
            "ambiguous",
        ),
    ],
)
def test_composition_errors(source: str, fragment: str) -> None:
    with pytest.raises(CompositionError, match=fragment):
        parse(source)


def test_composition_error_reports_position() -> None:
    with pytest.raises(CompositionError) as excinfo:
        parse(A + "compose X {\n  models: [A, Missing]\n}")
    assert (excinfo.value.line, excinfo.value.column) == (3, 15)
    assert str(excinfo.value).startswith("3:15: composition error: ")


def test_duplicate_top_level_name_is_rejected() -> None:
    with pytest.raises(EpiSyntaxError, match="'A' is defined more than once"):
        parse(A + A)
    with pytest.raises(EpiSyntaxError, match="'A' is defined more than once"):
        parse(A + B + "compose A { models: [A, B] }")


def test_undefined_symbol_in_sync_is_reported_by_checks() -> None:
    composed = model(
        A + B + "compose X {\n models: [A, B]\n sync a -> b, c -> d : nope * a_c }", "X"
    )
    (diagnostic,) = [d for d in check(composed) if d.code == "C3"]
    assert (diagnostic.line, diagnostic.column) == (5, 24)
