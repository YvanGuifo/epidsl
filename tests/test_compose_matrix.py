"""Contact matrices: a constant indexed by an ordered pair of strata (``by X, X``)."""

import numpy as np
import pytest
from scipy.optimize import brentq

from epidsl.backends.gillespie import GillespieBackend
from epidsl.backends.ode import ODEBackend
from epidsl.calibration import Observations, calibrate, simulate_at, with_constants
from epidsl.checks import check
from epidsl.compose import CompositionError
from epidsl.ir import EpiModel, Expr, Prior, SimulationConfig
from epidsl.parser import parse

SIR = """
model SIR {
    semantics: [ODE, Gillespie]
    compartment S = 9900, I = 100, R = 0
    parameter beta = 0.3 prior uniform(0.05, 1) [1/day]
    rate gamma = 0.1 [1/day]
    flow S -> I : beta * S * I / N
    flow I -> R : gamma * I
}
"""
AGE = "model Age { semantics: [ODE, Gillespie] compartment young = 3000, old = 7000 }\n"
AGEING = (
    "model Age { semantics: [ODE, Gillespie] compartment young = 3000, old = 7000 "
    "rate aging = 0.001 [1/day]  flow young -> old : aging * young }\n"
)
MATRIX = (
    "parameter beta by Age, Age = "
    "{ young, young: 0.8, young, old: 0.2, old, young: 0.3, old, old: 0.5 }"
)


def compose(body: str, age: str = AGE) -> str:
    return SIR + age + f"compose M {{ models: [SIR, Age] {body} }}"


def model(source: str) -> EpiModel:
    return parse(source)[-1]


def expr(text: str) -> Expr:
    (m,) = parse(f"model _ {{ flow -> X : {text} }}")
    return m.flows[0].expr


def simulate(m: EpiModel, t_end: float = 200.0, n_points: int = 101):
    return ODEBackend().run(m, SimulationConfig(t_end=t_end, n_points=n_points))


# --- declaring a matrix ----------------------------------------------------------------


def test_one_constant_per_ordered_pair() -> None:
    m = model(compose(MATRIX + " [1/week]"))

    assert [(k.name, k.value, k.unit) for k in m.constants[:4]] == [
        ("beta_young_young", 0.8, "1/week"),
        ("beta_young_old", 0.2, "1/week"),
        ("beta_old_young", 0.3, "1/week"),
        ("beta_old_old", 0.5, "1/week"),
    ]
    assert all(k.kind == "parameter" for k in m.constants[:4])
    assert all(k.prior == Prior("uniform", 0.05, 1.0) for k in m.constants[:4])
    assert m.constants[4].name == "gamma"
    assert check(m) == []


def test_a_matrix_declared_as_a_rate_carries_no_prior() -> None:
    body = "rate beta by Age, Age = { young, young: 1, young, old: 2, old, young: 3, old, old: 4 }"
    m = model(compose(body))
    assert [(k.name, k.value, k.kind, k.prior, k.unit) for k in m.constants[:4]] == [
        ("beta_young_young", 1.0, "rate", None, "1/day"),
        ("beta_young_old", 2.0, "rate", None, "1/day"),
        ("beta_old_young", 3.0, "rate", None, "1/day"),
        ("beta_old_old", 4.0, "rate", None, "1/day"),
    ]


def test_an_explicit_prior_overrides_the_inherited_one() -> None:
    m = model(compose(MATRIX + " prior loguniform(0.01, 2)"))
    assert m.constants[0].prior == Prior("loguniform", 0.01, 2.0)


# --- what the rate becomes -------------------------------------------------------------


def test_the_rate_sums_over_the_group_met() -> None:
    flows = {(f.source, f.target): f.expr for f in model(compose(MATRIX)).flows}

    assert flows[("S_young", "I_young")] == expr(
        "beta_young_young * S_young * I_young / N + beta_young_old * S_young * I_old / N"
    )
    assert flows[("S_old", "I_old")] == expr(
        "beta_old_young * S_old * I_young / N + beta_old_old * S_old * I_old / N"
    )
    # A flow that does not use the matrix is lifted as before.
    assert flows[("I_young", "R_young")] == expr("gamma * I_young")


def test_a_group_outside_the_state_met_contributes_zero() -> None:
    # A flow of the Age model itself, whose rate names another age group: the term for
    # the state it does not name has nothing to sum, and must come out as zero.
    age = (
        "model Age { compartment young = 3000, old = 7000 "
        "rate aging = 0.001 [1/day]  flow young -> old : aging * old }\n"
    )
    body = "rate aging by Age, Age = { young, young: 1, young, old: 2, old, young: 3, old, old: 4 }"
    flows = {(f.source, f.target): f.expr for f in model(compose(body, age)).flows}

    assert flows[("S_young", "S_old")] == expr(
        "aging_young_young * 0 + aging_young_old * (S_old + I_old + R_old)"
    )


def test_the_matrix_is_used_in_a_sync_rate_too() -> None:
    body = MATRIX + "  sync S -> I, young -> old : beta * S_young * I"
    sync = model(compose(body, AGEING)).flows[-1]
    assert (sync.source, sync.target) == ("S_young", "I_old")
    assert sync.expr == expr(
        "beta_young_young * S_young * I_young + beta_young_old * S_young * I_old"
    )


# --- that the semantics is right -------------------------------------------------------


def test_equal_rows_reproduce_the_one_way_stratification() -> None:
    vector = simulate(model(compose("parameter beta by Age = { young: 0.6, old: 0.2 }")))
    matrix = simulate(
        model(
            compose(
                "parameter beta by Age, Age = "
                "{ young, young: 0.6, young, old: 0.6, old, young: 0.2, old, old: 0.2 }"
            )
        )
    )
    for name in vector.y:
        np.testing.assert_allclose(matrix.y[name], vector.y[name], rtol=1e-6, atol=1e-6)


def test_no_cross_mixing_gives_two_independent_epidemics() -> None:
    # With the off-diagonal entries at zero, each age group is a closed SIR of its own:
    # dS_a/dt = -beta_aa * S_a * I_a / N, so its reproduction number is
    # beta_aa * N_a / (gamma * N) and the final-size relation must hold within the group.
    body = (
        "rate beta by Age, Age = { young, young: 1.0, young, old: 0, old, young: 0, old, old: 0.2 }"
    )
    result = simulate(model(compose(body)), t_end=4000.0)

    n, gamma = 10000.0, 0.1
    for group, size, beta in (("young", 3000.0, 1.0), ("old", 7000.0, 0.2)):
        s = result.y[f"S_{group}"]
        i = result.y[f"I_{group}"]
        r = result.y[f"R_{group}"]
        np.testing.assert_allclose(s + i + r, size, rtol=1e-8)
        assert i[-1] < 1e-6
        r0 = beta * size / (gamma * n)

        def final_size(x: float, s0: float = s[0], r0: float = r0, size: float = size) -> float:
            return float(np.log(x / s0) + r0 * (size - x) / size)

        s_inf = brentq(final_size, 1e-9, s[0] / r0)
        assert s[-1] == pytest.approx(s_inf, rel=1e-5)


def test_who_mixes_with_whom_changes_the_outcome() -> None:
    # The same total contact per group, placed differently: within groups, or across.
    within = simulate(
        model(
            compose(
                "parameter beta by Age, Age = "
                "{ young, young: 0.5, young, old: 0.1, old, young: 0.1, old, old: 0.5 }"
            )
        ),
        t_end=400.0,
    )
    across = simulate(
        model(
            compose(
                "parameter beta by Age, Age = "
                "{ young, young: 0.1, young, old: 0.5, old, young: 0.5, old, old: 0.1 }"
            )
        ),
        t_end=400.0,
    )
    assert within.y["R_young"][-1] != pytest.approx(across.y["R_young"][-1], rel=0.02)
    # The old are the larger group, so the young are infected more when mixing is across.
    assert across.y["R_young"][-1] > within.y["R_young"][-1]


def test_a_stochastic_backend_runs_a_matrix_model() -> None:
    m = model(compose(MATRIX))
    result = GillespieBackend().run(m, SimulationConfig(t_end=60.0, n_points=7, seed=1))
    assert np.array_equal(sum(result.y.values()), np.full(7, 10000.0))
    assert result.y["R_young"][-1] > 0 and result.y["R_old"][-1] > 0


def test_calibration_brackets_the_matrix_entries() -> None:
    # Only that each 95% interval covers the truth. Four entries are weakly identified
    # from two case series: see D20, where the distance surface is measured.
    body = MATRIX + (
        "  rate rho_young = 1 [1]  rate rho_old = 1 [1] "
        "  observe cases_young = rho_young * incidence(S_young -> I_young) "
        "  observe cases_old = rho_old * incidence(S_old -> I_old) "
    )
    source = compose(body)
    truth = {
        "beta_young_young": 0.9,
        "beta_young_old": 0.15,
        "beta_old_young": 0.15,
        "beta_old_old": 0.4,
    }
    times = np.arange(7.0, 91.0, 7.0)
    backend = ODEBackend(rtol=1e-6, atol=1e-6)
    truth_run = simulate_at(backend, with_constants(model(source), truth), times, seed=None)
    observations = Observations(
        t=times, y={name: truth_run.observed[name] for name in ("cases_young", "cases_old")}
    )

    result = calibrate(backend, model(source), observations, particles=80, populations=6, seed=1)

    assert result.names == list(truth)
    for name, value in truth.items():
        assert result.quantile(0.025)[name] < value < result.quantile(0.975)[name]


# --- refusals --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        (
            "parameter beta by Age, SIR = { young, young: 1, young, old: 1,"
            " old, young: 1, old, old: 1 }",
            "a matrix is indexed by one model twice",
        ),
        (
            "parameter beta by Age, Age = { young, young: 1, old, young: 1, old, old: 1 }",
            "has no value for 'young, old'",
        ),
        (
            "parameter beta by Age, Age = { young, young: 1, young, young: 2, young, old: 1,"
            " old, young: 1, old, old: 1 }",
            "'young, young' is given more than once",
        ),
        (
            "parameter beta by Age, Age = { young, young: 1, young, nope: 1, old, young: 1,"
            " old, old: 1 }",
            "'nope' is not a compartment of model 'Age'",
        ),
        (
            "parameter beta by Age, Age = { young: 1, old: 1 }",
            "needs a pair of states",
        ),
        (
            "parameter beta by Age = { young, young: 1, old, old: 1 }",
            "takes one state",
        ),
    ],
)
def test_refusals(body: str, fragment: str) -> None:
    with pytest.raises(CompositionError, match=fragment):
        parse(compose(body))


def test_a_matrix_needs_something_to_sum_over() -> None:
    source = SIR.replace("beta * S * I / N", "beta * S")
    with pytest.raises(CompositionError, match="names no group of compartments"):
        parse(source + AGE + f"compose M {{ models: [SIR, Age] {MATRIX} }}")


def test_two_matrices_of_different_models_in_one_rate() -> None:
    sex = "model Sex { compartment f = 5000, m = 5000 }\n"
    source = SIR.replace("beta * S * I / N", "beta * gamma * S * I / N")
    body = MATRIX + ("  rate gamma by Sex, Sex = { f, f: 1, f, m: 1, m, f: 1, m, m: 1 }")
    with pytest.raises(CompositionError, match="matrices of two different models"):
        parse(source + AGE + sex + f"compose M {{ models: [SIR, Age, Sex] {body} }}")


def test_a_generated_matrix_name_must_be_free() -> None:
    clash = SIR.replace("rate gamma", "rate beta_old_young = 1 [1/day]\n    rate gamma")
    with pytest.raises(CompositionError, match="'beta_old_young' is ambiguous"):
        parse(clash + AGE + f"compose M {{ models: [SIR, Age] {MATRIX} }}")


def test_a_block_observable_cannot_use_a_matrix() -> None:
    body = MATRIX + "  observe x = beta * I"
    with pytest.raises(CompositionError, match="beta_young_young"):
        parse(compose(body))


def test_refusal_reports_position() -> None:
    source = compose(
        "\n  parameter beta by Age, Age = { young, young: 1, young, oops: 1,"
        " old, young: 1, old, old: 1 }"
    )
    with pytest.raises(CompositionError) as excinfo:
        parse(source)
    line = source.splitlines()[excinfo.value.line - 1]
    assert line[excinfo.value.column - 1 :].startswith("oops")
