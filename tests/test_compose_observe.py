"""Observables declared in a compose block, and priors on stratified parameters."""

import numpy as np
import pytest

from epidsl.backends import BackendResult
from epidsl.backends.gillespie import GillespieBackend
from epidsl.backends.ode import ODEBackend
from epidsl.calibration import Observations, calibrate, simulate_at, with_constants
from epidsl.checks import Severity, check
from epidsl.compose import CompositionError
from epidsl.ir import EpiModel, Incidence, Prior, SimulationConfig
from epidsl.parser import parse

SIR = """
model SIR {
    semantics: [ODE, Gillespie]
    compartment S = 9900, I = 100, R = 0
    parameter beta = 0.3 prior uniform(0.05, 1) [1/day]
    rate gamma = 0.1 [1/day]
    rate rho = 0.5 [1]
    flow S -> I : beta * S * I / N
    flow I -> R : gamma * I
}
"""
STATIC_AGE = "model Age { semantics: [ODE, Gillespie] compartment young = 3000, old = 7000 }\n"
AGEING = """
model Age {
    semantics: [ODE, Gillespie]
    compartment young = 3000, old = 7000
    rate aging = 0.01 [1/day]
    flow young -> old : aging * young
}
"""


def compose(body: str, age: str = STATIC_AGE) -> str:
    return SIR + age + f"compose M {{ models: [SIR, Age] {body} }}"


def model(source: str) -> EpiModel:
    return parse(source)[-1]


def simulate(source: str, n_points: int = 11) -> BackendResult:
    return ODEBackend().run(model(source), SimulationConfig(t_end=100.0, n_points=n_points))


BY_AGE = (
    "parameter beta by Age = { young: 0.5, old: 0.25 } "
    "rate rho by Age = { young: 0.8, old: 0.4 } "
    "observe cases_young = rho_young * incidence(S_young -> I_young) "
    "observe cases_old = rho_old * incidence(S_old -> I_old) "
)


# --- observables in a compose block ----------------------------------------------------


def test_observable_on_one_stratum() -> None:
    m = model(compose(BY_AGE))
    assert [o.name for o in m.observables] == ["cases_young", "cases_old"]
    assert check(m) == []

    result = simulate(compose(BY_AGE))
    for group, rho in (("young", 0.8), ("old", 0.4)):
        cases, s = result.observed[f"cases_{group}"], result.y[f"S_{group}"]
        assert cases[0] == 0.0
        # Nobody ages here, so S_young only loses people to I_young.
        np.testing.assert_allclose(cases[1:], rho * -np.diff(s), rtol=1e-6, atol=1e-6)


def test_group_names_stand_for_totals() -> None:
    body = (
        "observe share_young = I_young / young "
        "observe all_infected = I "
        "observe infections = incidence(S -> I) "
        "observe ageing = incidence(young -> old) "
    )
    m = model(compose(body, AGEING))
    assert m.observables[2].expr == Incidence(
        (
            ("S_young", "I_young"),
            ("S_young", "I_old"),
            ("S_old", "I_young"),
            ("S_old", "I_old"),
        )
    )

    result = simulate(compose(body, AGEING))
    y = result.y
    young = y["S_young"] + y["I_young"] + y["R_young"]
    np.testing.assert_allclose(result.observed["share_young"], y["I_young"] / young)
    np.testing.assert_allclose(result.observed["all_infected"], y["I_young"] + y["I_old"])
    np.testing.assert_allclose(
        result.observed["infections"][1:],
        -np.diff(y["S_young"] + y["S_old"]),
        rtol=1e-6,
        atol=1e-6,
    )
    # The population is closed, so the young group only shrinks through ageing.
    np.testing.assert_allclose(result.observed["ageing"][1:], -np.diff(young), rtol=1e-6, atol=1e-6)


def test_block_observables_follow_those_of_the_models() -> None:
    parts = SIR.replace("    flow I -> R", "    observe prevalence = I / N\n    flow I -> R")
    source = parts + STATIC_AGE + "compose M { models: [SIR, Age]  observe old_infected = I_old }"
    assert [o.name for o in model(source).observables] == ["prevalence", "old_infected"]


def test_gillespie_counts_stratum_incidence_exactly() -> None:
    config = SimulationConfig(t_end=40.0, n_points=9, seed=1)
    result = GillespieBackend().run(model(compose(BY_AGE)), config)
    np.testing.assert_allclose(
        result.observed["cases_young"][1:], 0.8 * -np.diff(result.y["S_young"])
    )


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        ("observe x = incidence(S_young -> nope)", "'nope' is not a compartment or group"),
        ("observe x = incidence(nope -> I)", "'nope' is not a compartment or group"),
        ("observe I_young = I", "the name 'I_young' is already used"),
        ("observe young = I", "the name 'young' is already used"),
        ("observe gamma = I", "the name 'gamma' is already used"),
        ("observe x = I  observe x = S", "the name 'x' is already used"),
        (
            "rate rho by Age = { young: 0.8, old: 0.4 }  observe x = rho * I",
            "uses the stratified constant 'rho'",
        ),
    ],
)
def test_refusals(body: str, fragment: str) -> None:
    with pytest.raises(CompositionError, match=fragment):
        parse(compose(body))


def test_refusal_reports_position() -> None:
    source = compose("\n  observe x = incidence(S_young -> nope)")
    with pytest.raises(CompositionError) as excinfo:
        parse(source)
    line = source.splitlines()[excinfo.value.line - 1]
    assert line[excinfo.value.column - 1 :].startswith("S_young -> nope")


def test_name_clash_with_a_model_observable() -> None:
    parts = SIR.replace("    flow I -> R", "    observe prevalence = I / N\n    flow I -> R")
    with pytest.raises(CompositionError, match="the name 'prevalence' is already used"):
        parse(parts + STATIC_AGE + "compose M { models: [SIR, Age]  observe prevalence = I }")


def test_static_checks_still_apply_to_block_observables() -> None:
    m = model(compose("\n  observe x = nope * I\n  observe y = incidence(R_young -> S_young)"))
    found = [(d.code, d.severity) for d in check(m) if d.severity is Severity.ERROR]
    assert found == [("C3", Severity.ERROR), ("C12", Severity.ERROR)]


# --- constants declared in a compose block ---------------------------------------------


def test_block_can_declare_its_own_constants() -> None:
    body = (
        "rate scale = 2 [1] "
        "parameter report_young = 0.5 prior uniform(0.1, 1) [1] "
        "observe doubled = scale * I "
        "observe cases_young = report_young * incidence(S_young -> I_young)"
    )
    m = model(compose(body))

    assert [k.name for k in m.constants] == ["beta", "gamma", "rho", "scale", "report_young"]
    assert m.constants[-1].prior == Prior("uniform", 0.1, 1.0)
    assert [d for d in check(m) if d.severity is Severity.ERROR] == []
    result = simulate(compose(body))
    np.testing.assert_allclose(
        result.observed["doubled"], 2 * (result.y["I_young"] + result.y["I_old"])
    )
    np.testing.assert_allclose(
        result.observed["cases_young"][1:], 0.5 * -np.diff(result.y["S_young"]), rtol=1e-6
    )


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        ("rate gamma = 1 [1/day]", "the name 'gamma' is already used"),
        ("rate k = 1 [1]  rate k = 2 [1]", "the name 'k' is already used"),
        (
            "rate beta by Age = { young: 1, old: 1 }  rate beta = 1 [1/day]",
            "the name 'beta' is already used",
        ),
        ("rate beta_old = 1 [1/day]  rate beta by Age = { young: 1, old: 1 }", "already used"),
        ("rate S_young = 1 [1]", "'S_young' is ambiguous"),
    ],
)
def test_block_constant_refusals(body: str, fragment: str) -> None:
    with pytest.raises(CompositionError, match=fragment):
        parse(compose(body))


# --- priors on stratified parameters ---------------------------------------------------


def test_stratified_parameter_takes_a_prior() -> None:
    m = model(compose("parameter beta by Age = { young: 0.5, old: 0.25 } prior uniform(0.1, 2)"))
    young, old = m.constants[0], m.constants[1]
    assert (young.name, young.value, young.prior) == ("beta_young", 0.5, Prior("uniform", 0.1, 2.0))
    assert (old.name, old.value, old.prior) == ("beta_old", 0.25, Prior("uniform", 0.1, 2.0))
    assert young.unit == "1/day"


def test_prior_is_inherited_by_a_stratified_parameter_only() -> None:
    inherited = model(compose("parameter beta by Age = { young: 0.5, old: 0.25 }"))
    assert [k.prior for k in inherited.constants[:2]] == [Prior("uniform", 0.05, 1.0)] * 2

    fixed = model(compose("rate beta by Age = { young: 0.5, old: 0.25 }"))
    assert [k.prior for k in fixed.constants[:2]] == [None, None]


def test_value_outside_the_prior_is_reported_on_the_composed_model() -> None:
    m = model(compose("parameter beta by Age = { young: 5, old: 0.25 }"))
    (diagnostic,) = [d for d in check(m) if d.severity is Severity.ERROR]
    assert diagnostic.code == "C11" and "beta_young" in diagnostic.message


def test_calibration_recovers_transmission_by_age_from_cases_by_age() -> None:
    source = compose(BY_AGE)
    truth = {"beta_young": 0.6, "beta_old": 0.2}
    times = np.arange(7.0, 85.0, 7.0)
    backend = ODEBackend(rtol=1e-6, atol=1e-6)
    simulated = simulate_at(backend, with_constants(model(source), truth), times, seed=None)
    observations = Observations(
        t=times, y={name: simulated.observed[name] for name in ("cases_young", "cases_old")}
    )

    result = calibrate(backend, model(source), observations, particles=80, populations=6, seed=1)

    assert result.names == ["beta_young", "beta_old"]
    for name, value in truth.items():
        assert result.quantile(0.025)[name] < value < result.quantile(0.975)[name]
        assert result.mean()[name] == pytest.approx(value, rel=0.1)
