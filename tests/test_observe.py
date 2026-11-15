"""Observables (``observe``), incidence and explicit output times."""

from pathlib import Path

import numpy as np
import pytest

from epidsl.backends import Backend
from epidsl.backends.gillespie import GillespieBackend
from epidsl.backends.ode import ODEBackend
from epidsl.backends.tau_leap import TauLeapBackend
from epidsl.calibration import Observations, calibrate, distance, simulate_at, with_constants
from epidsl.checks import Severity, check
from epidsl.cli import main
from epidsl.compose import CompositionError
from epidsl.ensemble import run_ensemble
from epidsl.ir import EpiModel, Incidence, SimulationConfig
from epidsl.parser import parse

SIR = """
model SIR {
    semantics: [ODE, Gillespie, TauLeap]
    compartment S = 990, I = 10, R = 0
    parameter beta = 0.3 prior uniform(0.1, 1) [1/day]
    rate gamma = 0.1 [1/day]
    parameter rho = 0.5 prior uniform(0.1, 1) [1]
    flow S -> I : beta * S * I / N
    flow I -> R : gamma * I
    observe cases = rho * incidence(S -> I)
    observe ever_infected = I + R
}
"""
BACKENDS: list[Backend] = [ODEBackend(), GillespieBackend(), TauLeapBackend()]


def model(source: str, name: str | None = None) -> EpiModel:
    models = parse(source)
    return models[-1] if name is None else next(m for m in models if m.name == name)


def body(statements: str) -> EpiModel:
    return model(f"model M {{ {statements} }}")


# --- language --------------------------------------------------------------------------


def test_observe_is_parsed() -> None:
    cases, ever = model(SIR).observables
    assert (cases.name, ever.name) == ("cases", "ever_infected")
    incidences = [n for n in _nodes(cases.expr) if isinstance(n, Incidence)]
    assert incidences == [Incidence((("S", "I"),))]
    assert check(model(SIR)) == []


def _nodes(expr: object) -> list[object]:
    from epidsl.expr import walk

    return list(walk(expr))  # type: ignore[arg-type]


def test_incidence_of_inflow_and_outflow_is_parsed() -> None:
    m = body(
        "compartment X = 5  flow -> X : 1  flow X -> : X  "
        "observe births = incidence(-> X)  observe deaths = incidence(X ->)"
    )
    assert m.observables[0].expr == Incidence(((None, "X"),))
    assert m.observables[1].expr == Incidence((("X", None),))
    assert check(m) == []


CLEAN = "compartment S = 9, I = 1  rate b = 0.1 [1/day]  flow S -> I : b * S * I / N  "


@pytest.mark.parametrize(
    ("statement", "code", "fragment"),
    [
        ("observe x = incidence(S -> Z)", "C2", "'Z' is not a compartment"),
        ("observe x = incidence(I -> S)", "C12", "no flow goes from I to S"),
        ("observe x = incidence(-> S)", "C12", "no flow goes into S"),
        ("observe x = nope * I", "C3", "undefined symbol 'nope'"),
        ("observe S = I", "C1", "'S' is declared more than once"),
        ("observe x = I  observe x = S", "C1", "'x' is declared more than once"),
        ("observe N = I", "C8", "'N' is a reserved name"),
        ("observe x = I  observe y = x", "C3", "undefined symbol 'x'"),
    ],
)
def test_observe_checks(statement: str, code: str, fragment: str) -> None:
    found = [d for d in check(body(CLEAN + statement)) if d.severity is Severity.ERROR]
    assert [(d.code) for d in found] == [code]
    assert fragment in found[0].message


def test_incidence_is_refused_in_a_flow_rate() -> None:
    m = body("compartment S = 9, I = 1  flow S -> I : incidence(S -> I)")
    assert [(d.code, d.severity) for d in check(m)] == [("C12", Severity.ERROR)]


def test_constant_used_only_by_an_observable_is_not_unused() -> None:
    m = body(CLEAN + "rate rho = 0.5 [1]  observe x = rho * I")
    assert check(m) == []


# --- simulation ------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: type(b).__name__)
def test_incidence_counts_what_left_the_source(backend: Backend) -> None:
    result = backend.run(model(SIR), SimulationConfig(t_end=100.0, n_points=21, seed=1))
    s, cases = result.y["S"], result.observed["cases"]

    assert list(result.observed) == ["cases", "ever_infected"]
    assert cases[0] == 0.0
    # S only loses people through S -> I, and half of them are reported.
    np.testing.assert_allclose(cases[1:], 0.5 * -np.diff(s), rtol=1e-6, atol=1e-6)
    assert cases.sum() == pytest.approx(0.5 * (s[0] - s[-1]), rel=1e-6)
    np.testing.assert_allclose(result.observed["ever_infected"], result.y["I"] + result.y["R"])


@pytest.mark.parametrize("backend", BACKENDS[1:], ids=lambda b: type(b).__name__)
def test_inflow_and_outflow_incidence_balance_the_population(backend: Backend) -> None:
    m = model(
        "model M { semantics: [Gillespie, TauLeap] compartment X = 5000 "
        "rate lam = 5000 [1/day] rate mu = 1 [1/day] "
        "flow -> X : lam  flow X -> : mu * X "
        "observe arrivals = incidence(-> X)  observe departures = incidence(X ->) }"
    )
    result = backend.run(m, SimulationConfig(t_end=5.0, n_points=6, seed=2))
    net = result.observed["arrivals"] - result.observed["departures"]

    assert np.array_equal(net[1:], np.diff(result.y["X"]))
    assert result.observed["arrivals"][1:].min() > 4000


def test_flows_with_the_same_endpoints_are_summed() -> None:
    m = body(
        "compartment A = 100, B  rate k = 0.1 [1/day] "
        "flow A -> B : k * A  flow A -> B : 2 * k * A  observe moved = incidence(A -> B)"
    )
    result = ODEBackend().run(m, SimulationConfig(t_end=10.0, n_points=6))
    np.testing.assert_allclose(result.observed["moved"][1:], -np.diff(result.y["A"]), rtol=1e-7)


def test_observable_that_cannot_be_evaluated_is_reported() -> None:
    m = body("compartment A = 1, B  flow A -> B : A  observe ratio = A / B")
    with pytest.raises(RuntimeError, match="observable 'ratio' could not be evaluated"):
        ODEBackend().run(m, SimulationConfig(t_end=1.0, n_points=3))


# --- explicit output times -------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: type(b).__name__)
def test_output_at_given_times(backend: Backend) -> None:
    times = [0.0, 1.0, 5.0, 20.0, 21.0]
    result = backend.run(model(SIR), SimulationConfig.at(times, seed=3))

    assert list(result.t) == times
    assert [series[0] for series in result.y.values()] == [990.0, 10.0, 0.0]
    # Incidence is per interval, whatever its length.
    np.testing.assert_allclose(
        result.observed["cases"][1:], 0.5 * -np.diff(result.y["S"]), rtol=1e-6, atol=1e-6
    )


def test_ode_at_given_times_matches_the_closed_form() -> None:
    m = body("compartment X = 100  rate k = 0.5 [1/day]  flow X -> : k * X")
    result = ODEBackend().run(m, SimulationConfig.at([0.0, 0.3, 2.0, 7.5]))
    np.testing.assert_allclose(result.y["X"], 100 * np.exp(-0.5 * result.t), rtol=1e-6)


def test_times_must_increase() -> None:
    with pytest.raises(ValueError, match="increasing"):
        SimulationConfig.at([0.0, 2.0, 1.0])
    with pytest.raises(ValueError, match="at least two"):
        SimulationConfig.at([0.0])


def test_simulate_at_counts_incidence_from_time_zero() -> None:
    # The first observation is at t = 10: its incidence covers [0, 10].
    aligned = simulate_at(ODEBackend(), model(SIR), np.array([10.0, 20.0]), seed=None)
    full = ODEBackend().run(model(SIR), SimulationConfig.at([0.0, 10.0, 20.0]))

    assert list(aligned.t) == [10.0, 20.0]
    np.testing.assert_allclose(aligned.observed["cases"], full.observed["cases"][1:])
    np.testing.assert_allclose(aligned.y["S"], full.y["S"][1:])
    assert aligned.observed["cases"][0] > 0


# --- composition, ensembles ------------------------------------------------------------

AGE = (
    "model Age { semantics: [ODE, Gillespie, TauLeap] compartment young = 300, old = 700 "
    "rate aging = 0.01 [1/day]  flow young -> old : aging * young }"
)


def test_observables_are_lifted_to_the_composed_model() -> None:
    source = SIR + AGE + "compose M { models: [SIR, Age] }"
    alone = ODEBackend().run(model(source, "SIR"), SimulationConfig(t_end=100.0, n_points=11))
    joint = ODEBackend().run(model(source), SimulationConfig(t_end=100.0, n_points=11))

    assert list(joint.observed) == ["cases", "ever_infected"]
    assert check(model(source)) == []
    for name in ("cases", "ever_infected"):
        np.testing.assert_allclose(joint.observed[name], alone.observed[name], rtol=1e-6, atol=1e-6)


def test_lifted_incidence_includes_synchronised_transitions() -> None:
    source = (
        "model Disease { compartment S = 990, I = 10, R = 0 "
        "rate beta = 0.3 [1/day] rate gamma = 0.1 [1/day] rate severe = 0.2 [1] "
        "flow S -> I : (1 - severe) * beta * S * I / N  flow I -> R : gamma * I "
        "observe infections = incidence(S -> I) }"
        "model Place { compartment home = 1000, hospital = 0  rate out = 0.05 [1/day] "
        "flow hospital -> home : out * hospital }"
        "compose M { models: [Disease, Place] "
        "sync S -> I, home -> hospital : severe * beta * S_home * I / N }"
    )
    result = ODEBackend().run(model(source), SimulationConfig(t_end=100.0, n_points=11))
    susceptible = result.y["S_home"] + result.y["S_hospital"]
    np.testing.assert_allclose(
        result.observed["infections"][1:], -np.diff(susceptible), rtol=1e-6, atol=1e-6
    )


def test_composition_refusals_for_observables() -> None:
    with pytest.raises(CompositionError, match="observable 'cases' uses the stratified constant"):
        parse(
            SIR + AGE + "compose M { models: [SIR, Age] "
            "rate rho by Age = { young: 0.5, old: 0.9 } }"
        )
    other = "model B { compartment c = 1000, d  flow c -> d : c  observe cases = c }"
    with pytest.raises(CompositionError, match="'cases' is declared by both"):
        parse(SIR + other + "compose M { models: [SIR, B] }")


def test_ensemble_summarises_observables() -> None:
    ensemble = run_ensemble(
        GillespieBackend(), model(SIR), SimulationConfig(t_end=50.0, n_points=6, seed=0), runs=10
    )
    assert ensemble.names == ["S", "I", "R", "cases", "ever_infected"]
    mean = ensemble.mean()
    assert mean["cases"][0] == 0.0 and mean["cases"][1:].min() > 0
    np.testing.assert_allclose(mean["ever_infected"], mean["I"] + mean["R"])


# --- calibration and command line ------------------------------------------------------


def weekly_cases(truth: dict[str, float]) -> Observations:
    times = np.arange(7.0, 99.0, 7.0)
    result = simulate_at(ODEBackend(), with_constants(model(SIR), truth), times, seed=None)
    return Observations(t=times, y={"cases": result.observed["cases"]})


def test_distance_needs_matching_times_for_observables() -> None:
    observations = weekly_cases({"beta": 0.3, "rho": 0.4})
    aligned = simulate_at(ODEBackend(), model(SIR), observations.t, seed=None)
    coarse = ODEBackend().run(model(SIR), SimulationConfig(t_end=98.0, n_points=50))

    assert distance(aligned, observations) > 0
    with pytest.raises(ValueError, match="same times"):
        distance(coarse, observations)


def test_calibration_recovers_transmission_and_reporting_from_weekly_cases() -> None:
    truth = {"beta": 0.3, "rho": 0.4}
    result = calibrate(
        ODEBackend(rtol=1e-6, atol=1e-6),
        model(SIR),
        weekly_cases(truth),
        particles=120,
        populations=7,
        seed=1,
    )
    assert result.names == ["beta", "rho"]
    for name, value in truth.items():
        assert result.quantile(0.025)[name] < value < result.quantile(0.975)[name]
        assert result.mean()[name] == pytest.approx(value, rel=0.1)


def test_cli_run_writes_observable_columns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "sir.epidsl"
    source.write_text(SIR)
    assert main(["run", str(source), "--t-end", "20", "--points", "3"]) == 0
    rows = [line.split(",") for line in capsys.readouterr().out.splitlines()]

    assert rows[0] == ["t", "S", "I", "R", "cases", "ever_infected"]
    assert float(rows[1][4]) == 0.0
    assert float(rows[2][4]) == pytest.approx(0.5 * (float(rows[1][1]) - float(rows[2][1])))


def test_cli_calibrates_on_an_observable_column(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source, data = tmp_path / "sir.epidsl", tmp_path / "cases.csv"
    source.write_text(SIR)
    observations = weekly_cases({"beta": 0.3, "rho": 0.4})
    pairs = zip(observations.t, observations.y["cases"], strict=True)
    lines = ["t,cases", *(f"{t},{c}" for t, c in pairs)]
    data.write_text("\n".join(lines) + "\n")

    args = ["calibrate", str(source), "--data", str(data), "--particles", "80"]
    assert main([*args, "--populations", "6", "--seed", "2"]) == 0
    table = [line.split(",") for line in capsys.readouterr().out.splitlines()]
    estimates = {row[0]: float(row[1]) for row in table[1:]}
    assert estimates["beta"] == pytest.approx(0.3, rel=0.1)
    assert estimates["rho"] == pytest.approx(0.4, rel=0.15)

    data.write_text("t,nope\n7,1\n14,2\n")
    assert main(args) == 1
    assert "'nope' is not a compartment or observable" in capsys.readouterr().err
