"""Rates that depend on time, in the stochastic backends."""

import numpy as np
import pytest

from epidsl.backends import Backend
from epidsl.backends.gillespie import GillespieBackend
from epidsl.backends.ode import ODEBackend
from epidsl.backends.tau_leap import TauLeapBackend
from epidsl.ir import EpiModel, SimulationConfig
from epidsl.parser import parse

STOCHASTIC: list[Backend] = [GillespieBackend(), TauLeapBackend()]
IDS = ["Gillespie", "TauLeap"]


def model(body: str) -> EpiModel:
    return parse(f"model M {{ semantics: [ODE, Gillespie, TauLeap] {body} }}")[0]


def finals(backend: Backend, m: EpiModel, name: str, t_end: float, runs: int) -> np.ndarray:
    return np.array(
        [
            backend.run(m, SimulationConfig(t_end=t_end, n_points=2, seed=seed)).y[name][-1]
            for seed in range(runs)
        ]
    )


DECAYING_INFLOW = model("compartment X = 0  rate lam = 60 [1/day]  flow -> X : lam * exp(-t)")
SEASONAL_DEATH = model("compartment X = 50  rate k = 0.4 [1/day]  flow X -> : k * (1 + sin(t)) * X")


@pytest.mark.parametrize("backend", STOCHASTIC, ids=IDS)
def test_decaying_inflow_has_the_poisson_law(backend: Backend) -> None:
    # Arrivals at rate lam * exp(-t): the count at T is Poisson(lam * (1 - exp(-T))).
    runs, expected = 300, 60 * (1 - np.exp(-3.0))
    values = finals(backend, DECAYING_INFLOW, "X", 3.0, runs)

    assert abs(values.mean() - expected) < 4 * np.sqrt(expected / runs)
    assert values.var(ddof=1) == pytest.approx(expected, rel=0.3)


def test_output_times_between_jumps_do_not_disturb_the_law() -> None:
    # Many output times and few jumps: the waiting time is carried across each output
    # time, and the count must still be Poisson(lam * (1 - exp(-T))).
    m = model("compartment X = 0  rate lam = 20 [1/day]  flow -> X : lam * exp(-t)")
    runs, expected = 300, 20 * (1 - np.exp(-3.0))
    values = np.array(
        [
            GillespieBackend()
            .run(m, SimulationConfig(t_end=3.0, n_points=121, seed=seed))
            .y["X"][-1]
            for seed in range(runs)
        ]
    )
    assert abs(values.mean() - expected) < 4 * np.sqrt(expected / runs)
    assert values.var(ddof=1) == pytest.approx(expected, rel=0.3)


def test_tau_leap_tracks_a_fast_changing_rate() -> None:
    # Large enough to leap, and a rate that falls by a factor of 3000 over the run.
    # Freezing the rate at the start of each leap overestimated this count by 5%.
    m = model("compartment X = 0  rate lam = 50000 [1/day]  flow -> X : lam * exp(-t)")
    results = [
        TauLeapBackend().run(m, SimulationConfig(t_end=8.0, n_points=3, seed=seed))
        for seed in range(8)
    ]
    mean = np.mean([r.y["X"][-1] for r in results])

    assert all(r.metadata["leaps"] > 10 for r in results)
    assert mean == pytest.approx(50000 * (1 - np.exp(-8.0)), rel=0.005)


@pytest.mark.parametrize("backend", STOCHASTIC, ids=IDS)
def test_seasonal_death_has_the_binomial_law(backend: Backend) -> None:
    # Each individual survives to T with probability exp(-k * (T + 1 - cos(T))).
    runs, n, horizon = 300, 50, 4.0
    p = np.exp(-0.4 * (horizon + 1 - np.cos(horizon)))
    values = finals(backend, SEASONAL_DEATH, "X", horizon, runs)

    assert abs(values.mean() - n * p) < 4 * np.sqrt(n * p * (1 - p) / runs)
    assert values.var(ddof=1) == pytest.approx(n * p * (1 - p), rel=0.3)


@pytest.mark.parametrize("backend", STOCHASTIC, ids=IDS)
def test_a_rate_that_starts_at_zero_wakes_up(backend: Backend) -> None:
    # Nothing can happen before t = 5; afterwards arrivals come at rate 10 * (t - 5).
    m = model("compartment X = 0  rate a = 10 [1/day]  flow -> X : a * max(0, t - 5)")
    results = [
        backend.run(m, SimulationConfig(t_end=10.0, n_points=3, seed=seed)) for seed in range(100)
    ]

    assert all(r.y["X"][1] == 0.0 for r in results)
    mean = np.mean([r.y["X"][-1] for r in results])
    assert abs(mean - 125.0) < 4 * np.sqrt(125.0 / 100)


@pytest.mark.parametrize("backend", STOCHASTIC, ids=IDS)
def test_same_seed_gives_same_trajectory(backend: Backend) -> None:
    config = SimulationConfig(t_end=4.0, n_points=9, seed=3)
    a, b = backend.run(SEASONAL_DEATH, config), backend.run(SEASONAL_DEATH, config)
    assert np.array_equal(a.y["X"], b.y["X"])
    assert np.array_equal(a.y["X"], np.round(a.y["X"])) and a.y["X"].min() >= 0


SEASONAL_SIR = model(
    "compartment S = 99000, I = 1000, R = 0 "
    "rate beta = 0.3 [1/day] rate gamma = 0.1 [1/day] "
    "flow S -> I : beta * (1 + 0.6 * sin(t / 8)) * S * I / N "
    "flow I -> R : gamma * I "
    "observe cases = incidence(S -> I)"
)


def test_tau_leap_follows_the_ode_under_seasonal_forcing() -> None:
    config = SimulationConfig(t_end=160.0, n_points=81)
    reference = ODEBackend().run(SEASONAL_SIR, config)
    mean = {name: np.zeros(81) for name in reference.y}
    runs = 20
    for seed in range(runs):
        result = TauLeapBackend().run(
            SEASONAL_SIR, SimulationConfig(t_end=160.0, n_points=81, seed=seed)
        )
        for name in mean:
            mean[name] += result.y[name] / runs
    gap = max(float(np.abs(mean[name] - reference.y[name]).max()) for name in mean) / 100000.0

    assert gap < 0.012
    assert result.metadata["leaps"] > 0
    # Incidence is counted in leaps too.
    np.testing.assert_allclose(result.observed["cases"][1:], -np.diff(result.y["S"]))


def test_gillespie_follows_the_ode_under_seasonal_forcing() -> None:
    m = model(
        "compartment S = 570, I = 30, R = 0 "
        "rate beta = 0.3 [1/day] rate gamma = 0.1 [1/day] "
        "flow S -> I : beta * (1 + 0.6 * sin(t / 8)) * S * I / N "
        "flow I -> R : gamma * I"
    )
    reference = ODEBackend().run(m, SimulationConfig(t_end=60.0, n_points=7))
    runs = 16
    mean = np.zeros(7)
    for seed in range(runs):
        mean += (
            GillespieBackend().run(m, SimulationConfig(t_end=60.0, n_points=7, seed=seed)).y["R"]
            / runs
        )
    assert np.abs(mean - reference.y["R"]).max() / 600.0 < 0.05


@pytest.mark.parametrize("backend", STOCHASTIC, ids=IDS)
def test_a_rate_that_turns_negative_is_reported(backend: Backend) -> None:
    m = model("compartment X = 500  rate k = 1 [1/day]  flow X -> : k * (1 - t) * X")
    with pytest.raises(RuntimeError, match="negative"):
        backend.run(m, SimulationConfig(t_end=3.0, n_points=4, seed=1))


def test_time_independent_models_keep_their_seeded_results() -> None:
    # The exact code path for models without `t` is unchanged, draw for draw.
    m = model(
        "compartment S = 190, I = 10, R = 0  rate b = 0.4 [1/day] rate g = 0.1 [1/day] "
        "flow S -> I : b * S * I / N  flow I -> R : g * I"
    )
    config = SimulationConfig(t_end=30.0, n_points=4, seed=11)
    assert GillespieBackend().run(m, config).y["R"].tolist() == [0.0, 37.0, 109.0, 164.0]
    assert TauLeapBackend().run(m, config).y["R"].tolist() == [0.0, 37.0, 109.0, 165.0]
