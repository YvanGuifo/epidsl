import numpy as np
import pytest

from epidsl.backends import BackendResult
from epidsl.backends.ode import ODEBackend
from epidsl.backends.tau_leap import TauLeapBackend
from epidsl.ir import EpiModel, SimulationConfig
from epidsl.parser import parse

SIR = """
model SIR {
    semantics: [ODE, TauLeap]
    compartment S = 99000, I = 1000, R = 0
    rate beta = 0.3 [1/day]
    rate gamma = 0.1 [1/day]
    flow S -> I : beta * S * I / N
    flow I -> R : gamma * I
}
"""


def model(source: str) -> EpiModel:
    return parse(source)[-1]


def run(source: str, t_end: float, seed: int = 0, n_points: int = 11) -> BackendResult:
    config = SimulationConfig(t_end=t_end, n_points=n_points, seed=seed)
    return TauLeapBackend().run(model(source), config)


def final_values(source: str, name: str, t_end: float, runs: int) -> np.ndarray:
    m = model(source)
    backend = TauLeapBackend()
    return np.array(
        [
            backend.run(m, SimulationConfig(t_end=t_end, n_points=2, seed=seed)).y[name][-1]
            for seed in range(runs)
        ]
    )


def test_same_seed_gives_same_trajectory() -> None:
    a, b, c = run(SIR, 60.0, seed=1), run(SIR, 60.0, seed=1), run(SIR, 60.0, seed=2)
    assert all(np.array_equal(a.y[name], b.y[name]) for name in a.y)
    assert any(not np.array_equal(a.y[name], c.y[name]) for name in a.y)


def test_counts_are_integers_and_population_is_conserved() -> None:
    result = run(SIR, 160.0, seed=3, n_points=81)
    stacked = np.array(list(result.y.values()))

    assert np.array_equal(stacked, np.round(stacked))
    assert stacked.min() >= 0
    assert np.array_equal(stacked.sum(axis=0), np.full(81, 100000.0))
    assert [series[0] for series in result.y.values()] == [99000.0, 1000.0, 0.0]
    assert result.t[0] == 0.0 and result.t[-1] == 160.0
    assert result.metadata["backend"] == "TauLeap"
    assert result.metadata["seed"] == 3


def test_leaps_replace_most_individual_events() -> None:
    result = run(SIR, 160.0, seed=3)
    leaps, exact = result.metadata["leaps"], result.metadata["exact_steps"]
    assert isinstance(leaps, int) and isinstance(exact, int)
    # About 190 000 individual events happen in this epidemic.
    assert 0 < leaps < 20_000
    assert exact < 20_000


def mean_gap_to_ode(epsilon: float, runs: int = 30) -> float:
    """Largest gap between the mean of ``runs`` trajectories and the ODE, as a fraction of N."""
    m = model(SIR)
    reference = ODEBackend().run(m, SimulationConfig(t_end=160.0, n_points=81))
    backend = TauLeapBackend(epsilon=epsilon)
    mean = {name: np.zeros(81) for name in reference.y}
    for seed in range(runs):
        result = backend.run(m, SimulationConfig(t_end=160.0, n_points=81, seed=seed))
        for name in mean:
            mean[name] += result.y[name] / runs
    return max(float(np.abs(mean[name] - reference.y[name]).max()) for name in mean) / 100000.0


def test_mean_trajectory_converges_to_the_ode_as_epsilon_shrinks() -> None:
    # Measured on this model: about 0.6% at epsilon = 0.03 and 0.2% at 0.01.
    default, fine = mean_gap_to_ode(0.03), mean_gap_to_ode(0.01)
    assert default < 0.012
    assert fine < 0.005
    assert fine < default


def test_pure_death_process_matches_binomial_law() -> None:
    source = (
        "model D { semantics: [TauLeap] compartment X = 10000 "
        "rate k = 1 [1/day]  flow X -> : k * X }"
    )
    runs, n, p = 200, 10000, np.exp(-1.0)
    values = final_values(source, "X", 1.0, runs)

    # Leaping is approximate: allow 2% on the mean, far above the sampling error.
    assert values.mean() == pytest.approx(n * p, rel=0.02)
    assert values.var(ddof=1) == pytest.approx(n * p * (1 - p), rel=0.4)


def test_immigration_death_process_keeps_poisson_variance() -> None:
    source = (
        "model M { semantics: [TauLeap] compartment X = 2000 "
        "rate lam = 2000 [1/day] rate mu = 1 [1/day] "
        "flow -> X : lam  flow X -> : mu * X }"
    )
    values = final_values(source, "X", 10.0, 200)

    assert values.mean() == pytest.approx(2000.0, rel=0.01)
    assert values.var(ddof=1) == pytest.approx(2000.0, rel=0.35)


def test_small_population_is_simulated_exactly() -> None:
    source = (
        "model D { semantics: [TauLeap] compartment X = 50  rate k = 1 [1/day]  flow X -> : k * X }"
    )
    result = run(source, 200.0, n_points=5)
    assert result.metadata["leaps"] == 0
    assert result.metadata["exact_steps"] == 50
    assert list(result.y["X"][1:]) == [0.0, 0.0, 0.0, 0.0]

    runs, n, p = 400, 50, np.exp(-1.0)
    values = final_values(source, "X", 1.0, runs)
    assert abs(values.mean() - n * p) < 4 * np.sqrt(n * p * (1 - p) / runs)


def test_composed_model_runs() -> None:
    source = (
        SIR
        + "model Age { semantics: [TauLeap] compartment young = 40000, old = 60000 "
        + "rate aging = 0.01 [1/day]  flow young -> old : aging * young }"
        + "compose AgeSIR { models: [SIR, Age] }"
    )
    result = run(source, 100.0, seed=5)
    assert np.array_equal(sum(result.y.values()), np.full(11, 100000.0))


@pytest.mark.parametrize(
    ("source", "error", "fragment"),
    [
        ("model M { compartment X = 1  flow X -> : X }", ValueError, "TauLeap semantics"),
        (
            "model M { semantics: [TauLeap] compartment X = 1.5  flow X -> : X }",
            ValueError,
            "whole number",
        ),
        (
            "model M { semantics: [TauLeap] compartment X = 5  flow X -> : X * exp(-t) }",
            ValueError,
            "time-dependent",
        ),
        (
            "model M { semantics: [TauLeap] compartment X = 5  flow X -> : -1 * X }",
            RuntimeError,
            "negative",
        ),
        (
            "model M { semantics: [TauLeap] compartment X = 5, Y  flow X -> Y : X / Y }",
            RuntimeError,
            "could not be evaluated",
        ),
    ],
)
def test_refusals(source: str, error: type[Exception], fragment: str) -> None:
    with pytest.raises(error, match=fragment):
        run(source, 1.0)


def test_step_budget_stops_runaway_simulations() -> None:
    m = model("model G { semantics: [TauLeap] compartment X = 1  flow -> X : X * X }")
    with pytest.raises(RuntimeError, match="more than 1000 steps"):
        TauLeapBackend(max_steps=1000).run(m, SimulationConfig(t_end=100.0, seed=0))


def test_epsilon_must_be_a_fraction() -> None:
    with pytest.raises(ValueError, match="epsilon"):
        TauLeapBackend(epsilon=0.0)
    with pytest.raises(ValueError, match="epsilon"):
        TauLeapBackend(epsilon=1.0)
