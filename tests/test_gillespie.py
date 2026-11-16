import numpy as np
import pytest

from epidsl.backends import BackendResult
from epidsl.backends.gillespie import GillespieBackend
from epidsl.ir import EpiModel, SimulationConfig
from epidsl.parser import parse

SIR = """
model SIR {
    semantics: [ODE, Gillespie]
    compartment S = 190, I = 10, R = 0
    rate beta = 0.4 [1/day]
    rate gamma = 0.1 [1/day]
    flow S -> I : beta * S * I / N
    flow I -> R : gamma * I
}
"""


def model(source: str, name: str | None = None) -> EpiModel:
    models = parse(source)
    return models[-1] if name is None else next(m for m in models if m.name == name)


def run(source: str, t_end: float, seed: int = 0, n_points: int = 11) -> BackendResult:
    config = SimulationConfig(t_end=t_end, n_points=n_points, seed=seed)
    return GillespieBackend().run(model(source), config)


def final_values(source: str, name: str, t_end: float, runs: int) -> np.ndarray:
    m = model(source)
    backend = GillespieBackend()
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
    result = run(SIR, 200.0, seed=3, n_points=101)
    stacked = np.array(list(result.y.values()))

    assert np.array_equal(stacked, np.round(stacked))
    assert stacked.min() >= 0
    assert np.array_equal(stacked.sum(axis=0), np.full(101, 200.0))
    assert [series[0] for series in result.y.values()] == [190.0, 10.0, 0.0]
    assert result.t[0] == 0.0 and result.t[-1] == 200.0
    assert result.metadata["backend"] == "Gillespie"
    assert result.metadata["seed"] == 3
    assert isinstance(result.metadata["events"], int) and result.metadata["events"] > 0


DEATH = (
    "model D { semantics: [Gillespie] compartment X = 50  rate k = 1 [1/day]  flow X -> : k * X }"
)


def test_pure_death_process_has_binomial_law() -> None:
    # Each of the 50 individuals survives to t = 1 independently with p = exp(-1).
    runs, n, p = 400, 50, np.exp(-1.0)
    values = final_values(DEATH, "X", 1.0, runs)

    standard_error = np.sqrt(n * p * (1 - p) / runs)
    assert abs(values.mean() - n * p) < 4 * standard_error
    assert values.var(ddof=1) == pytest.approx(n * p * (1 - p), rel=0.3)


def test_immigration_death_process_reaches_poisson_stationary_law() -> None:
    source = (
        "model M { semantics: [Gillespie] compartment X = 20 "
        "rate lam = 20 [1/day] rate mu = 1 [1/day] "
        "flow -> X : lam  flow X -> : mu * X }"
    )
    runs = 400
    values = final_values(source, "X", 10.0, runs)

    # Stationary law is Poisson(lam / mu): mean and variance are both 20.
    assert abs(values.mean() - 20.0) < 4 * np.sqrt(20.0 / runs)
    assert values.var(ddof=1) == pytest.approx(20.0, rel=0.3)


def test_extinct_state_is_absorbing() -> None:
    result = run(DEATH, 200.0, n_points=5)
    assert list(result.y["X"][1:]) == [0.0, 0.0, 0.0, 0.0]


def test_flow_cannot_fire_from_an_empty_source() -> None:
    result = run("model C { semantics: [Gillespie] compartment X = 3  flow X -> : 5 }", 100.0)
    assert result.y["X"][-1] == 0.0
    assert result.metadata["events"] == 3


def test_composed_model_runs() -> None:
    source = (
        SIR
        + "model Age { semantics: [ODE, Gillespie] compartment young = 80, old = 120 "
        + "rate aging = 0.01 [1/day]  flow young -> old : aging * young }"
        + "compose AgeSIR { models: [SIR, Age] }"
    )
    result = run(source, 100.0, seed=5)
    total = sum(result.y.values())

    assert list(result.y) == ["S_young", "S_old", "I_young", "I_old", "R_young", "R_old"]
    assert np.array_equal(total, np.full(11, 200.0))


@pytest.mark.parametrize(
    ("source", "error", "fragment"),
    [
        ("model M { compartment X = 1  flow X -> : X }", ValueError, "Gillespie semantics"),
        (
            "model M { semantics: [Gillespie] compartment X = 1  flow X -> : nope }",
            ValueError,
            "C3",
        ),
        (
            "model M { semantics: [Gillespie] compartment X = 1.5  flow X -> : X }",
            ValueError,
            "whole number",
        ),
        (
            "model M { semantics: [Gillespie] compartment X = 5  flow X -> : -1 * X }",
            RuntimeError,
            "negative",
        ),
        (
            "model M { semantics: [Gillespie] compartment X = 5, Y  flow X -> Y : X / Y }",
            RuntimeError,
            "could not be evaluated",
        ),
    ],
)
def test_refusals(source: str, error: type[Exception], fragment: str) -> None:
    with pytest.raises(error, match=fragment):
        run(source, 1.0)


def test_event_budget_stops_runaway_simulations() -> None:
    m = model("model G { semantics: [Gillespie] compartment X = 1  flow -> X : X * X }")
    with pytest.raises(RuntimeError, match="more than 1000 events"):
        GillespieBackend(max_events=1000).run(m, SimulationConfig(t_end=100.0, seed=0))
