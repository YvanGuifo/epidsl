"""Tests for the fast time-variable Gillespie inversion (D27).

The optimisation replaces quad+brentq (separate quadrature then root-finding,
~500–6000 rate evaluations per event) with a single solve_ivp call that
integrates the cumulative rate and fires a terminal event when it reaches the
drawn target (~20–100 evaluations per event).

TDD: these tests are written before the implementation.
  - Tests on TimedJumps.advance()          → red until advance() is added.
  - Benchmark test                          → red until advance() is called by Gillespie.
  - Statistical correctness tests           → verify correctness after the change.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from epidsl.backends import compile_model
from epidsl.backends._stochastic import TimedJumps
from epidsl.backends.gillespie import GillespieBackend
from epidsl.ir import EpiModel, Semantics, SimulationConfig
from epidsl.parser import parse

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _model(body: str) -> EpiModel:
    return parse(f"model M {{ semantics: [ODE, Gillespie, TauLeap] {body} }}")[0]


def _run(src: str, t_end: float, seed: int = 0, n_points: int = 11) -> dict:
    models = parse(src)
    cfg = SimulationConfig(t_end=t_end, n_points=n_points, seed=seed)
    return GillespieBackend().run(models[0], cfg).y


def _finals(m: EpiModel, name: str, t_end: float, runs: int) -> np.ndarray:
    return np.array(
        [
            GillespieBackend()
            .run(m, SimulationConfig(t_end=t_end, n_points=2, seed=seed))
            .y[name][-1]
            for seed in range(runs)
        ]
    )


# ---------------------------------------------------------------------------
# TimedJumps.advance() unit tests
# ---------------------------------------------------------------------------


DECAYING = _model("compartment X = 0  rate lam = 60 [1/day]  flow -> X : lam * exp(-t)")


def _make_jumps(model: EpiModel) -> tuple[TimedJumps, object]:
    compiled = compile_model(model, Semantics.GILLESPIE)
    return TimedJumps(compiled, model.name, "Gillespie"), compiled


def test_advance_method_exists() -> None:
    """TimedJumps.advance() must exist."""
    jumps, _ = _make_jumps(DECAYING)
    assert callable(getattr(jumps, "advance", None)), (
        "TimedJumps must expose an 'advance' method"
    )


def test_advance_no_jump_returns_none_and_correct_residual() -> None:
    """When the integral over [start, stop] is less than owed, advance() must return
    (None, residual) where residual = owed - integral.
    """
    jumps, compiled = _make_jumps(DECAYING)
    state = compiled.initial.copy()
    start, stop = 0.0, 0.1
    # integral of 60*exp(-t) from 0 to 0.1 ≈ 60*(1-exp(-0.1)) ≈ 5.71
    expected_integral = 60.0 * (1 - np.exp(-0.1))
    owed = expected_integral + 10.0  # much larger than integral → no jump

    t_jump, residual = jumps.advance(state, start, stop, owed)

    assert t_jump is None, f"Expected no jump, got t_jump = {t_jump}"
    assert residual == pytest.approx(owed - expected_integral, rel=1e-4), (
        f"Residual should be owed - integral ≈ {owed - expected_integral:.4f}, got {residual}"
    )


def test_advance_jump_found_returns_correct_time() -> None:
    """When owed < integral over [start, stop], advance() must return a jump time
    that is strictly inside (start, stop) and is correct within tolerance.
    """
    # constant rate lam = 10 → integral from 0 to 1 = 10; set owed = 5 → t_jump = 0.5
    m = _model("compartment X = 0  rate lam = 10 [1/day]  flow -> X : lam")
    jumps, compiled = _make_jumps(m)
    state = compiled.initial.copy()
    start, stop = 0.0, 1.0
    owed = 5.0  # exactly half the interval

    t_jump, _residual = jumps.advance(state, start, stop, owed)

    assert t_jump is not None, "Expected a jump but got None"
    assert start < t_jump < stop, f"t_jump = {t_jump} not in ({start}, {stop})"
    assert t_jump == pytest.approx(0.5, abs=1e-6), (
        f"For constant rate 10 and owed=5, t_jump should be 0.5, got {t_jump}"
    )


def test_advance_exponential_rate_jump_time() -> None:
    """advance() on a time-varying rate: 60*exp(-t), owed = 20.
    Integral from 0 to T = 60*(1-exp(-T)) = 20  →  T = -ln(1 - 20/60) = ln(3/2) ≈ 0.405.
    """
    jumps, compiled = _make_jumps(DECAYING)
    state = compiled.initial.copy()
    owed = 20.0
    expected_t = -np.log(1 - 20.0 / 60.0)  # ≈ 0.405

    t_jump, _residual = jumps.advance(state, 0.0, 2.0, owed)

    assert t_jump is not None
    assert t_jump == pytest.approx(expected_t, rel=1e-4), (
        f"Expected t_jump ≈ {expected_t:.6f}, got {t_jump}"
    )


# ---------------------------------------------------------------------------
# End-to-end statistical correctness
# ---------------------------------------------------------------------------


SEASONAL_DEATH = _model(
    "compartment X = 50  rate k = 0.4 [1/day]  flow X -> : k * (1 + sin(t)) * X"
)


def test_fast_decaying_inflow_has_poisson_law() -> None:
    """Fast Gillespie: arrivals at rate lam*exp(-t) → X ~ Poisson(lam*(1-exp(-T)))."""
    runs, expected = 300, 60.0 * (1 - np.exp(-3.0))
    values = _finals(DECAYING, "X", 3.0, runs)

    assert abs(values.mean() - expected) < 4 * np.sqrt(expected / runs)
    assert values.var(ddof=1) == pytest.approx(expected, rel=0.3)


def test_fast_seasonal_death_has_binomial_law() -> None:
    """Fast Gillespie: seasonal death → survivors ~ Binomial(n, p)."""
    runs, n, horizon = 300, 50, 4.0
    p = np.exp(-0.4 * (horizon + 1 - np.cos(horizon)))
    values = _finals(SEASONAL_DEATH, "X", horizon, runs)

    assert abs(values.mean() - n * p) < 4 * np.sqrt(n * p * (1 - p) / runs)
    assert values.var(ddof=1) == pytest.approx(n * p * (1 - p), rel=0.3)


def test_fast_same_seed_reproduces_trajectory() -> None:
    """Fast Gillespie: same seed → same trajectory."""
    cfg = SimulationConfig(t_end=4.0, n_points=9, seed=3)
    a = GillespieBackend().run(SEASONAL_DEATH, cfg)
    b = GillespieBackend().run(SEASONAL_DEATH, cfg)
    assert np.array_equal(a.y["X"], b.y["X"])


# ---------------------------------------------------------------------------
# Performance regression test
# ---------------------------------------------------------------------------


def test_seasonal_benchmark_under_three_seconds() -> None:
    """Fast Gillespie must run examples/sir_seasonal.epidsl in < 3 s.

    Baseline (quad + brentq): ~11 s.  Target (solve_ivp + event): < 3 s.
    """
    src = open("examples/sir_seasonal.epidsl").read()
    models = parse(src)
    model = models[0]
    cfg = SimulationConfig(t_end=120.0, n_points=120, seed=42)
    backend = GillespieBackend()

    # warmup
    backend.run(model, cfg)

    t0 = time.perf_counter()
    backend.run(model, cfg)
    elapsed = time.perf_counter() - t0

    assert elapsed < 3.0, (
        f"Gillespie seasonal should finish in < 3 s with fast inversion, took {elapsed:.2f} s"
    )
