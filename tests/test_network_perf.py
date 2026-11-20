"""D42 — Vectorised Gillespie on network: correctness and performance tests.

Baseline (pure-Python per-agent loop):
  N=200, t_end=20 : ~0.51 s
  N=500, t_end=20 : ~3.33 s

Targets after vectorisation (scipy.sparse matmul):
  N=200, t_end=20 : < 0.15 s   (> 3× speedup)
  N=500, t_end=20 : < 0.80 s   (> 4× speedup)
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from epidsl.backends.network import NetworkBackend
from epidsl.ir import SimulationConfig
from epidsl.parser import parse

# ---------------------------------------------------------------------------
# Shared model sources
# ---------------------------------------------------------------------------

_SIR_TMPL = """
model SIR {{
    semantics: [Network]
    compartment S = {s}, I = {i}, R = 0
    rate beta = 0.3 [1/contact/day]
    rate gamma = 0.1 [1/day]
    network watts_strogatz(k=6, rewire=0.1)
    flow S -> I : beta * contacts(I)
    flow I -> R : gamma
}}
"""

_SIR_ATTR_SRC = """
model SIR {
    semantics: [Network]
    compartment S = 90, I = 10, R = 0
    attribute risk ~ categorical(Low=0.7, High=0.3)
    rate beta_low  = 0.1 [1/contact/day]
    rate beta_high = 0.5 [1/contact/day]
    rate gamma = 0.1 [1/day]
    network watts_strogatz(k=6, rewire=0.1)
    flow S -> I : beta_low * neighbor_attr(I, risk, Low)
               + beta_high * neighbor_attr(I, risk, High)
    flow I -> R : gamma
}
"""


def _sir_model(n_total: int) -> object:
    s = n_total - max(1, n_total // 10)
    i = n_total - s
    return parse(_SIR_TMPL.format(s=s, i=i))[0]


def _run(model: object, t_end: float = 10.0, seed: int = 42) -> object:
    return NetworkBackend().run(model, SimulationConfig(t_end=t_end, n_points=11, seed=seed))


# ---------------------------------------------------------------------------
# Correctness: same seed → bit-identical trajectory
# ---------------------------------------------------------------------------


def test_deterministic_sir_small():
    model = _sir_model(50)
    r1 = _run(model, seed=7)
    r2 = _run(model, seed=7)
    np.testing.assert_array_equal(r1.y["S"], r2.y["S"])
    np.testing.assert_array_equal(r1.y["I"], r2.y["I"])
    np.testing.assert_array_equal(r1.y["R"], r2.y["R"])


def test_deterministic_neighbor_attr():
    model = parse(_SIR_ATTR_SRC)[0]
    cfg = SimulationConfig(t_end=10, n_points=6, seed=99)
    r1 = NetworkBackend().run(model, cfg)
    r2 = NetworkBackend().run(model, cfg)
    np.testing.assert_array_equal(r1.y["S"], r2.y["S"])
    np.testing.assert_array_equal(r1.y["I"], r2.y["I"])


def test_population_conserved_sir():
    """S+I+R must equal N at every output point."""
    model = _sir_model(80)
    r = _run(model, t_end=15, seed=42)
    total = r.y["S"] + r.y["I"] + r.y["R"]
    expected = np.full_like(total, 80.0)
    np.testing.assert_array_equal(total, expected)


def test_population_conserved_neighbor_attr():
    model = parse(_SIR_ATTR_SRC)[0]
    r = NetworkBackend().run(model, SimulationConfig(t_end=10, n_points=6, seed=42))
    total = r.y["S"] + r.y["I"] + r.y["R"]
    np.testing.assert_array_equal(total, np.full_like(total, 100.0))


# ---------------------------------------------------------------------------
# Performance: vectorised loop must be fast enough
# (These tests are RED with the pure-Python per-agent loop.)
# ---------------------------------------------------------------------------


def test_neighbor_attr_filter_is_applied():
    """neighbor_attr(I, risk, High) vs neighbor_attr(I, risk, Low) on the same population.

    With attribute risk ~ categorical(High=0.99, Low=0.01), almost all I-neighbours
    are High-risk.  So neighbor_attr(I, risk, High) produces a much faster epidemic
    than neighbor_attr(I, risk, Low).  If the filter is ignored (sabotage), both
    models would use contacts(I) and the difference would disappear.
    """
    src_high = """
model SIR {
    semantics: [Network]
    compartment S = 90, I = 10, R = 0
    attribute risk ~ categorical(High=0.99, Low=0.01)
    rate beta = 0.5 [1/contact/day]
    rate gamma = 0.1 [1/day]
    network watts_strogatz(k=6, rewire=0.1)
    flow S -> I : beta * neighbor_attr(I, risk, High)
    flow I -> R : gamma
}
"""
    src_low = """
model SIR {
    semantics: [Network]
    compartment S = 90, I = 10, R = 0
    attribute risk ~ categorical(High=0.99, Low=0.01)
    rate beta = 0.5 [1/contact/day]
    rate gamma = 0.1 [1/day]
    network watts_strogatz(k=6, rewire=0.1)
    flow S -> I : beta * neighbor_attr(I, risk, Low)
    flow I -> R : gamma
}
"""
    cfg = SimulationConfig(t_end=30, n_points=4, seed=42)
    r_high = NetworkBackend().run(parse(src_high)[0], cfg)
    r_low  = NetworkBackend().run(parse(src_low)[0],  cfg)
    # neighbour_attr(I, risk, High) finds ≈99 % of I-neighbours → epidemic spreads fast.
    # neighbour_attr(I, risk, Low)  finds ≈ 1 % of I-neighbours → near-zero spread.
    assert r_high.y["R"][-1] > r_low.y["R"][-1] + 10, (
        f"filter not working: R_high={r_high.y['R'][-1]}, R_low={r_low.y['R'][-1]}"
    )


@pytest.mark.slow
def test_perf_N200_under_015s():
    """N=200, t_end=20 must complete in < 0.15 s (baseline ≈ 0.51 s)."""
    model = _sir_model(200)
    cfg = SimulationConfig(t_end=20, n_points=11, seed=42)
    t0 = time.perf_counter()
    NetworkBackend().run(model, cfg)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.15, f"N=200 took {elapsed:.3f} s — expected < 0.15 s (vectorised)"


@pytest.mark.slow
def test_perf_N500_under_08s():
    """N=500, t_end=20 must complete in < 0.80 s (baseline ≈ 3.33 s)."""
    model = _sir_model(500)
    cfg = SimulationConfig(t_end=20, n_points=11, seed=42)
    t0 = time.perf_counter()
    NetworkBackend().run(model, cfg)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.80, f"N=500 took {elapsed:.3f} s — expected < 0.80 s (vectorised)"
