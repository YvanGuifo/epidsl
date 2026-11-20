"""Tests for D29 — reporting delay on observables (observe X = expr delay(d))."""

from __future__ import annotations

import numpy as np
import pytest

from epidsl.backends.ode import ODEBackend
from epidsl.ir import SimulationConfig
from epidsl.parser import parse

# ---------------------------------------------------------------------------
# Shared DSL snippets
# ---------------------------------------------------------------------------

_SIR_ODE = """
model SIR {
    semantics: [ODE]
    compartment S = 990, I = 10, R = 0
    rate beta  = 0.0003 [1/day]
    rate gamma = 0.1    [1/day]
    flow S -> I : beta * S * I
    flow I -> R : gamma * I
    observe cases = incidence(S -> I)
}
"""

_SIR_DELAY_7 = """
model SIR {
    semantics: [ODE]
    compartment S = 990, I = 10, R = 0
    rate beta  = 0.0003 [1/day]
    rate gamma = 0.1    [1/day]
    flow S -> I : beta * S * I
    flow I -> R : gamma * I
    observe cases = incidence(S -> I) delay(7)
}
"""

_SIR_DELAY_0 = """
model SIR {
    semantics: [ODE]
    compartment S = 990, I = 10, R = 0
    rate beta  = 0.0003 [1/day]
    rate gamma = 0.1    [1/day]
    flow S -> I : beta * S * I
    flow I -> R : gamma * I
    observe cases = incidence(S -> I) delay(0)
}
"""

_SIR_DELAY_AND_NOISE = """
model SIR {
    semantics: [ODE]
    compartment S = 990, I = 10, R = 0
    rate beta  = 0.0003 [1/day]
    rate gamma = 0.1    [1/day]
    flow S -> I : beta * S * I
    flow I -> R : gamma * I
    observe cases = incidence(S -> I) ~ poisson() delay(3)
}
"""

_SIR_DELAY_NEG = """
model SIR {
    semantics: [ODE]
    compartment S = 990, I = 10, R = 0
    rate beta  = 0.0003 [1/day]
    rate gamma = 0.1    [1/day]
    flow S -> I : beta * S * I
    flow I -> R : gamma * I
    observe cases = incidence(S -> I) delay(-1)
}
"""

_SIR_DELAY_LARGE = """
model SIR {
    semantics: [ODE]
    compartment S = 990, I = 10, R = 0
    rate beta  = 0.0003 [1/day]
    rate gamma = 0.1    [1/day]
    flow S -> I : beta * S * I
    flow I -> R : gamma * I
    observe cases = incidence(S -> I) delay(999)
}
"""


# ---------------------------------------------------------------------------
# Parsing tests
# ---------------------------------------------------------------------------


def test_delay_parses() -> None:
    """delay(7) is accepted and stored in observable.delay."""
    (m,) = [x for x in parse(_SIR_DELAY_7) if x.name == "SIR"]
    (obs,) = m.observables
    assert obs.delay == 7.0


def test_delay_default_zero() -> None:
    """With no delay clause, observable.delay defaults to 0.0."""
    (m,) = [x for x in parse(_SIR_ODE) if x.name == "SIR"]
    (obs,) = m.observables
    assert obs.delay == 0.0


def test_delay_zero_explicit_parses() -> None:
    """delay(0) is valid and sets delay = 0.0."""
    (m,) = [x for x in parse(_SIR_DELAY_0) if x.name == "SIR"]
    (obs,) = m.observables
    assert obs.delay == 0.0


def test_delay_with_noise_parses() -> None:
    """delay clause is accepted alongside a noise distribution."""
    (m,) = [x for x in parse(_SIR_DELAY_AND_NOISE) if x.name == "SIR"]
    (obs,) = m.observables
    assert obs.delay == 3.0
    assert obs.noise is not None
    assert obs.noise.dist == "poisson"


# ---------------------------------------------------------------------------
# Check C14 — negative delay
# ---------------------------------------------------------------------------


def test_negative_delay_c14() -> None:
    """C14: a negative delay value must produce a C14 error diagnostic."""
    from epidsl.checks import check

    (m,) = [x for x in parse(_SIR_DELAY_NEG) if x.name == "SIR"]
    diags = check(m)
    assert any(d.code == "C14" for d in diags), (
        f"Expected C14 diagnostic for negative delay, got: {[d.code for d in diags]}"
    )


# ---------------------------------------------------------------------------
# Backend: shifted output
# ---------------------------------------------------------------------------


def test_delay_zero_identical_to_no_delay() -> None:
    """delay(0) produces the same output as no delay clause."""
    (m_ref,) = [x for x in parse(_SIR_ODE) if x.name == "SIR"]
    (m_zero,) = [x for x in parse(_SIR_DELAY_0) if x.name == "SIR"]
    cfg = SimulationConfig(t_end=30.0, n_points=31)
    r_ref = ODEBackend().run(m_ref, cfg)
    r_zero = ODEBackend().run(m_zero, cfg)
    assert np.allclose(r_ref.observed["cases"], r_zero.observed["cases"])


def test_delay_shifts_output_right() -> None:
    """With delay(d), observed[k] should equal the undelayed value at t[k] - d.

    Concretely: the first d-worth of time steps must be 0, and the rest must
    match the undelayed series shifted by that many steps.
    """
    (m_ref,) = [x for x in parse(_SIR_ODE) if x.name == "SIR"]
    (m_del,) = [x for x in parse(_SIR_DELAY_7) if x.name == "SIR"]
    # Daily time grid: dt = 1 day, delay = 7 → 7 leading zeros
    cfg = SimulationConfig(t_end=30.0, n_points=31)
    r_ref = ODEBackend().run(m_ref, cfg)
    r_del = ODEBackend().run(m_del, cfg)
    delayed = r_del.observed["cases"]
    undelayed = r_ref.observed["cases"]
    # First 7 steps must be 0 (no information before t = 0)
    assert np.all(delayed[:7] == 0.0), f"Leading zeros missing: {delayed[:7]}"
    # Steps 7..end must match undelayed[0..end-7]
    assert np.allclose(delayed[7:], undelayed[:-7], atol=1e-6), (
        "Shifted tail does not match the undelayed series"
    )


def test_delay_large_all_zeros() -> None:
    """When delay >= t_end, all observed values should be 0."""
    (m,) = [x for x in parse(_SIR_DELAY_LARGE) if x.name == "SIR"]
    cfg = SimulationConfig(t_end=30.0, n_points=31)
    result = ODEBackend().run(m, cfg)
    assert np.all(result.observed["cases"] == 0.0)


def test_delay_with_noise_runs() -> None:
    """delay + noise is accepted and produces a non-negative output."""
    (m,) = [x for x in parse(_SIR_DELAY_AND_NOISE) if x.name == "SIR"]
    cfg = SimulationConfig(t_end=30.0, n_points=31, seed=0)
    result = ODEBackend().run(m, cfg)
    # ODE backend ignores noise → deterministic mean, same shift rules apply
    assert "cases" in result.observed
    assert len(result.observed["cases"]) == 31
