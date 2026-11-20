"""Tests for D39 — heterogeneous agents via discrete attributes."""

from __future__ import annotations

import numpy as np
import pytest

from epidsl.checks import check
from epidsl.ir import SimulationConfig
from epidsl.parser import parse

# ---------------------------------------------------------------------------
# Reference models
# ---------------------------------------------------------------------------

_SIR_RISK = """
model SIR {
    semantics: [Network]
    compartment S = 980, I = 20, R = 0
    attribute risk ~ categorical(Low=0.7, High=0.3)
    rate beta_low  = 0.1   [1/contact/day]
    rate beta_high = 0.4   [1/contact/day]
    rate gamma = 0.1       [1/day]
    network watts_strogatz(k=6, rewire=0.1)
    flow S -> I : (beta_low * attr(risk, Low) + beta_high * attr(risk, High)) * contacts(I)
    flow I -> R : gamma
}
"""

_SIR_THREE_CATEGORIES = """
model SIR {
    semantics: [Network]
    compartment S = 70, I = 5, R = 5
    attribute age ~ categorical(Young=0.4, Adult=0.4, Elderly=0.2)
    rate beta = 0.3   [1/contact/day]
    rate gamma = 0.1  [1/day]
    network watts_strogatz(k=6, rewire=0.1)
    flow S -> I : beta * contacts(I)
    flow I -> R : gamma * (1.0 + attr(age, Elderly))
}
"""

_ODE_WITH_ATTR = """
model SIR {
    semantics: [ODE]
    compartment S = 990, I = 10, R = 0
    attribute risk ~ categorical(Low=0.7, High=0.3)
    rate beta = 0.3
    rate gamma = 0.1
    flow S -> I : beta * S * I
    flow I -> R : gamma * I
}
"""

_NO_NETWORK_ATTR = """
model M {
    semantics: [ODE]
    compartment S = 90, I = 10, R = 0
    rate beta = 0.3
    network watts_strogatz(k=6, rewire=0.1)
    flow S -> I : beta * attr(risk, High) * contacts(I)
    flow I -> R : 0.1
}
"""

_ATTR_WRONG_ARITY_1 = """
model M {
    semantics: [Network]
    compartment S = 90, I = 10, R = 0
    attribute risk ~ categorical(Low=0.7, High=0.3)
    rate beta = 0.3
    network complete()
    flow S -> I : beta * attr(risk)
    flow I -> R : 0.1
}
"""

_ATTR_WRONG_ARITY_3 = """
model M {
    semantics: [Network]
    compartment S = 90, I = 10, R = 0
    attribute risk ~ categorical(Low=0.7, High=0.3)
    rate beta = 0.3
    network complete()
    flow S -> I : beta * attr(risk, Low, High)
    flow I -> R : 0.1
}
"""

_ATTR_UNDECLARED = """
model M {
    semantics: [Network]
    compartment S = 90, I = 10, R = 0
    rate beta = 0.3
    network complete()
    flow S -> I : beta * attr(age, Young)
    flow I -> R : 0.1
}
"""


# ---------------------------------------------------------------------------
# 1. Parsing and IR
# ---------------------------------------------------------------------------


def test_attribute_parses() -> None:
    """model with attribute declaration must parse without error."""
    (m,) = parse(_SIR_RISK)
    assert len(m.attributes) == 1
    assert m.attributes[0].name == "risk"


def test_attribute_categories_stored() -> None:
    """categories and proportions must be stored correctly."""
    (m,) = parse(_SIR_RISK)
    cats = dict(m.attributes[0].categories)
    assert abs(cats["Low"] - 0.7) < 1e-9
    assert abs(cats["High"] - 0.3) < 1e-9


def test_attribute_three_categories_stored() -> None:
    """three-category attribute must be stored with all categories."""
    (m,) = parse(_SIR_THREE_CATEGORIES)
    cats = dict(m.attributes[0].categories)
    assert abs(cats["Young"] - 0.4) < 1e-9
    assert abs(cats["Adult"] - 0.4) < 1e-9
    assert abs(cats["Elderly"] - 0.2) < 1e-9


def test_attribute_no_errors() -> None:
    """model with valid attribute must have no check errors."""
    (m,) = parse(_SIR_RISK)
    errors = [d for d in check(m) if d.severity.value == "error"]
    assert errors == [], errors


# ---------------------------------------------------------------------------
# 2. Static checks
# ---------------------------------------------------------------------------


def test_attr_wrong_arity_1_raises() -> None:
    """attr(risk) with one argument must trigger C-A2."""
    (m,) = parse(_ATTR_WRONG_ARITY_1)
    codes = [d.code for d in check(m) if d.severity.value == "error"]
    assert "C-A2" in codes


def test_attr_wrong_arity_3_raises() -> None:
    """attr(risk, Low, High) with three arguments must trigger C-A2."""
    (m,) = parse(_ATTR_WRONG_ARITY_3)
    codes = [d.code for d in check(m) if d.severity.value == "error"]
    assert "C-A2" in codes


def test_attr_undeclared_raises() -> None:
    """attr(age, Young) without an attribute declaration must trigger C-A3."""
    (m,) = parse(_ATTR_UNDECLARED)
    codes = [d.code for d in check(m) if d.severity.value == "error"]
    assert "C-A3" in codes


# ---------------------------------------------------------------------------
# 3. NetworkBackend — attribute-based simulation
# ---------------------------------------------------------------------------


def test_attribute_backend_runs() -> None:
    """NetworkBackend must run a model with an attribute declaration."""
    from epidsl.backends.network import NetworkBackend
    (m,) = parse(_SIR_RISK)
    cfg = SimulationConfig(t_end=20.0, n_points=11, seed=42)
    result = NetworkBackend().run(m, cfg)
    assert len(result.t) == 11
    assert "S" in result.y and "I" in result.y and "R" in result.y


def test_attribute_population_conserved() -> None:
    """Population must be conserved with attribute-dependent rates."""
    from epidsl.backends.network import NetworkBackend
    (m,) = parse(_SIR_RISK)
    cfg = SimulationConfig(t_end=20.0, n_points=11, seed=42)
    result = NetworkBackend().run(m, cfg)
    total = result.y["S"] + result.y["I"] + result.y["R"]
    N = sum(c.initial for c in m.compartments)
    assert np.allclose(total, N, atol=0)


def test_attribute_seed_reproducible() -> None:
    """Same seed must give identical trajectories."""
    from epidsl.backends.network import NetworkBackend
    (m,) = parse(_SIR_RISK)
    cfg = SimulationConfig(t_end=10.0, n_points=11, seed=7)
    r1 = NetworkBackend().run(m, cfg)
    r2 = NetworkBackend().run(m, cfg)
    assert np.array_equal(r1.y["I"], r2.y["I"])


def test_high_risk_infects_faster() -> None:
    """High-risk agents (beta_high=0.4) must produce a higher epidemic peak
    than the same model where all agents use beta_low=0.1."""
    from epidsl.backends.network import NetworkBackend

    _ALL_HIGH = """
model SIR {
    semantics: [Network]
    compartment S = 90, I = 10, R = 0
    attribute risk ~ categorical(High=1.0)
    rate beta_low  = 0.1   [1/contact/day]
    rate beta_high = 0.4   [1/contact/day]
    rate gamma = 0.1       [1/day]
    network watts_strogatz(k=6, rewire=0.1)
    flow S -> I : (beta_low * attr(risk, Low) + beta_high * attr(risk, High)) * contacts(I)
    flow I -> R : gamma
}
"""
    _ALL_LOW = """
model SIR {
    semantics: [Network]
    compartment S = 90, I = 10, R = 0
    attribute risk ~ categorical(Low=1.0)
    rate beta_low  = 0.1   [1/contact/day]
    rate beta_high = 0.4   [1/contact/day]
    rate gamma = 0.1       [1/day]
    network watts_strogatz(k=6, rewire=0.1)
    flow S -> I : (beta_low * attr(risk, Low) + beta_high * attr(risk, High)) * contacts(I)
    flow I -> R : gamma
}
"""
    cfg = SimulationConfig(t_end=20.0, n_points=21, seed=42)
    (m_high,) = parse(_ALL_HIGH)
    (m_low,) = parse(_ALL_LOW)
    r_high = NetworkBackend().run(m_high, cfg)
    r_low = NetworkBackend().run(m_low, cfg)
    assert r_high.y["I"].max() > r_low.y["I"].max(), (
        f"high peak {r_high.y['I'].max()} not > low peak {r_low.y['I'].max()}"
    )


def test_three_category_runs() -> None:
    """Backend must run a model with three attribute categories."""
    from epidsl.backends.network import NetworkBackend
    (m,) = parse(_SIR_THREE_CATEGORIES)
    cfg = SimulationConfig(t_end=10.0, n_points=11, seed=42)
    result = NetworkBackend().run(m, cfg)
    assert len(result.t) == 11
