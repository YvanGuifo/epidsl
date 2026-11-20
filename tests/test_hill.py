"""Tests for D36 — hill(x, k, n) Hill saturation function in rate expressions."""

from __future__ import annotations

import numpy as np
import pytest

from epidsl.backends.gillespie import GillespieBackend
from epidsl.backends.ode import ODEBackend
from epidsl.checks import check
from epidsl.expr import FUNCTIONS, compile_expr
from epidsl.ir import SimulationConfig
from epidsl.parser import parse

# ---------------------------------------------------------------------------
# Reference model: hospital-saturation effect on recovery.
# gamma_base = 0.1 [1/day], gamma_max = 0.2 [1/day] when uncrowded.
# hill(I, k=50, n=2) ≈ 0 for I << 50, ≈ 1 for I >> 50.
# Effective recovery: gamma_base * (1 + (1 - hill(I, 50, 2)))
# → gamma=0.2 when I=0, gamma→0.1 when I→∞ (saturation reduces care quality)
# ---------------------------------------------------------------------------

_SIR_HILL = """
model SIR {
    semantics: [ODE, Gillespie, TauLeap]
    compartment S = 990, I = 10, R = 0
    rate beta      = 0.0003  [1/day]
    rate gamma_lo  = 0.1     [1/day]
    rate k_hosp    = 50      [1]
    rate n_hill    = 2       [1]
    flow S -> I : beta * S * I
    flow I -> R : gamma_lo * (1 + (1 - hill(I, k_hosp, n_hill))) * I
}
"""


# ---------------------------------------------------------------------------
# 1. Parsing and registration
# ---------------------------------------------------------------------------


def test_hill_registered_in_functions() -> None:
    """hill must appear in FUNCTIONS with arity 3."""
    assert "hill" in FUNCTIONS
    arity, impl = FUNCTIONS["hill"]
    assert arity == 3


def test_hill_parses() -> None:
    """A flow using hill(I, k_hosp, n_hill) must parse without error."""
    (m,) = parse(_SIR_HILL)
    assert m.name == "SIR"


def test_hill_no_check_error() -> None:
    """The static checker must report no errors for valid hill() usage."""
    (m,) = parse(_SIR_HILL)
    diags = check(m)
    errors = [d for d in diags if d.severity.value == "error"]
    assert errors == [], errors


# ---------------------------------------------------------------------------
# 2. Semantics of hill — mathematical properties
# ---------------------------------------------------------------------------


def test_hill_zero_at_zero() -> None:
    """hill(0, k, n) = 0 for any k > 0, n > 0."""
    _, impl = FUNCTIONS["hill"]
    assert impl(0.0, 50.0, 2.0) == 0.0
    assert impl(0.0, 1.0, 1.0) == 0.0


def test_hill_half_at_k() -> None:
    """hill(k, k, n) = 0.5 (definition of the half-saturation constant)."""
    _, impl = FUNCTIONS["hill"]
    assert abs(impl(50.0, 50.0, 2.0) - 0.5) < 1e-12
    assert abs(impl(1.0, 1.0, 1.0) - 0.5) < 1e-12
    assert abs(impl(10.0, 10.0, 4.0) - 0.5) < 1e-12


def test_hill_approaches_one_for_large_x() -> None:
    """hill(x, k, n) → 1 as x → ∞."""
    _, impl = FUNCTIONS["hill"]
    assert impl(1e9, 50.0, 2.0) > 0.9999999


def test_hill_monotone_in_x() -> None:
    """hill must be strictly increasing in x for fixed k and n."""
    _, impl = FUNCTIONS["hill"]
    xs = np.linspace(0.1, 200.0, 200)
    vals = np.array([impl(x, 50.0, 2.0) for x in xs])
    assert np.all(np.diff(vals) > 0)


def test_hill_steeper_for_larger_n() -> None:
    """For x > k, hill(x, k, n2) > hill(x, k, n1) when n2 > n1 (steeper curve)."""
    _, impl = FUNCTIONS["hill"]
    x, k = 80.0, 50.0
    assert impl(x, k, 4.0) > impl(x, k, 2.0) > impl(x, k, 1.0)


def test_hill_bounded_in_zero_one() -> None:
    """hill must always return a value in [0, 1]."""
    _, impl = FUNCTIONS["hill"]
    for x in [0.0, 1.0, 50.0, 100.0, 1000.0]:
        v = impl(x, 50.0, 2.0)
        assert 0.0 <= v <= 1.0, f"hill({x}) = {v} outside [0, 1]"


# ---------------------------------------------------------------------------
# 3. compile_expr integration
# ---------------------------------------------------------------------------


def test_hill_in_compiled_expression() -> None:
    """compile_expr must evaluate hill(x, 50, 2) correctly at key points."""
    from epidsl.ir import Call, Num, Sym

    expr = Call("hill", (Sym("t"), Num(50.0), Num(2.0)), line=0, column=0)
    f = compile_expr(expr, {}, {})
    state = np.zeros(1)
    assert f(state, 0.0) == 0.0
    assert abs(f(state, 50.0) - 0.5) < 1e-12
    assert f(state, 1e9) > 0.9999


# ---------------------------------------------------------------------------
# 4. ODE dynamics
# ---------------------------------------------------------------------------


def test_hill_ode_runs() -> None:
    """ODE backend must simulate a model with hill() without raising."""
    (m,) = parse(_SIR_HILL)
    cfg = SimulationConfig(t_end=60.0, n_points=61)
    result = ODEBackend().run(m, cfg)
    assert len(result.t) == 61
    assert "I" in result.y


def test_hill_ode_population_conserved() -> None:
    """Total population must remain constant despite hill-modulated recovery."""
    (m,) = parse(_SIR_HILL)
    cfg = SimulationConfig(t_end=60.0, n_points=61)
    result = ODEBackend().run(m, cfg)
    total = result.y["S"] + result.y["I"] + result.y["R"]
    assert np.allclose(total, 1000.0, atol=1e-6)


def test_hill_ode_no_extra_breakpoints() -> None:
    """hill() is continuous; output length must equal n_points exactly."""
    (m,) = parse(_SIR_HILL)
    cfg = SimulationConfig(t_end=60.0, n_points=7)
    result = ODEBackend().run(m, cfg)
    assert len(result.t) == 7, f"expected 7 points, got {len(result.t)}"


# ---------------------------------------------------------------------------
# 5. Stochastic backend
# ---------------------------------------------------------------------------


def test_hill_gillespie_runs() -> None:
    """Gillespie backend must handle hill() without error."""
    (m,) = parse(_SIR_HILL)
    cfg = SimulationConfig(t_end=60.0, n_points=61, seed=42)
    result = GillespieBackend().run(m, cfg)
    total = result.y["S"] + result.y["I"] + result.y["R"]
    assert np.all(total >= 0)


# ---------------------------------------------------------------------------
# 6. Static checker — wrong arity
# ---------------------------------------------------------------------------


def test_hill_wrong_arity_raises_c3() -> None:
    """hill(x, k) with two arguments must raise C3 (wrong arity)."""
    src = """
    model M {
        semantics: [ODE]
        compartment S = 100, I = 1
        rate beta = 0.001 [1/day]
        flow S -> I : beta * hill(I, 50) * S
    }
    """
    (m,) = parse(src)
    diags = check(m)
    assert any(d.code == "C3" for d in diags), [d.code for d in diags]
