"""Tests for D34 — clamp(x, lo, hi) value-bounding function in rate expressions."""

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
# Reference model: vaccination campaign where coverage is a linear ramp
# clamped to [0, 1] so it cannot exceed 100 % or go negative.
#
# beta * (1 - clamp(vacc_rate * t, 0, 1)) * S * I
# vacc_rate = 0.05 [1/day]  → coverage reaches 1.0 at t = 20 days
# beta = 0.0003 [1/day], gamma = 0.1 [1/day], N = 1000
# ---------------------------------------------------------------------------

_SIR_CLAMP = """
model SIR {
    semantics: [ODE, Gillespie, TauLeap]
    compartment S = 990, I = 10, R = 0
    rate beta      = 0.0003  [1/day]
    rate vacc_rate = 0.05    [1/day]
    rate gamma     = 0.1     [1/day]
    flow S -> I : beta * (1 - clamp(vacc_rate * t, 0, 1)) * S * I
    flow I -> R : gamma * I
}
"""


# ---------------------------------------------------------------------------
# 1. Parsing and registration
# ---------------------------------------------------------------------------


def test_clamp_registered_in_functions() -> None:
    """clamp must appear in FUNCTIONS with arity 3."""
    assert "clamp" in FUNCTIONS
    arity, impl = FUNCTIONS["clamp"]
    assert arity == 3


def test_clamp_parses() -> None:
    """A flow using clamp(vacc_rate * t, 0, 1) must parse without error."""
    (m,) = parse(_SIR_CLAMP)
    assert m.name == "SIR"


def test_clamp_no_check_error() -> None:
    """The static checker must report no errors for valid clamp() usage."""
    (m,) = parse(_SIR_CLAMP)
    diags = check(m)
    errors = [d for d in diags if d.severity.value == "error"]
    assert errors == [], errors


# ---------------------------------------------------------------------------
# 2. Semantics of clamp — boundary conditions (TST-BOUNDARY-004)
# ---------------------------------------------------------------------------


def test_clamp_below_lo() -> None:
    """clamp(x, lo, hi) = lo for x < lo."""
    _, impl = FUNCTIONS["clamp"]
    assert impl(-5.0, 0.0, 1.0) == 0.0


def test_clamp_at_lo() -> None:
    """clamp(x, lo, hi) = lo for x == lo."""
    _, impl = FUNCTIONS["clamp"]
    assert impl(0.0, 0.0, 1.0) == 0.0


def test_clamp_inside() -> None:
    """clamp(x, lo, hi) = x for lo < x < hi."""
    _, impl = FUNCTIONS["clamp"]
    assert impl(0.5, 0.0, 1.0) == 0.5


def test_clamp_at_hi() -> None:
    """clamp(x, lo, hi) = hi for x == hi."""
    _, impl = FUNCTIONS["clamp"]
    assert impl(1.0, 0.0, 1.0) == 1.0


def test_clamp_above_hi() -> None:
    """clamp(x, lo, hi) = hi for x > hi."""
    _, impl = FUNCTIONS["clamp"]
    assert impl(2.5, 0.0, 1.0) == 1.0


def test_clamp_lo_equals_hi() -> None:
    """clamp(x, v, v) = v for any x."""
    _, impl = FUNCTIONS["clamp"]
    assert impl(-1.0, 0.5, 0.5) == 0.5
    assert impl(0.5, 0.5, 0.5) == 0.5
    assert impl(2.0, 0.5, 0.5) == 0.5


# ---------------------------------------------------------------------------
# 3. compile_expr integration
# ---------------------------------------------------------------------------


def test_clamp_in_compiled_expression() -> None:
    """compile_expr must evaluate clamp(vacc_rate*t, 0, 1) correctly at several points."""
    from epidsl.ir import BinOp, Call, Num, Sym

    # clamp(0.05 * t, 0, 1)
    inner = BinOp("*", Num(0.05), Sym("t"))
    expr = Call("clamp", (inner, Num(0.0), Num(1.0)), line=0, column=0)
    f = compile_expr(expr, {}, {})
    state = np.zeros(1)
    assert f(state, 0.0) == 0.0    # 0.05 * 0 = 0.0  → clamped to lo
    assert f(state, 10.0) == 0.5   # 0.05 * 10 = 0.5 → inside
    assert f(state, 20.0) == 1.0   # 0.05 * 20 = 1.0 → at hi
    assert f(state, 30.0) == 1.0   # 0.05 * 30 = 1.5 → clamped to hi


# ---------------------------------------------------------------------------
# 4. ODE dynamics
# ---------------------------------------------------------------------------


def test_clamp_ode_runs() -> None:
    """ODE backend must simulate a model with clamp() without raising."""
    (m,) = parse(_SIR_CLAMP)
    cfg = SimulationConfig(t_end=60.0, n_points=61)
    result = ODEBackend().run(m, cfg)
    assert len(result.t) == 61
    assert "I" in result.y


def test_clamp_ode_population_conserved() -> None:
    """Total population must remain constant despite clamped vaccination."""
    (m,) = parse(_SIR_CLAMP)
    cfg = SimulationConfig(t_end=60.0, n_points=61)
    result = ODEBackend().run(m, cfg)
    total = result.y["S"] + result.y["I"] + result.y["R"]
    assert np.allclose(total, 1000.0, atol=1e-6)


def test_clamp_ode_coverage_saturates() -> None:
    """After t=20, coverage=1 so transmission stops; I must decline for t > 20."""
    (m,) = parse(_SIR_CLAMP)
    cfg = SimulationConfig(t_end=60.0, n_points=61)
    result = ODEBackend().run(m, cfg)
    I = result.y["I"]
    # After full coverage (t > 20) the epidemic must have declined
    assert I[50] < I[20], f"I[20]={I[20]:.2f}, I[50]={I[50]:.2f}"


def test_clamp_ode_epidemic_grows_before_coverage() -> None:
    """Before vaccination ramps up (early times), I must grow."""
    (m,) = parse(_SIR_CLAMP)
    cfg = SimulationConfig(t_end=60.0, n_points=61)
    result = ODEBackend().run(m, cfg)
    I = result.y["I"]
    assert I[5] > I[0], f"I[0]={I[0]:.2f}, I[5]={I[5]:.2f}"


def test_clamp_ode_no_extra_breakpoints() -> None:
    """clamp() has no discontinuities; output length must equal n_points exactly."""
    (m,) = parse(_SIR_CLAMP)
    cfg = SimulationConfig(t_end=60.0, n_points=7)
    result = ODEBackend().run(m, cfg)
    assert len(result.t) == 7, f"expected 7 points, got {len(result.t)}"


# ---------------------------------------------------------------------------
# 5. Stochastic backend
# ---------------------------------------------------------------------------


def test_clamp_gillespie_runs() -> None:
    """Gillespie backend must handle clamp() without error."""
    (m,) = parse(_SIR_CLAMP)
    cfg = SimulationConfig(t_end=60.0, n_points=61, seed=42)
    result = GillespieBackend().run(m, cfg)
    total = result.y["S"] + result.y["I"] + result.y["R"]
    assert np.all(total >= 0)


# ---------------------------------------------------------------------------
# 6. Static checker — wrong arity
# ---------------------------------------------------------------------------


def test_clamp_wrong_arity_raises_c3() -> None:
    """clamp(x, lo) with two arguments must raise C3 (wrong arity)."""
    src = """
    model M {
        semantics: [ODE]
        compartment S = 100, I = 1
        rate beta = 0.001 [1/day]
        flow S -> I : beta * clamp(t, 0) * S * I
    }
    """
    (m,) = parse(src)
    diags = check(m)
    assert any(d.code == "C3" for d in diags), [d.code for d in diags]
