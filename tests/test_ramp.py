"""Tests for D32 — ramp(t, t0, t1) linear ramp function in rate expressions."""

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
# Reference model: progressive lockdown from day 20 to day 40.
# R0_before = 0.0003 * 1000 / 0.1 = 3.0  (epidemic grows before the ramp)
# R0_after  = 0.00009 * 1000 / 0.1 = 0.9 (epidemic declines after the ramp)
# ---------------------------------------------------------------------------

_SIR_RAMP = """
model SIR {
    semantics: [ODE, Gillespie, TauLeap]
    compartment S = 990, I = 10, R = 0
    rate beta  = 0.0003    [1/day]
    rate phi   = 0.7       [1]
    rate gamma = 0.1       [1/day]
    flow S -> I : beta * (1 - phi * ramp(t, 20, 40)) * S * I
    flow I -> R : gamma * I
}
"""


# ---------------------------------------------------------------------------
# 1. Parsing and registration
# ---------------------------------------------------------------------------


def test_ramp_registered_in_functions() -> None:
    """ramp must appear in FUNCTIONS with arity 3."""
    assert "ramp" in FUNCTIONS
    arity, impl = FUNCTIONS["ramp"]
    assert arity == 3


def test_ramp_parses() -> None:
    """A flow using ramp(t, 20, 40) must parse without error."""
    (m,) = parse(_SIR_RAMP)
    assert m.name == "SIR"


def test_ramp_no_check_error() -> None:
    """The static checker must report no errors for valid ramp() usage."""
    (m,) = parse(_SIR_RAMP)
    diags = check(m)
    errors = [d for d in diags if d.severity.value == "error"]
    assert errors == [], errors


# ---------------------------------------------------------------------------
# 2. Semantics of ramp — boundary conditions (TST-BOUNDARY-002)
# ---------------------------------------------------------------------------


def test_ramp_zero_before_start() -> None:
    """ramp(t, t0, t1) = 0 for t < t0."""
    _, impl = FUNCTIONS["ramp"]
    assert impl(19.9999, 20.0, 40.0) == 0.0


def test_ramp_zero_at_start() -> None:
    """ramp(t, t0, t1) = 0 for t == t0 (closed on the left)."""
    _, impl = FUNCTIONS["ramp"]
    assert impl(20.0, 20.0, 40.0) == 0.0


def test_ramp_one_at_end() -> None:
    """ramp(t, t0, t1) = 1 for t == t1 (closed on the right)."""
    _, impl = FUNCTIONS["ramp"]
    assert impl(40.0, 20.0, 40.0) == 1.0


def test_ramp_one_after_end() -> None:
    """ramp(t, t0, t1) = 1 for t > t1."""
    _, impl = FUNCTIONS["ramp"]
    assert impl(40.0001, 20.0, 40.0) == 1.0


def test_ramp_midpoint() -> None:
    """ramp at the midpoint must equal 0.5."""
    _, impl = FUNCTIONS["ramp"]
    assert impl(30.0, 20.0, 40.0) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 3. compile_expr integration
# ---------------------------------------------------------------------------


def test_ramp_in_compiled_expression() -> None:
    """compile_expr must evaluate ramp(t, 20, 40) correctly at several time points."""
    from epidsl.ir import Call, Num, Sym

    expr = Call("ramp", (Sym("t"), Num(20.0), Num(40.0)), line=0, column=0)
    f = compile_expr(expr, {}, {})
    state = np.zeros(1)
    assert f(state, 10.0) == 0.0
    assert f(state, 20.0) == 0.0
    assert f(state, 30.0) == pytest.approx(0.5)
    assert f(state, 40.0) == 1.0
    assert f(state, 50.0) == 1.0


# ---------------------------------------------------------------------------
# 4. ODE dynamics
# ---------------------------------------------------------------------------


def test_ramp_ode_runs() -> None:
    """ODE backend must simulate a model with ramp() without raising."""
    (m,) = parse(_SIR_RAMP)
    cfg = SimulationConfig(t_end=60.0, n_points=61)
    result = ODEBackend().run(m, cfg)
    assert len(result.t) == 61
    assert "I" in result.y


def test_ramp_ode_population_conserved() -> None:
    """Total population must remain constant despite the ramp."""
    (m,) = parse(_SIR_RAMP)
    cfg = SimulationConfig(t_end=60.0, n_points=61)
    result = ODEBackend().run(m, cfg)
    total = result.y["S"] + result.y["I"] + result.y["R"]
    assert np.allclose(total, 1000.0, atol=1e-6)


def test_ramp_ode_epidemic_declines_after_ramp() -> None:
    """With R0_after = 0.9, I(t) must decline after the ramp ends."""
    (m,) = parse(_SIR_RAMP)
    cfg = SimulationConfig(t_end=60.0, n_points=61)
    result = ODEBackend().run(m, cfg)
    I = result.y["I"]
    assert I[50] < I[40], f"I[40]={I[40]:.2f}, I[50]={I[50]:.2f}"


def test_ramp_ode_epidemic_grows_before_ramp() -> None:
    """With R0_before = 3, I(t) must grow before the ramp starts."""
    (m,) = parse(_SIR_RAMP)
    cfg = SimulationConfig(t_end=60.0, n_points=61)
    result = ODEBackend().run(m, cfg)
    I = result.y["I"]
    assert I[15] > I[5], f"I[5]={I[5]:.2f}, I[15]={I[15]:.2f}"


def test_ramp_ode_breakpoints_injected() -> None:
    """Both t0 and t1 of ramp must be injected as ODE breakpoints.

    t0=20.5 and t1=40.5 are NOT on the coarse grid (dt=10, 7 points).
    Without injection, LSODA integrates over the kinks; with it, it
    crosses them cleanly.  Either way the output must have exactly
    n_points=7 values (the extra breakpoints are trimmed).
    """
    src = """
    model M {
        semantics: [ODE]
        compartment S = 990, I = 10, R = 0
        rate beta_hi = 0.0003  [1/day]
        rate beta_lo = 0.00001 [1/day]
        rate gamma   = 0.1     [1/day]
        flow S -> I : (beta_hi * (1 - ramp(t, 20.5, 40.5)) + beta_lo * ramp(t, 20.5, 40.5)) * S * I
        flow I -> R : gamma * I
    }
    """
    (m,) = parse(src)
    cfg = SimulationConfig(t_end=60.0, n_points=7)
    result = ODEBackend().run(m, cfg)
    assert len(result.t) == 7, f"expected 7 points, got {len(result.t)}"
    S = result.y["S"]
    assert np.all(np.diff(S) <= 0), f"S is not monotone: {S}"
    assert S[0] - S[-1] > 50, f"S barely changed: drop={S[0]-S[-1]:.1f}"


# ---------------------------------------------------------------------------
# 5. Stochastic backend
# ---------------------------------------------------------------------------


def test_ramp_gillespie_runs() -> None:
    """Gillespie backend must handle ramp() without error."""
    (m,) = parse(_SIR_RAMP)
    cfg = SimulationConfig(t_end=60.0, n_points=61, seed=42)
    result = GillespieBackend().run(m, cfg)
    total = result.y["S"] + result.y["I"] + result.y["R"]
    assert np.all(total >= 0)


# ---------------------------------------------------------------------------
# 6. Static checker — wrong arity
# ---------------------------------------------------------------------------


def test_ramp_wrong_arity_raises_c3() -> None:
    """ramp(t, 20) with two arguments must raise C3 (wrong arity)."""
    src = """
    model M {
        semantics: [ODE]
        compartment S = 100, I = 1
        rate beta = 0.001 [1/day]
        flow S -> I : beta * ramp(t, 20) * S * I
    }
    """
    (m,) = parse(src)
    diags = check(m)
    assert any(d.code == "C3" for d in diags), [d.code for d in diags]
