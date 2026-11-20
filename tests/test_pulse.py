"""Tests for D33 — pulse(t, t0, t1) rectangular pulse function in rate expressions."""

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
# Reference model: temporary lockdown from day 20 to day 50.
# R0_before  = 0.0003 * 1000 / 0.1 = 3.0  (grows before the pulse)
# R0_during  = 0.00003 * 1000 / 0.1 = 0.3  (declines during the lockdown)
# R0_after   = 3.0  (would grow again once lockdown ends)
# ---------------------------------------------------------------------------

_SIR_PULSE = """
model SIR {
    semantics: [ODE, Gillespie, TauLeap]
    compartment S = 990, I = 10, R = 0
    rate beta  = 0.0003    [1/day]
    rate phi   = 0.9       [1]
    rate gamma = 0.1       [1/day]
    flow S -> I : beta * (1 - phi * pulse(t, 20, 50)) * S * I
    flow I -> R : gamma * I
}
"""


# ---------------------------------------------------------------------------
# 1. Parsing and registration
# ---------------------------------------------------------------------------


def test_pulse_registered_in_functions() -> None:
    """pulse must appear in FUNCTIONS with arity 3."""
    assert "pulse" in FUNCTIONS
    arity, impl = FUNCTIONS["pulse"]
    assert arity == 3


def test_pulse_parses() -> None:
    """A flow using pulse(t, 20, 50) must parse without error."""
    (m,) = parse(_SIR_PULSE)
    assert m.name == "SIR"


def test_pulse_no_check_error() -> None:
    """The static checker must report no errors for valid pulse() usage."""
    (m,) = parse(_SIR_PULSE)
    diags = check(m)
    errors = [d for d in diags if d.severity.value == "error"]
    assert errors == [], errors


# ---------------------------------------------------------------------------
# 2. Semantics of pulse — boundary conditions (TST-BOUNDARY-003)
# ---------------------------------------------------------------------------


def test_pulse_zero_before_start() -> None:
    """pulse(t, t0, t1) = 0 for t < t0."""
    _, impl = FUNCTIONS["pulse"]
    assert impl(19.9999, 20.0, 50.0) == 0.0


def test_pulse_one_at_start() -> None:
    """pulse(t, t0, t1) = 1 for t == t0 (closed on the left)."""
    _, impl = FUNCTIONS["pulse"]
    assert impl(20.0, 20.0, 50.0) == 1.0


def test_pulse_one_during() -> None:
    """pulse(t, t0, t1) = 1 for t0 < t < t1."""
    _, impl = FUNCTIONS["pulse"]
    assert impl(35.0, 20.0, 50.0) == 1.0


def test_pulse_zero_at_end() -> None:
    """pulse(t, t0, t1) = 0 for t == t1 (open on the right)."""
    _, impl = FUNCTIONS["pulse"]
    assert impl(50.0, 20.0, 50.0) == 0.0


def test_pulse_zero_after_end() -> None:
    """pulse(t, t0, t1) = 0 for t > t1."""
    _, impl = FUNCTIONS["pulse"]
    assert impl(50.0001, 20.0, 50.0) == 0.0


# ---------------------------------------------------------------------------
# 3. compile_expr integration
# ---------------------------------------------------------------------------


def test_pulse_in_compiled_expression() -> None:
    """compile_expr must evaluate pulse(t, 20, 50) correctly at several time points."""
    from epidsl.ir import Call, Num, Sym

    expr = Call("pulse", (Sym("t"), Num(20.0), Num(50.0)), line=0, column=0)
    f = compile_expr(expr, {}, {})
    state = np.zeros(1)
    assert f(state, 10.0) == 0.0
    assert f(state, 20.0) == 1.0
    assert f(state, 35.0) == 1.0
    assert f(state, 50.0) == 0.0
    assert f(state, 60.0) == 0.0


# ---------------------------------------------------------------------------
# 4. ODE dynamics
# ---------------------------------------------------------------------------


def test_pulse_ode_runs() -> None:
    """ODE backend must simulate a model with pulse() without raising."""
    (m,) = parse(_SIR_PULSE)
    cfg = SimulationConfig(t_end=60.0, n_points=61)
    result = ODEBackend().run(m, cfg)
    assert len(result.t) == 61
    assert "I" in result.y


def test_pulse_ode_population_conserved() -> None:
    """Total population must remain constant despite the pulse discontinuities."""
    (m,) = parse(_SIR_PULSE)
    cfg = SimulationConfig(t_end=60.0, n_points=61)
    result = ODEBackend().run(m, cfg)
    total = result.y["S"] + result.y["I"] + result.y["R"]
    assert np.allclose(total, 1000.0, atol=1e-6)


def test_pulse_ode_epidemic_declines_during_pulse() -> None:
    """With R0_during = 0.3, I(t) must decline during the lockdown window."""
    (m,) = parse(_SIR_PULSE)
    cfg = SimulationConfig(t_end=60.0, n_points=61)
    result = ODEBackend().run(m, cfg)
    I = result.y["I"]
    # Compare I at t=25 and t=45, both well inside [20, 50)
    assert I[45] < I[25], f"I[25]={I[25]:.2f}, I[45]={I[45]:.2f}"


def test_pulse_ode_epidemic_grows_before_pulse() -> None:
    """With R0_before = 3, I(t) must grow before the lockdown starts."""
    (m,) = parse(_SIR_PULSE)
    cfg = SimulationConfig(t_end=60.0, n_points=61)
    result = ODEBackend().run(m, cfg)
    I = result.y["I"]
    assert I[15] > I[5], f"I[5]={I[5]:.2f}, I[15]={I[15]:.2f}"


def test_pulse_ode_breakpoints_injected() -> None:
    """Both t0 and t1 of pulse must be injected as ODE breakpoints.

    t0=20.5 and t1=49.5 are NOT on the coarse grid (dt=10, 7 points).
    The output must have exactly n_points=7 values after trimming.
    """
    src = """
    model M {
        semantics: [ODE]
        compartment S = 990, I = 10, R = 0
        rate beta_lo = 0.00003 [1/day]
        rate beta_hi = 0.0003  [1/day]
        rate gamma   = 0.1     [1/day]
        flow S -> I : (beta_lo * pulse(t, 20.5, 49.5) + beta_hi * (1 - pulse(t, 20.5, 49.5))) * S * I
        flow I -> R : gamma * I
    }
    """
    (m,) = parse(src)
    cfg = SimulationConfig(t_end=60.0, n_points=7)
    result = ODEBackend().run(m, cfg)
    assert len(result.t) == 7, f"expected 7 points, got {len(result.t)}"
    S = result.y["S"]
    assert S[0] - S[-1] > 20, f"S barely changed: drop={S[0]-S[-1]:.1f}"


# ---------------------------------------------------------------------------
# 5. Stochastic backend
# ---------------------------------------------------------------------------


def test_pulse_gillespie_runs() -> None:
    """Gillespie backend must handle pulse() without error."""
    (m,) = parse(_SIR_PULSE)
    cfg = SimulationConfig(t_end=60.0, n_points=61, seed=42)
    result = GillespieBackend().run(m, cfg)
    total = result.y["S"] + result.y["I"] + result.y["R"]
    assert np.all(total >= 0)


# ---------------------------------------------------------------------------
# 6. Static checker — wrong arity
# ---------------------------------------------------------------------------


def test_pulse_wrong_arity_raises_c3() -> None:
    """pulse(t, 20) with two arguments must raise C3 (wrong arity)."""
    src = """
    model M {
        semantics: [ODE]
        compartment S = 100, I = 1
        rate beta = 0.001 [1/day]
        flow S -> I : beta * pulse(t, 20) * S * I
    }
    """
    (m,) = parse(src)
    diags = check(m)
    assert any(d.code == "C3" for d in diags), [d.code for d in diags]
