"""Tests for D31 — step(t, t0) Heaviside function in rate expressions."""

from __future__ import annotations

import numpy as np

from epidsl.backends.gillespie import GillespieBackend
from epidsl.backends.ode import ODEBackend
from epidsl.checks import check
from epidsl.expr import FUNCTIONS, compile_expr
from epidsl.ir import SimulationConfig
from epidsl.parser import parse

# ---------------------------------------------------------------------------
# Reference model: lockdown at day 30 reduces beta by 70 %.
# R0_before = 0.0003 * 1000 / 0.1 = 3  (epidemic grows)
# R0_after  = 0.00009 * 1000 / 0.1 = 0.9 (epidemic declines)
# ---------------------------------------------------------------------------

_SIR_LOCKDOWN = """
model SIR {
    semantics: [ODE, Gillespie, TauLeap]
    compartment S = 990, I = 10, R = 0
    rate beta  = 0.0003    [1/day]
    rate phi   = 0.7
    rate gamma = 0.1       [1/day]
    flow S -> I : beta * (1 - phi * step(t, 30)) * S * I
    flow I -> R : gamma * I
}
"""


# ---------------------------------------------------------------------------
# 1. Parsing and registration
# ---------------------------------------------------------------------------


def test_step_registered_in_functions() -> None:
    """step must appear in FUNCTIONS before the first parse."""
    assert "step" in FUNCTIONS
    arity, impl = FUNCTIONS["step"]
    assert arity == 2


def test_step_parses() -> None:
    """A flow using step(t, 30) must parse without error."""
    (m,) = parse(_SIR_LOCKDOWN)
    assert m.name == "SIR"


def test_step_no_check_error() -> None:
    """The static checker must not report any error for a valid step() usage."""
    (m,) = parse(_SIR_LOCKDOWN)
    diags = check(m)
    errors = [d for d in diags if d.severity.value == "error"]
    assert errors == [], errors


# ---------------------------------------------------------------------------
# 2. Semantics of step — boundary conditions (TST-BOUNDARY-001)
# ---------------------------------------------------------------------------


def test_step_zero_strictly_before_threshold() -> None:
    """step(t, t0) = 0 for t < t0."""
    _, impl = FUNCTIONS["step"]
    assert impl(29.9999, 30.0) == 0.0


def test_step_one_exactly_at_threshold() -> None:
    """step(t, t0) = 1 for t == t0 (the step is closed on the right)."""
    _, impl = FUNCTIONS["step"]
    assert impl(30.0, 30.0) == 1.0


def test_step_one_strictly_after_threshold() -> None:
    """step(t, t0) = 1 for t > t0."""
    _, impl = FUNCTIONS["step"]
    assert impl(30.0001, 30.0) == 1.0


def test_step_at_zero_threshold() -> None:
    """step(t, 0) = 1 for all t >= 0 (always-on intervention)."""
    _, impl = FUNCTIONS["step"]
    assert impl(0.0, 0.0) == 1.0
    assert impl(10.0, 0.0) == 1.0


# ---------------------------------------------------------------------------
# 3. compile_expr integration
# ---------------------------------------------------------------------------


def test_step_in_compiled_expression() -> None:
    """compile_expr must evaluate step(t, 30) correctly at several time points."""
    from epidsl.ir import Call, Num, Sym

    expr = Call("step", (Sym("t"), Num(30.0)), line=0, column=0)
    f = compile_expr(expr, {}, {})
    state = np.zeros(1)
    assert f(state, 29.0) == 0.0
    assert f(state, 30.0) == 1.0
    assert f(state, 31.0) == 1.0


# ---------------------------------------------------------------------------
# 4. ODE dynamics
# ---------------------------------------------------------------------------


def test_step_ode_runs() -> None:
    """ODE backend must simulate a model with step() without raising."""
    (m,) = parse(_SIR_LOCKDOWN)
    cfg = SimulationConfig(t_end=60.0, n_points=61)
    result = ODEBackend().run(m, cfg)
    assert len(result.t) == 61
    assert "I" in result.y


def test_step_ode_population_conserved() -> None:
    """Total population must remain constant despite the step discontinuity."""
    (m,) = parse(_SIR_LOCKDOWN)
    cfg = SimulationConfig(t_end=60.0, n_points=61)
    result = ODEBackend().run(m, cfg)
    total = result.y["S"] + result.y["I"] + result.y["R"]
    assert np.allclose(total, 1000.0, atol=1e-6)


def test_step_ode_epidemic_declines_after_lockdown() -> None:
    """With R0_after < 1, I(t) must be strictly decreasing after the lockdown.

    We compare I at t=40 vs I at t=50: a 10-day window well past the step.
    """
    (m,) = parse(_SIR_LOCKDOWN)
    cfg = SimulationConfig(t_end=60.0, n_points=61)
    result = ODEBackend().run(m, cfg)
    # t=40 is index 40, t=50 is index 50 (n_points=61 → dt=1)
    I = result.y["I"]
    assert I[40] > I[50], f"I did not decline after lockdown: I[40]={I[40]:.2f}, I[50]={I[50]:.2f}"


def test_step_ode_epidemic_grows_before_lockdown() -> None:
    """With R0_before = 3, I(t) must grow during [5, 25] before the lockdown."""
    (m,) = parse(_SIR_LOCKDOWN)
    cfg = SimulationConfig(t_end=60.0, n_points=61)
    result = ODEBackend().run(m, cfg)
    I = result.y["I"]
    assert I[25] > I[5], f"I did not grow before lockdown: I[5]={I[5]:.2f}, I[25]={I[25]:.2f}"


# ---------------------------------------------------------------------------
# 5. Stochastic backend
# ---------------------------------------------------------------------------


def test_step_gillespie_runs() -> None:
    """Gillespie backend must handle step() without error."""
    (m,) = parse(_SIR_LOCKDOWN)
    cfg = SimulationConfig(t_end=60.0, n_points=61, seed=42)
    result = GillespieBackend().run(m, cfg)
    total = result.y["S"] + result.y["I"] + result.y["R"]
    assert np.all(total >= 0)


# ---------------------------------------------------------------------------
# 6. Static checker — wrong arity
# ---------------------------------------------------------------------------


def test_step_ode_threshold_not_on_grid() -> None:
    """When t0 is not on the user grid, the ODE backend must inject it as a
    breakpoint so that LSODA crosses the discontinuity cleanly.

    Model: instantaneous switch at t=30.5 from beta=0.0003 to beta=0.00001.
    Grid: 7 points at dt=10 — t0 is NOT on the grid.
    Without the injection, the solver interpolates over the step and can
    over-estimate or under-estimate S(t=30 vs t=40).  With it, the two
    segments are solved independently and S decreases monotonically.
    """
    src = """
    model M {
        semantics: [ODE]
        compartment S = 990, I = 10, R = 0
        rate beta_hi = 0.0003  [1/day]
        rate beta_lo = 0.00001 [1/day]
        rate gamma   = 0.1     [1/day]
        flow S -> I : (beta_hi * (1 - step(t, 30.5)) + beta_lo * step(t, 30.5)) * S * I
        flow I -> R : gamma * I
    }
    """
    (m,) = parse(src)
    # Coarse grid: 0, 10, 20, 30, 40, 50, 60 — t0=30.5 falls between 30 and 40
    cfg = SimulationConfig(t_end=60.0, n_points=7)
    result = ODEBackend().run(m, cfg)
    # Output must have exactly n_points values (injected breakpoint must be trimmed)
    assert len(result.t) == 7, f"expected 7 points, got {len(result.t)}"
    # S must be strictly monotonically decreasing
    S = result.y["S"]
    assert np.all(np.diff(S) <= 0), f"S is not monotone: {S}"
    # And must actually decrease (epidemic happened before the switch)
    assert S[0] - S[-1] > 50, f"S barely changed: drop={S[0]-S[-1]:.1f}"


def test_step_wrong_arity_raises_c3() -> None:
    """step(t) with one argument must raise C3 (wrong arity)."""
    src = """
    model M {
        semantics: [ODE]
        compartment S = 100, I = 1
        rate beta = 0.001 [1/day]
        flow S -> I : beta * step(t) * S * I
    }
    """
    (m,) = parse(src)
    diags = check(m)
    assert any(d.code == "C3" for d in diags), [d.code for d in diags]
