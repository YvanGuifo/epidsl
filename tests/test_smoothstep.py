"""Tests for D35 — smoothstep(t, t0, t1) smooth cubic interpolation."""

from __future__ import annotations

import numpy as np

from epidsl.backends.gillespie import GillespieBackend
from epidsl.backends.ode import ODEBackend
from epidsl.checks import check
from epidsl.expr import FUNCTIONS, compile_expr
from epidsl.ir import SimulationConfig
from epidsl.parser import parse

# ---------------------------------------------------------------------------
# Reference model: lockdown that ramps up smoothly over [10, 30] then stays.
# phi=0.8, beta=0.0003, gamma=0.1, N=1000
# R0_before = 3.0, R0_after_full = 0.6
# ---------------------------------------------------------------------------

_SIR_SMOOTH = """
model SIR {
    semantics: [ODE, Gillespie, TauLeap]
    compartment S = 990, I = 10, R = 0
    rate beta  = 0.0003  [1/day]
    rate phi   = 0.8     [1]
    rate gamma = 0.1     [1/day]
    flow S -> I : beta * (1 - phi * smoothstep(t, 10, 30)) * S * I
    flow I -> R : gamma * I
}
"""


# ---------------------------------------------------------------------------
# 1. Parsing and registration
# ---------------------------------------------------------------------------


def test_smoothstep_registered_in_functions() -> None:
    """smoothstep must appear in FUNCTIONS with arity 3."""
    assert "smoothstep" in FUNCTIONS
    arity, impl = FUNCTIONS["smoothstep"]
    assert arity == 3


def test_smoothstep_parses() -> None:
    """A flow using smoothstep(t, 10, 30) must parse without error."""
    (m,) = parse(_SIR_SMOOTH)
    assert m.name == "SIR"


def test_smoothstep_no_check_error() -> None:
    """The static checker must report no errors for valid smoothstep() usage."""
    (m,) = parse(_SIR_SMOOTH)
    diags = check(m)
    errors = [d for d in diags if d.severity.value == "error"]
    assert errors == [], errors


# ---------------------------------------------------------------------------
# 2. Semantics of smoothstep — boundary and interior values
# ---------------------------------------------------------------------------


def test_smoothstep_zero_before_t0() -> None:
    """smoothstep(t, t0, t1) = 0 for t < t0."""
    _, impl = FUNCTIONS["smoothstep"]
    assert impl(5.0, 10.0, 30.0) == 0.0


def test_smoothstep_zero_at_t0() -> None:
    """smoothstep(t, t0, t1) = 0 at t == t0."""
    _, impl = FUNCTIONS["smoothstep"]
    assert impl(10.0, 10.0, 30.0) == 0.0


def test_smoothstep_one_after_t1() -> None:
    """smoothstep(t, t0, t1) = 1 for t > t1."""
    _, impl = FUNCTIONS["smoothstep"]
    assert impl(35.0, 10.0, 30.0) == 1.0


def test_smoothstep_one_at_t1() -> None:
    """smoothstep(t, t0, t1) = 1 at t == t1."""
    _, impl = FUNCTIONS["smoothstep"]
    assert impl(30.0, 10.0, 30.0) == 1.0


def test_smoothstep_midpoint() -> None:
    """smoothstep(t_mid) = 0.5 when t_mid is the midpoint of [t0, t1]."""
    _, impl = FUNCTIONS["smoothstep"]
    result = impl(20.0, 10.0, 30.0)  # s = 0.5 → 3*0.25 - 2*0.125 = 0.5
    assert abs(result - 0.5) < 1e-12


def test_smoothstep_quarter_point() -> None:
    """smoothstep at s=0.25: 3*(0.25)^2 - 2*(0.25)^3 = 0.15625."""
    _, impl = FUNCTIONS["smoothstep"]
    # t=15 on [10,30]: s=0.25
    expected = 3 * 0.25**2 - 2 * 0.25**3  # 0.15625
    result = impl(15.0, 10.0, 30.0)
    assert abs(result - expected) < 1e-12


def test_smoothstep_monotone() -> None:
    """smoothstep must be strictly increasing over (t0, t1)."""
    _, impl = FUNCTIONS["smoothstep"]
    t_vals = np.linspace(10.0, 30.0, 100)
    vals = np.array([impl(t, 10.0, 30.0) for t in t_vals])
    assert np.all(np.diff(vals) >= 0)


# ---------------------------------------------------------------------------
# 3. compile_expr integration
# ---------------------------------------------------------------------------


def test_smoothstep_in_compiled_expression() -> None:
    """compile_expr must evaluate smoothstep(t, 10, 30) correctly."""
    from epidsl.ir import Call, Num, Sym

    expr = Call("smoothstep", (Sym("t"), Num(10.0), Num(30.0)), line=0, column=0)
    f = compile_expr(expr, {}, {})
    state = np.zeros(1)
    assert f(state, 5.0) == 0.0
    assert f(state, 10.0) == 0.0
    assert abs(f(state, 20.0) - 0.5) < 1e-12
    assert f(state, 30.0) == 1.0
    assert f(state, 35.0) == 1.0


# ---------------------------------------------------------------------------
# 4. ODE dynamics
# ---------------------------------------------------------------------------


def test_smoothstep_ode_runs() -> None:
    """ODE backend must simulate a model with smoothstep() without raising."""
    (m,) = parse(_SIR_SMOOTH)
    cfg = SimulationConfig(t_end=60.0, n_points=61)
    result = ODEBackend().run(m, cfg)
    assert len(result.t) == 61
    assert "I" in result.y


def test_smoothstep_ode_population_conserved() -> None:
    """Total population must remain constant."""
    (m,) = parse(_SIR_SMOOTH)
    cfg = SimulationConfig(t_end=60.0, n_points=61)
    result = ODEBackend().run(m, cfg)
    total = result.y["S"] + result.y["I"] + result.y["R"]
    assert np.allclose(total, 1000.0, atol=1e-6)


def test_smoothstep_ode_epidemic_slows_during_ramp() -> None:
    """I must grow slower (or decline) after the smooth ramp reaches full effect."""
    (m,) = parse(_SIR_SMOOTH)
    cfg = SimulationConfig(t_end=60.0, n_points=61)
    result = ODEBackend().run(m, cfg)
    I = result.y["I"]
    # Growth rate before ramp (t=5..10) vs after full ramp (t=35..40)
    growth_before = I[10] - I[5]
    growth_after = I[40] - I[35]
    assert growth_after < growth_before, (
        f"growth_before={growth_before:.3f}, growth_after={growth_after:.3f}"
    )


def test_smoothstep_ode_no_extra_breakpoints() -> None:
    """smoothstep is C1 continuous; output length must equal n_points exactly."""
    (m,) = parse(_SIR_SMOOTH)
    cfg = SimulationConfig(t_end=60.0, n_points=7)
    result = ODEBackend().run(m, cfg)
    assert len(result.t) == 7, f"expected 7 points, got {len(result.t)}"


# ---------------------------------------------------------------------------
# 5. Stochastic backend
# ---------------------------------------------------------------------------


def test_smoothstep_gillespie_runs() -> None:
    """Gillespie backend must handle smoothstep() without error."""
    (m,) = parse(_SIR_SMOOTH)
    cfg = SimulationConfig(t_end=60.0, n_points=61, seed=42)
    result = GillespieBackend().run(m, cfg)
    total = result.y["S"] + result.y["I"] + result.y["R"]
    assert np.all(total >= 0)


# ---------------------------------------------------------------------------
# 6. Static checker — wrong arity
# ---------------------------------------------------------------------------


def test_smoothstep_wrong_arity_raises_c3() -> None:
    """smoothstep(t, 10) with two arguments must raise C3 (wrong arity)."""
    src = """
    model M {
        semantics: [ODE]
        compartment S = 100, I = 1
        rate beta = 0.001 [1/day]
        flow S -> I : beta * smoothstep(t, 10) * S * I
    }
    """
    (m,) = parse(src)
    diags = check(m)
    assert any(d.code == "C3" for d in diags), [d.code for d in diags]
