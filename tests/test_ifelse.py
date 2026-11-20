"""Tests for D37 — ifelse(cond, a, b) conditional expression in rate expressions."""

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
# Reference model: time-triggered transmission change.
# Before t=30: beta_hi (high transmission); after t=30: beta_lo (lockdown).
# ifelse(t - 30, beta_lo, beta_hi) selects the right rate.
# (time-based condition avoids state-dependent ODE discontinuity)
# ---------------------------------------------------------------------------

_SIR_IFELSE = """
model SIR {
    semantics: [ODE, Gillespie, TauLeap]
    compartment S = 990, I = 10, R = 0
    rate beta_hi = 0.0003 [1/day]
    rate beta_lo = 0.0001 [1/day]
    rate gamma   = 0.1    [1/day]
    flow S -> I : ifelse(t - 30, beta_lo, beta_hi) * S * I
    flow I -> R : gamma * I
}
"""


# ---------------------------------------------------------------------------
# 1. Parsing and registration
# ---------------------------------------------------------------------------


def test_ifelse_registered_in_functions() -> None:
    """ifelse must appear in FUNCTIONS with arity 3."""
    assert "ifelse" in FUNCTIONS
    arity, impl = FUNCTIONS["ifelse"]
    assert arity == 3


def test_ifelse_parses() -> None:
    """A flow using ifelse(cond, a, b) must parse without error."""
    (m,) = parse(_SIR_IFELSE)
    assert m.name == "SIR"


def test_ifelse_no_check_error() -> None:
    """The static checker must report no errors for valid ifelse() usage."""
    (m,) = parse(_SIR_IFELSE)
    diags = check(m)
    errors = [d for d in diags if d.severity.value == "error"]
    assert errors == [], errors


# ---------------------------------------------------------------------------
# 2. Semantics of ifelse — mathematical properties
# ---------------------------------------------------------------------------


def test_ifelse_positive_cond_returns_a() -> None:
    """ifelse(cond, a, b) returns a when cond > 0."""
    _, impl = FUNCTIONS["ifelse"]
    assert impl(1.0, 42.0, 99.0) == 42.0
    assert impl(0.001, 0.0, 1.0) == 0.0
    assert impl(100.0, -5.0, 5.0) == -5.0


def test_ifelse_zero_cond_returns_b() -> None:
    """ifelse(0, a, b) returns b (cond not strictly > 0)."""
    _, impl = FUNCTIONS["ifelse"]
    assert impl(0.0, 42.0, 99.0) == 99.0


def test_ifelse_negative_cond_returns_b() -> None:
    """ifelse(cond, a, b) returns b when cond < 0."""
    _, impl = FUNCTIONS["ifelse"]
    assert impl(-1.0, 42.0, 99.0) == 99.0
    assert impl(-0.001, 0.0, 1.0) == 1.0
    assert impl(-100.0, -5.0, 5.0) == 5.0


def test_ifelse_a_and_b_independent() -> None:
    """The returned value must come from a XOR b, never a mix."""
    _, impl = FUNCTIONS["ifelse"]
    assert impl(1.0, 1.0, 2.0) == 1.0
    assert impl(-1.0, 1.0, 2.0) == 2.0


# ---------------------------------------------------------------------------
# 3. compile_expr integration
# ---------------------------------------------------------------------------


def test_ifelse_in_compiled_expression() -> None:
    """compile_expr must evaluate ifelse(t - 30, 0.1, 0.3) correctly."""
    from epidsl.ir import BinOp, Call, Num, Sym

    sub_expr = BinOp("-", Sym("t"), Num(30.0))
    ifelse_expr = Call("ifelse", (sub_expr, Num(0.1), Num(0.3)), line=0, column=0)
    f = compile_expr(ifelse_expr, {}, {})
    state = np.zeros(1)
    # t = 10 < 30 → cond = -20 < 0 → b = 0.3
    assert abs(f(state, 10.0) - 0.3) < 1e-12
    # t = 50 > 30 → cond = 20 > 0 → a = 0.1
    assert abs(f(state, 50.0) - 0.1) < 1e-12
    # t = 30 → cond = 0 → b = 0.3
    assert abs(f(state, 30.0) - 0.3) < 1e-12


# ---------------------------------------------------------------------------
# 4. ODE dynamics
# ---------------------------------------------------------------------------


def test_ifelse_ode_runs() -> None:
    """ODE backend must simulate a model with ifelse() without raising."""
    (m,) = parse(_SIR_IFELSE)
    cfg = SimulationConfig(t_end=60.0, n_points=61)
    result = ODEBackend().run(m, cfg)
    assert len(result.t) == 61
    assert "I" in result.y


def test_ifelse_ode_population_conserved() -> None:
    """Total population must remain constant with ifelse-modulated transmission."""
    (m,) = parse(_SIR_IFELSE)
    cfg = SimulationConfig(t_end=60.0, n_points=61)
    result = ODEBackend().run(m, cfg)
    total = result.y["S"] + result.y["I"] + result.y["R"]
    assert np.allclose(total, 1000.0, atol=1e-6)


def test_ifelse_ode_no_extra_breakpoints() -> None:
    """ifelse() does not inject ODE breakpoints; output length must equal n_points."""
    (m,) = parse(_SIR_IFELSE)
    cfg = SimulationConfig(t_end=60.0, n_points=7)
    result = ODEBackend().run(m, cfg)
    assert len(result.t) == 7, f"expected 7 points, got {len(result.t)}"


# ---------------------------------------------------------------------------
# 5. Stochastic backend
# ---------------------------------------------------------------------------


def test_ifelse_gillespie_runs() -> None:
    """Gillespie backend must handle ifelse() without error."""
    (m,) = parse(_SIR_IFELSE)
    cfg = SimulationConfig(t_end=60.0, n_points=61, seed=42)
    result = GillespieBackend().run(m, cfg)
    total = result.y["S"] + result.y["I"] + result.y["R"]
    assert np.all(total >= 0)


# ---------------------------------------------------------------------------
# 6. Static checker — wrong arity
# ---------------------------------------------------------------------------


def test_ifelse_wrong_arity_raises_c3() -> None:
    """ifelse(cond, a) with two arguments must raise C3 (wrong arity)."""
    src = """
    model M {
        semantics: [ODE]
        compartment S = 100, I = 1
        rate beta = 0.001 [1/day]
        flow S -> I : beta * ifelse(I, S) * S
    }
    """
    (m,) = parse(src)
    diags = check(m)
    assert any(d.code == "C3" for d in diags), [d.code for d in diags]
