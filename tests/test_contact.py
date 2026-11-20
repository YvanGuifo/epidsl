"""Tests for D30 — contact matrices (by Age, Age) for stratified transmission."""

from __future__ import annotations

import numpy as np
import pytest

from epidsl.backends.ode import ODEBackend
from epidsl.compose import CompositionError
from epidsl.ir import SimulationConfig
from epidsl.parser import parse

# ---------------------------------------------------------------------------
# Case A: beta declared as scalar in the SIR model, overridden by matrix in compose.
# This is the primary workflow.
# ---------------------------------------------------------------------------

_SIR_SCALAR = """
model SIR {
    semantics: [ODE]
    compartment S = 900, I = 100, R = 0
    rate beta  = 0.0003 [1/day]
    rate gamma = 0.1    [1/day]
    flow S -> I : beta * S * I
    flow I -> R : gamma * I
    observe cases = incidence(S -> I)
}
"""

_AGE2 = """
model Age {
    semantics: [ODE]
    compartment young = 600, old = 400
}
"""

_AGE_SIR_A = (
    _SIR_SCALAR
    + _AGE2
    + """
compose AgeSIR {
    models: [SIR, Age]
    parameter beta by Age, Age = {
        young, young: 0.0005, young, old: 0.0001,
        old,  young: 0.0001, old,  old:  0.0003
    } [1/day]
}
"""
)

# ---------------------------------------------------------------------------
# Case B: beta NOT in SIR, declared inline as a matrix — enabled by D30.
# ---------------------------------------------------------------------------

_SIR_NO_BETA = """
model SIR {
    semantics: [ODE]
    compartment S = 900, I = 100, R = 0
    rate gamma = 0.1 [1/day]
    flow S -> I : beta * S * I
    flow I -> R : gamma * I
}
"""

_AGE_SIR_B = (
    _SIR_NO_BETA
    + _AGE2
    + """
compose AgeSIR {
    models: [SIR, Age]
    parameter beta by Age, Age = {
        young, young: 0.0005, young, old: 0.0001,
        old,  young: 0.0001, old,  old:  0.0003
    } [1/day]
}
"""
)

# Block: off-diagonal = 0 (no cross-group transmission).
_AGE_SIR_BLOCK = (
    _SIR_SCALAR
    + _AGE2
    + """
compose AgeSIR {
    models: [SIR, Age]
    parameter beta by Age, Age = {
        young, young: 0.0005, young, old: 0.0,
        old,  young: 0.0,    old,  old:  0.0005
    } [1/day]
}
"""
)

# Non-migration matrix with a missing diagonal — must still require all entries.
_AGE_SIR_MISSING = (
    _SIR_SCALAR
    + _AGE2
    + """
compose AgeSIR {
    models: [SIR, Age]
    parameter beta by Age, Age = {
        young, old: 0.0001,
        old, young: 0.0001,
        old, old:   0.0003
    } [1/day]
}
"""
)

# 3-age-group model.
_AGE3 = """
model Age {
    semantics: [ODE]
    compartment child = 300, adult = 500, elder = 200
}
"""

_AGE_SIR_3 = (
    _SIR_SCALAR
    + _AGE3
    + """
compose AgeSIR {
    models: [SIR, Age]
    parameter beta by Age, Age = {
        child, child: 0.0010, child, adult: 0.0003, child, elder: 0.0001,
        adult, child: 0.0003, adult, adult: 0.0005, adult, elder: 0.0002,
        elder, child: 0.0001, elder, adult: 0.0002, elder, elder: 0.0004
    } [1/day]
}
"""
)


# ---------------------------------------------------------------------------
# Parsing and structure
# ---------------------------------------------------------------------------


def test_contact_matrix_case_a_parses() -> None:
    """Contact matrix overriding a scalar beta in SIR must parse."""
    models = parse(_AGE_SIR_A)
    assert any(m.name == "AgeSIR" for m in models)


def test_contact_matrix_case_b_parses() -> None:
    """Contact matrix declared inline (beta not in SIR) must parse (D30)."""
    models = parse(_AGE_SIR_B)
    assert any(m.name == "AgeSIR" for m in models)


def test_contact_diagonal_constants_created_case_a() -> None:
    """Diagonal constants (beta_young_young, beta_old_old) must exist — case A."""
    (m,) = [x for x in parse(_AGE_SIR_A) if x.name == "AgeSIR"]
    names = {c.name for c in m.constants}
    assert "beta_young_young" in names
    assert "beta_old_old" in names
    assert "beta_young_old" in names
    assert "beta_old_young" in names


def test_contact_diagonal_constants_created_case_b() -> None:
    """Diagonal constants must also exist when beta is declared inline (D30)."""
    (m,) = [x for x in parse(_AGE_SIR_B) if x.name == "AgeSIR"]
    names = {c.name for c in m.constants}
    assert "beta_young_young" in names, f"missing beta_young_young; got {names}"
    assert "beta_old_old" in names, f"missing beta_old_old; got {names}"


def test_contact_all_four_values_correct() -> None:
    """The generated constants must carry the values declared in the matrix."""
    (m,) = [x for x in parse(_AGE_SIR_A) if x.name == "AgeSIR"]
    vals = {c.name: c.value for c in m.constants}
    assert vals["beta_young_young"] == pytest.approx(0.0005)
    assert vals["beta_young_old"] == pytest.approx(0.0001)
    assert vals["beta_old_young"] == pytest.approx(0.0001)
    assert vals["beta_old_old"] == pytest.approx(0.0003)


def test_contact_matrix_flow_uses_both_groups() -> None:
    """The rate of S_young -> I_young must reference both I_young and I_old."""
    (m,) = [x for x in parse(_AGE_SIR_A) if x.name == "AgeSIR"]
    for flow in m.flows:
        if flow.source == "S_young" and flow.target == "I_young":
            s = _syms(flow.expr)
            assert "I_young" in s, f"I_young missing from {s}"
            assert "I_old" in s, f"I_old missing from {s}"
            return
    pytest.fail("S_young -> I_young flow not found")


def _syms(expr) -> set[str]:
    from epidsl.ir import BinOp, Neg, Sym
    result: set[str] = set()
    stack = [expr]
    while stack:
        e = stack.pop()
        match e:
            case Sym(name=n):
                result.add(n)
            case BinOp(_, left, right):
                stack.append(left); stack.append(right)
            case Neg(operand):
                stack.append(operand)
    return result


def test_contact_matrix_missing_diagonal_raises() -> None:
    """A contact matrix with a missing diagonal entry must be rejected."""
    with pytest.raises((CompositionError, Exception)):
        parse(_AGE_SIR_MISSING)


# ---------------------------------------------------------------------------
# ODE dynamics
# ---------------------------------------------------------------------------


def test_contact_matrix_ode_runs() -> None:
    """ODE backend must simulate a contact-matrix model without raising."""
    (m,) = [x for x in parse(_AGE_SIR_A) if x.name == "AgeSIR"]
    cfg = SimulationConfig(t_end=30.0, n_points=31)
    result = ODEBackend().run(m, cfg)
    assert "S_young" in result.y
    assert "I_old" in result.y
    assert len(result.t) == 31


def test_contact_matrix_population_conserved() -> None:
    """N = sum of all compartments must stay constant."""
    (m,) = [x for x in parse(_AGE_SIR_A) if x.name == "AgeSIR"]
    cfg = SimulationConfig(t_end=60.0, n_points=61)
    result = ODEBackend().run(m, cfg)
    total = sum(result.y[c.name] for c in m.compartments)
    assert np.allclose(total, 1000.0, atol=1e-6)


def test_contact_inline_ode_runs() -> None:
    """Inline contact matrix (case B) must also simulate correctly."""
    (m,) = [x for x in parse(_AGE_SIR_B) if x.name == "AgeSIR"]
    cfg = SimulationConfig(t_end=30.0, n_points=31)
    result = ODEBackend().run(m, cfg)
    total = sum(result.y[c.name] for c in m.compartments)
    assert np.allclose(total, 1000.0, atol=1e-6)


def test_contact_block_no_cross_infection() -> None:
    """With off-diagonal beta = 0, the ODE must still run with non-negative values."""
    (m,) = [x for x in parse(_AGE_SIR_BLOCK) if x.name == "AgeSIR"]
    cfg = SimulationConfig(t_end=30.0, n_points=31)
    result = ODEBackend().run(m, cfg)
    assert np.all(result.y["S_young"] >= 0)
    assert np.all(result.y["S_old"] >= 0)


def test_contact_epidemic_actually_occurs() -> None:
    """The contact sum (Σ_j beta_{i,j} * I_j) must drive transmission: S must
    decrease significantly over 60 days (not stay flat due to a corrupted sum).

    Critical: a block-diagonal matrix (off-diagonal = 0) must still produce
    intra-group transmission.  If the terms are multiplied instead of summed,
    the product includes one zero factor and collapses to 0 → no epidemic.
    """
    (m,) = [x for x in parse(_AGE_SIR_BLOCK) if x.name == "AgeSIR"]
    cfg = SimulationConfig(t_end=60.0, n_points=61)
    result = ODEBackend().run(m, cfg)
    s_young_drop = result.y["S_young"][0] - result.y["S_young"][-1]
    s_old_drop = result.y["S_old"][0] - result.y["S_old"][-1]
    # Even in the block model, intra-group beta=0.0005 drives a large epidemic.
    assert s_young_drop > 100.0, f"S_young barely changed: drop={s_young_drop:.1f}"
    assert s_old_drop > 50.0, f"S_old barely changed: drop={s_old_drop:.1f}"


def test_contact_matrix_3_groups_runs() -> None:
    """3-group contact matrix must compile and simulate without error."""
    (m,) = [x for x in parse(_AGE_SIR_3) if x.name == "AgeSIR"]
    cfg = SimulationConfig(t_end=30.0, n_points=31)
    result = ODEBackend().run(m, cfg)
    assert "S_child" in result.y
    assert "S_elder" in result.y
    total = sum(result.y[c.name] for c in m.compartments)
    assert np.allclose(total, 1000.0, atol=1e-5)
