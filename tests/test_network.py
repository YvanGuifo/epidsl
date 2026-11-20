"""Tests for metapopulation / migration (migrate statement)."""

from __future__ import annotations

import numpy as np
import pytest

from epidsl.backends.gillespie import GillespieBackend
from epidsl.backends.ode import ODEBackend
from epidsl.compose import CompositionError
from epidsl.ir import SimulationConfig
from epidsl.parser import parse

# ---------------------------------------------------------------------------
# Shared DSL snippets
# ---------------------------------------------------------------------------

_SIR = """
model SIR {
    semantics: [ODE, Gillespie, TauLeap]
    compartment S = 900, I = 100, R = 0
    rate beta = 0.0003 [1/day]
    rate gamma = 0.1  [1/day]
    flow S -> I : beta * S * I
    flow I -> R : gamma * I
}
"""

_PATCH2 = """
model Patch {
    semantics: [ODE, Gillespie, TauLeap]
    compartment A = 500, B = 500
}
"""

_PATCH3 = """
model Patch {
    semantics: [ODE, Gillespie, TauLeap]
    compartment A = 500, B = 250, C = 250
}
"""

# Metapopulation: SIR on 2 patches with symmetric migration
_META2 = (
    _SIR
    + _PATCH2
    + """
compose MetaSIR {
    models: [SIR, Patch]
    rate travel by Patch, Patch = {
        A, A: 0,   A, B: 0.1,
        B, A: 0.1, B, B: 0
    } [1/day]
    migrate S, I, R : travel
}
"""
)

# Same but with 3 patches
_META3 = (
    _SIR
    + _PATCH3
    + """
compose MetaSIR {
    models: [SIR, Patch]
    rate travel by Patch, Patch = {
        A, A: 0,    A, B: 0.05, A, C: 0.02,
        B, A: 0.05, B, B: 0,    B, C: 0.03,
        C, A: 0.02, C, B: 0.03, C, C: 0
    } [1/day]
    migrate S, I, R : travel
}
"""
)


# ---------------------------------------------------------------------------
# Parsing and structure tests
# ---------------------------------------------------------------------------


def test_migrate_flows_generated() -> None:
    """flatten() must add migration flows for every (src, dst, compartment) triple."""
    (m,) = [m for m in parse(_META2) if m.name == "MetaSIR"]
    # We expect migration flows: S_A <-> S_B, I_A <-> I_B, R_A <-> R_B  (6 flows)
    # Plus local SIR flows: 2 flows per patch = 4 local flows
    migration_flows = [
        f for f in m.flows if f.source is not None and f.target is not None
        and f.source != f.target
        and {f.source, f.target} in (
            {"S_A", "S_B"}, {"I_A", "I_B"}, {"R_A", "R_B"}
        )
    ]
    assert len(migration_flows) == 6, (
        f"expected 6 migration flows (2 directions × 3 compartments), got {len(migration_flows)}"
    )


def test_migrate_generates_correct_rate_names() -> None:
    """Migration flows reference the generated rate constants, e.g. travel_A_B."""
    from epidsl.ir import BinOp, Sym

    (m,) = [m for m in parse(_META2) if m.name == "MetaSIR"]
    constant_names = {c.name for c in m.constants}
    assert "travel_A_B" in constant_names
    assert "travel_B_A" in constant_names
    # Find S_A -> S_B flow and check its rate references travel_A_B
    for f in m.flows:
        if f.source == "S_A" and f.target == "S_B":
            assert isinstance(f.expr, BinOp), "migration rate should be rate * compartment"
            syms = {node.name for node in _walk_syms(f.expr)}
            assert "travel_A_B" in syms
            break
    else:
        pytest.fail("S_A -> S_B flow not found")


def _walk_syms(expr):
    from epidsl.ir import BinOp, Neg, Sym
    match expr:
        case Sym() as s:
            yield s
        case BinOp(_, left, right):
            yield from _walk_syms(left)
            yield from _walk_syms(right)
        case Neg(operand):
            yield from _walk_syms(operand)


def test_migrate_three_patches_flow_count() -> None:
    """With 3 patches and 3 compartments: 3 × 3 × (3-1) = 18 migration flows."""
    (m,) = [m for m in parse(_META3) if m.name == "MetaSIR"]
    local_states = {"A", "B", "C"}
    migration_flows = [
        f for f in m.flows
        if f.source is not None and f.target is not None
        and f.source.split("_")[1] in local_states  # second part is a patch name
        and f.target.split("_")[1] in local_states
        and f.source.split("_")[1] != f.target.split("_")[1]  # different patches
    ]
    assert len(migration_flows) == 18, (
        f"expected 18 migration flows, got {len(migration_flows)}"
    )


# ---------------------------------------------------------------------------
# ODE conservation and dynamics
# ---------------------------------------------------------------------------


def test_migrate_ode_total_population_conserved() -> None:
    """Total N = S + I + R must stay constant under migration + SIR dynamics."""
    (m,) = [m for m in parse(_META2) if m.name == "MetaSIR"]
    cfg = SimulationConfig(t_end=30.0, n_points=301)
    result = ODEBackend().run(m, cfg)
    total = sum(result.y[c.name] for c in m.compartments)
    assert np.allclose(total, 1000.0, atol=1e-6), "Total population drifted"


def test_migrate_ode_infection_spreads_between_patches() -> None:
    """With I only in A initially, infections should appear in B."""
    # Custom initial conditions: all I in patch A, patch B starts clean
    src = (
        """
model SIR {
    semantics: [ODE]
    compartment S = 900, I = 100, R = 0
    rate beta = 0.3 [1/day]
    rate gamma = 0.1 [1/day]
    flow S -> I : beta * S * I
    flow I -> R : gamma * I
}
model Patch {
    semantics: [ODE]
    compartment A = 500, B = 500
}
compose MetaSIR {
    models: [SIR, Patch]
    rate travel by Patch, Patch = {
        A, A: 0,   A, B: 0.5,
        B, A: 0.5, B, B: 0
    } [1/day]
    migrate I : travel
}
"""
    )
    (m,) = [x for x in parse(src) if x.name == "MetaSIR"]
    cfg = SimulationConfig(t_end=5.0, n_points=51)
    result = ODEBackend().run(m, cfg)
    # I_B starts at 50 (product initial), but with high migration it should grow
    # At t>0, I_B should be nonzero
    i_b = result.y["I_B"]
    assert i_b[-1] > i_b[0], "I_B should grow when I is present in patch A"


def test_migrate_ode_symmetric_no_net_drift() -> None:
    """With equal patch sizes and symmetric rates, no net migration at t=0."""
    (m,) = [m for m in parse(_META2) if m.name == "MetaSIR"]
    cfg = SimulationConfig(t_end=0.001, n_points=2)
    result = ODEBackend().run(m, cfg)
    # S in each patch should stay roughly equal at very short time (no infection yet)
    s_a = result.y["S_A"]
    s_b = result.y["S_B"]
    assert abs(s_a[-1] - s_b[-1]) < 1.0, "No net migration expected for symmetric setup"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_migrate_bad_rate_name_raises() -> None:
    """migrate with an unknown rate name should raise CompositionError."""
    src = (
        _SIR
        + _PATCH2
        + """
compose MetaSIR {
    models: [SIR, Patch]
    rate travel by Patch, Patch = {
        A, A: 0, A, B: 0.1,
        B, A: 0.1, B, B: 0
    } [1/day]
    migrate S : nonexistent_rate
}
"""
    )
    with pytest.raises((CompositionError, Exception)):
        parse(src)


def test_migrate_rate_not_matrix_raises() -> None:
    """migrate with a non-matrix rate should raise CompositionError."""
    src = (
        _SIR
        + _PATCH2
        + """
compose MetaSIR {
    models: [SIR, Patch]
    rate mu = 0.01 [1/day]
    migrate S : mu
}
"""
    )
    with pytest.raises((CompositionError, Exception)):
        parse(src)


def test_migrate_unknown_compartment_raises() -> None:
    """migrate listing a compartment not in any model should raise."""
    src = (
        _SIR
        + _PATCH2
        + """
compose MetaSIR {
    models: [SIR, Patch]
    rate travel by Patch, Patch = {
        A, A: 0, A, B: 0.1,
        B, A: 0.1, B, B: 0
    } [1/day]
    migrate S, X : travel
}
"""
    )
    with pytest.raises((CompositionError, Exception)):
        parse(src)


def test_migrate_patch_dimension_compartment_raises() -> None:
    """migrate with a compartment from the indexed model itself should raise."""
    src = (
        _SIR
        + _PATCH2
        + """
compose MetaSIR {
    models: [SIR, Patch]
    rate travel by Patch, Patch = {
        A, A: 0, A, B: 0.1,
        B, A: 0.1, B, B: 0
    } [1/day]
    migrate A : travel
}
"""
    )
    with pytest.raises((CompositionError, Exception)):
        parse(src)


# ---------------------------------------------------------------------------
# Stochastic: conservation
# ---------------------------------------------------------------------------


def test_migrate_gillespie_total_integer_conservation() -> None:
    """Gillespie run with migration must conserve total population (integer counts)."""
    (m,) = [m for m in parse(_META2) if m.name == "MetaSIR"]
    cfg = SimulationConfig(t_end=10.0, n_points=101, seed=42)
    result = GillespieBackend().run(m, cfg)
    total = sum(result.y[c.name] for c in m.compartments)
    assert np.allclose(total, 1000.0, atol=0.5), "Gillespie total drifted"
    # Also check that compartment values are non-negative
    for name, arr in result.y.items():
        assert np.all(arr >= 0), f"Negative count in {name}"


# ---------------------------------------------------------------------------
# D28 — Sparse matrices (entries omitted default to 0)
# ---------------------------------------------------------------------------

# Same topology as _META2 but diagonal zeros omitted.
_META2_SPARSE = (
    _SIR
    + _PATCH2
    + """
compose MetaSIR {
    models: [SIR, Patch]
    rate travel by Patch, Patch = {
        A, B: 0.1,
        B, A: 0.1
    } [1/day]
    migrate S, I, R : travel
}
"""
)

# Same topology as _META3 but diagonal zeros omitted.
_META3_SPARSE = (
    _SIR
    + _PATCH3
    + """
compose MetaSIR {
    models: [SIR, Patch]
    rate travel by Patch, Patch = {
        A, B: 0.05, A, C: 0.02,
        B, A: 0.05, B, C: 0.03,
        C, A: 0.02, C, B: 0.03
    } [1/day]
    migrate S, I, R : travel
}
"""
)

# Directed chain A→B→C: nodes B and C receive but don't send back.
_META3_CHAIN = (
    _SIR
    + _PATCH3
    + """
compose MetaSIR {
    models: [SIR, Patch]
    rate travel by Patch, Patch = {
        A, B: 0.05,
        B, C: 0.03
    } [1/day]
    migrate S, I, R : travel
}
"""
)


def test_sparse_diagonal_omitted_parses() -> None:
    """Diagonal zeros can be omitted in a by M, M matrix."""
    models = parse(_META2_SPARSE)
    (m,) = [m for m in models if m.name == "MetaSIR"]
    assert any(c.name.startswith("S_") for c in m.compartments)


def test_sparse_omitted_equivalent_to_explicit_zero() -> None:
    """Omitting a pair produces the same flows as writing it with value 0."""
    (m_sparse,) = [m for m in parse(_META2_SPARSE) if m.name == "MetaSIR"]
    (m_dense,) = [m for m in parse(_META2) if m.name == "MetaSIR"]
    sparse_edges = {(f.source, f.target) for f in m_sparse.flows}
    dense_edges = {(f.source, f.target) for f in m_dense.flows}
    assert sparse_edges == dense_edges


def test_sparse_three_patches_flow_count() -> None:
    """3-patch sparse matrix → same number of flows as the dense equivalent."""
    (m_sparse,) = [m for m in parse(_META3_SPARSE) if m.name == "MetaSIR"]
    (m_dense,) = [m for m in parse(_META3) if m.name == "MetaSIR"]
    assert len(m_sparse.flows) == len(m_dense.flows)


def test_sparse_invalid_patch_name_still_fails() -> None:
    """An explicit entry with a non-existent compartment name is still rejected."""
    bad = (
        _SIR
        + _PATCH2
        + """
compose MetaSIR {
    models: [SIR, Patch]
    rate travel by Patch, Patch = { A, Z: 0.1 } [1/day]
    migrate S, I, R : travel
}
"""
    )
    with pytest.raises((CompositionError, Exception)):
        parse(bad)


def test_sparse_chain_simulation_runs() -> None:
    """Asymmetric directed chain (A→B→C, no return) simulates without error."""
    (m,) = [m for m in parse(_META3_CHAIN) if m.name == "MetaSIR"]
    cfg = SimulationConfig(t_end=10.0, n_points=11)
    result = ODEBackend().run(m, cfg)
    assert "S_A" in result.y and "S_C" in result.y
