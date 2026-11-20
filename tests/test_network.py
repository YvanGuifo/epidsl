"""Tests for D38 — contact network backend (NetworkBackend)."""

from __future__ import annotations

import numpy as np

from epidsl.checks import check
from epidsl.ir import SimulationConfig
from epidsl.parser import parse

# ---------------------------------------------------------------------------
# Reference models
# ---------------------------------------------------------------------------

_SIR_WATTS = """
model SIR {
    semantics: [Network]
    compartment S = 990, I = 10, R = 0
    rate beta  = 0.3   [1/contact/day]
    rate gamma = 0.1   [1/day]
    network watts_strogatz(k=6, rewire=0.1)
    flow S -> I : beta * contacts(I)
    flow I -> R : gamma
}
"""

_SIR_ERDOS = """
model SIR {
    semantics: [Network]
    compartment S = 490, I = 10, R = 0
    rate beta  = 0.5   [1/contact/day]
    rate gamma = 0.1   [1/day]
    network erdos_renyi(p=0.01)
    flow S -> I : beta * contacts(I)
    flow I -> R : gamma
}
"""

_SIR_BARABASI = """
model SIR {
    semantics: [Network]
    compartment S = 490, I = 10, R = 0
    rate beta  = 0.5   [1/contact/day]
    rate gamma = 0.1   [1/day]
    network barabasi_albert(m=3)
    flow S -> I : beta * contacts(I)
    flow I -> R : gamma
}
"""

_SIR_COMPLETE = """
model SIR {
    semantics: [Network]
    compartment S = 90, I = 10, R = 0
    rate beta  = 0.003 [1/contact/day]
    rate gamma = 0.1   [1/day]
    network complete()
    flow S -> I : beta * contacts(I)
    flow I -> R : gamma
}
"""

_NO_NETWORK = """
model SIR {
    semantics: [Network]
    compartment S = 90, I = 10, R = 0
    rate beta  = 0.003 [1/contact/day]
    rate gamma = 0.1   [1/day]
    flow S -> I : beta * contacts(I)
    flow I -> R : gamma
}
"""

_CONTACTS_WRONG_ARITY = """
model M {
    semantics: [Network]
    compartment S = 90, I = 10, R = 0
    rate beta = 0.003 [1/contact/day]
    network complete()
    flow S -> I : beta * contacts(I, S)
}
"""


# ---------------------------------------------------------------------------
# 1. Parsing and IR
# ---------------------------------------------------------------------------


def test_network_parses_watts() -> None:
    """A model with network watts_strogatz must parse without error."""
    (m,) = parse(_SIR_WATTS)
    assert m.network is not None
    assert m.network.generator == "watts_strogatz"


def test_network_parses_erdos() -> None:
    (m,) = parse(_SIR_ERDOS)
    assert m.network is not None
    assert m.network.generator == "erdos_renyi"


def test_network_parses_barabasi() -> None:
    (m,) = parse(_SIR_BARABASI)
    assert m.network is not None
    assert m.network.generator == "barabasi_albert"


def test_network_parses_complete() -> None:
    (m,) = parse(_SIR_COMPLETE)
    assert m.network is not None
    assert m.network.generator == "complete"


def test_network_params_stored() -> None:
    """Network parameters must be stored in NetworkDecl.params."""
    (m,) = parse(_SIR_WATTS)
    params = dict(m.network.params)  # type: ignore[union-attr]
    assert abs(params["k"] - 6.0) < 1e-9
    assert abs(params["rewire"] - 0.1) < 1e-9


# ---------------------------------------------------------------------------
# 2. Static checks
# ---------------------------------------------------------------------------


def test_network_no_check_error_watts() -> None:
    (m,) = parse(_SIR_WATTS)
    errors = [d for d in check(m) if d.severity.value == "error"]
    assert errors == [], errors


def test_network_missing_declaration_raises() -> None:
    """semantics: [Network] without a network clause must be an error."""
    (m,) = parse(_NO_NETWORK)
    diags = check(m)
    codes = [d.code for d in diags if d.severity.value == "error"]
    assert "C-N1" in codes


def test_contacts_wrong_arity_raises() -> None:
    """contacts(I, S) with two arguments must raise C-N2."""
    (m,) = parse(_CONTACTS_WRONG_ARITY)
    diags = check(m)
    codes = [d.code for d in diags if d.severity.value == "error"]
    assert "C-N2" in codes


# ---------------------------------------------------------------------------
# 3. NetworkBackend — basic runs
# ---------------------------------------------------------------------------


def test_network_backend_runs_watts() -> None:
    """NetworkBackend must simulate a watts_strogatz model without error."""
    from epidsl.backends.network import NetworkBackend
    (m,) = parse(_SIR_WATTS)
    cfg = SimulationConfig(t_end=20.0, n_points=21, seed=42)
    result = NetworkBackend().run(m, cfg)
    assert len(result.t) == 21
    assert "S" in result.y and "I" in result.y and "R" in result.y


def test_network_population_conserved() -> None:
    """Total N must be conserved throughout the simulation."""
    from epidsl.backends.network import NetworkBackend
    (m,) = parse(_SIR_WATTS)
    cfg = SimulationConfig(t_end=20.0, n_points=21, seed=42)
    result = NetworkBackend().run(m, cfg)
    total = result.y["S"] + result.y["I"] + result.y["R"]
    N = sum(c.initial for c in m.compartments)
    assert np.allclose(total, N, atol=0), f"population not conserved: {total}"


def test_network_values_non_negative() -> None:
    """Compartment counts must be non-negative integers throughout."""
    from epidsl.backends.network import NetworkBackend
    (m,) = parse(_SIR_WATTS)
    cfg = SimulationConfig(t_end=30.0, n_points=31, seed=7)
    result = NetworkBackend().run(m, cfg)
    for name, series in result.y.items():
        assert np.all(series >= 0), f"{name} went negative"


def test_network_seed_reproducible() -> None:
    """Same seed must give identical trajectories."""
    from epidsl.backends.network import NetworkBackend
    (m,) = parse(_SIR_ERDOS)
    cfg = SimulationConfig(t_end=20.0, n_points=11, seed=99)
    r1 = NetworkBackend().run(m, cfg)
    r2 = NetworkBackend().run(m, cfg)
    assert np.array_equal(r1.y["I"], r2.y["I"])


def test_network_different_seeds_differ() -> None:
    """Different seeds must (almost certainly) give different trajectories."""
    from epidsl.backends.network import NetworkBackend
    (m,) = parse(_SIR_WATTS)
    cfg1 = SimulationConfig(t_end=20.0, n_points=11, seed=1)
    cfg2 = SimulationConfig(t_end=20.0, n_points=11, seed=2)
    r1 = NetworkBackend().run(m, cfg1)
    r2 = NetworkBackend().run(m, cfg2)
    assert not np.array_equal(r1.y["I"], r2.y["I"])


def test_network_complete_matches_mass_action_tendency() -> None:
    """On a complete graph, the network model must show epidemic growth."""
    from epidsl.backends.network import NetworkBackend
    (m,) = parse(_SIR_COMPLETE)
    cfg = SimulationConfig(t_end=10.0, n_points=11, seed=42)
    result = NetworkBackend().run(m, cfg)
    # epidemic should grow: peak I > initial I
    assert max(result.y["I"]) > 10


def test_network_erdos_runs() -> None:
    from epidsl.backends.network import NetworkBackend
    (m,) = parse(_SIR_ERDOS)
    cfg = SimulationConfig(t_end=20.0, n_points=11, seed=42)
    result = NetworkBackend().run(m, cfg)
    assert len(result.t) == 11


def test_network_barabasi_runs() -> None:
    from epidsl.backends.network import NetworkBackend
    (m,) = parse(_SIR_BARABASI)
    cfg = SimulationConfig(t_end=20.0, n_points=11, seed=42)
    result = NetworkBackend().run(m, cfg)
    assert len(result.t) == 11
