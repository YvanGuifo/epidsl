"""Tests pour D40 — règles de transition agent-agent (interact flows).

`interact A + B -> A' + B' : rate_expr` déclenche deux transitions simultanées :
  - l'agent focal A → A'
  - un voisin choisi aléatoirement parmi ses voisins en état B → B'

Le taux total pour un agent focal u en état A = rate_expr × count(B-voisins de u).
"""

import numpy as np
import pytest

from epidsl.ir import EpiModel, InteractFlow, SimulationConfig
from epidsl.parser import parse
from epidsl.checks import errors
from epidsl.backends.network import NetworkBackend

# ---------------------------------------------------------------------------
# Modèles de référence
# ---------------------------------------------------------------------------

_OPINION = """
model Opinion {
    semantics: [Network]
    compartment A = 50, B = 50
    rate mu = 1.0
    network watts_strogatz(k=4, rewire=0.1)
    interact A + B -> B + B : mu
    interact B + A -> A + A : mu
}
"""

_SIR_INTERACT = """
model SIR {
    semantics: [Network]
    compartment S = 90, I = 10, R = 0
    rate beta = 0.5
    rate gamma = 0.1
    network watts_strogatz(k=6, rewire=0.1)
    interact S + I -> I + I : beta
    flow I -> R : gamma
}
"""

_BAD_FOCAL_SRC = """
model Bad {
    semantics: [Network]
    compartment A = 50, B = 50
    rate mu = 0.5
    network watts_strogatz(k=4, rewire=0.1)
    interact X + B -> A + B : mu
}
"""

_BAD_NBR_SRC = """
model Bad {
    semantics: [Network]
    compartment A = 50, B = 50
    rate mu = 0.5
    network watts_strogatz(k=4, rewire=0.1)
    interact A + Y -> A + B : mu
}
"""

_BAD_FOCAL_TGT = """
model Bad {
    semantics: [Network]
    compartment A = 50, B = 50
    rate mu = 0.5
    network watts_strogatz(k=4, rewire=0.1)
    interact A + B -> Z + B : mu
}
"""

_BAD_NBR_TGT = """
model Bad {
    semantics: [Network]
    compartment A = 50, B = 50
    rate mu = 0.5
    network watts_strogatz(k=4, rewire=0.1)
    interact A + B -> A + W : mu
}
"""

_NO_NETWORK = """
model Bad {
    semantics: [ODE]
    compartment A = 50, B = 50
    rate mu = 0.5
    interact A + B -> B + B : mu
}
"""


# ---------------------------------------------------------------------------
# Phase rouge : parsing et IR
# ---------------------------------------------------------------------------


def test_interact_parses():
    """Un modèle avec interact se parse sans exception."""
    models = parse(_OPINION)
    assert len(models) == 1


def test_interact_ir_stored():
    """Les quatre états et l'expression sont stockés dans l'IR."""
    (model,) = parse(_OPINION)
    assert len(model.interact_flows) == 2
    first = model.interact_flows[0]
    assert isinstance(first, InteractFlow)
    assert first.focal_source == "A"
    assert first.neighbor_source == "B"
    assert first.focal_target == "B"
    assert first.neighbor_target == "B"


def test_interact_no_errors():
    """Un modèle interact valide n'a pas d'erreurs statiques."""
    (model,) = parse(_OPINION)
    assert errors(model) == []


def test_interact_sir_no_errors():
    """SIR avec interact + flow conventionnel n'a pas d'erreurs."""
    (model,) = parse(_SIR_INTERACT)
    assert errors(model) == []


# ---------------------------------------------------------------------------
# Vérifications statiques
# ---------------------------------------------------------------------------


def test_interact_bad_focal_source_raises():
    """C-I1 : état focal source inconnu → erreur."""
    (model,) = parse(_BAD_FOCAL_SRC)
    diags = errors(model)
    codes = [d.code for d in diags]
    assert "C-I1" in codes


def test_interact_bad_neighbor_source_raises():
    """C-I1 : état voisin source inconnu → erreur."""
    (model,) = parse(_BAD_NBR_SRC)
    diags = errors(model)
    codes = [d.code for d in diags]
    assert "C-I1" in codes


def test_interact_bad_focal_target_raises():
    """C-I1 : état focal cible inconnu → erreur."""
    (model,) = parse(_BAD_FOCAL_TGT)
    diags = errors(model)
    codes = [d.code for d in diags]
    assert "C-I1" in codes


def test_interact_bad_neighbor_target_raises():
    """C-I1 : état voisin cible inconnu → erreur."""
    (model,) = parse(_BAD_NBR_TGT)
    diags = errors(model)
    codes = [d.code for d in diags]
    assert "C-I1" in codes


def test_interact_no_network_semantics_raises():
    """C-I2 : interact sans sémantique Network → erreur."""
    (model,) = parse(_NO_NETWORK)
    diags = errors(model)
    codes = [d.code for d in diags]
    assert "C-I2" in codes


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def _opinion_cfg(t_end: float = 10.0, seed: int = 42) -> SimulationConfig:
    return SimulationConfig(t_end=t_end, n_points=51, seed=seed)


def test_interact_backend_runs():
    """La simulation avec interact se termine sans exception."""
    (model,) = parse(_OPINION)
    cfg = _opinion_cfg(t_end=5.0)
    result = NetworkBackend().run(model, cfg)
    assert "A" in result.y
    assert "B" in result.y
    assert len(result.y["A"]) == cfg.n_points


def test_interact_population_conserved():
    """La population totale est conservée à chaque pas."""
    (model,) = parse(_OPINION)
    cfg = _opinion_cfg(t_end=10.0)
    result = NetworkBackend().run(model, cfg)
    total = result.y["A"] + result.y["B"]
    assert np.allclose(total, 100.0), f"population non conservée : {total}"


def test_interact_seed_reproducible():
    """Deux runs avec la même graine donnent la même trajectoire."""
    (model,) = parse(_OPINION)
    cfg1 = _opinion_cfg(seed=99)
    cfg2 = _opinion_cfg(seed=99)
    r1 = NetworkBackend().run(model, cfg1)
    r2 = NetworkBackend().run(model, cfg2)
    assert np.array_equal(r1.y["A"], r2.y["A"])


def test_interact_consensus_reached():
    """La dynamique d'opinion converge vers un consensus à long terme."""
    (model,) = parse(_OPINION)
    cfg = SimulationConfig(t_end=200.0, n_points=101, seed=7)
    result = NetworkBackend().run(model, cfg)
    final_a = result.y["A"][-1]
    final_b = result.y["B"][-1]
    assert np.isclose(final_a + final_b, 100.0)
    assert min(final_a, final_b) < 10.0, (
        f"pas de consensus détecté : A={final_a:.0f}, B={final_b:.0f}"
    )


def test_interact_sir_population_conserved():
    """SIR avec interact conserve S + I + R."""
    (model,) = parse(_SIR_INTERACT)
    cfg = SimulationConfig(t_end=20.0, n_points=51, seed=42)
    result = NetworkBackend().run(model, cfg)
    total = result.y["S"] + result.y["I"] + result.y["R"]
    assert np.allclose(total, 100.0), f"population SIR non conservée : {total}"


def test_interact_sir_epidemic_spreads():
    """SIR avec interact : I augmente effectivement (épidémie détectée)."""
    (model,) = parse(_SIR_INTERACT)
    cfg = SimulationConfig(t_end=30.0, n_points=101, seed=42)
    result = NetworkBackend().run(model, cfg)
    max_i = result.y["I"].max()
    assert max_i > 10.0, f"aucune croissance de I détectée : max={max_i:.1f}"
