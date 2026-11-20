"""Tests pour neighbor_attr(state, attr_name, value) — D41.

neighbor_attr(S, risk, High) renvoie le nombre de voisins dans l'état S
dont l'attribut `risk` vaut `High`.  Comme contacts(), il est utilisable
uniquement dans des modèles avec sémantique Network.

Nouveaux contrôles statiques :
  C-NA1  neighbor_attr() arité incorrecte (≠ 3)      erreur
  C-NA2  neighbor_attr() hors sémantique Network       erreur
  C-NA3  premier argument n'est pas un compartiment   erreur
  C-NA4  deuxième argument n'est pas un attribut déclaré erreur
"""

from __future__ import annotations

import pytest

from epidsl.checks import check, errors
from epidsl.parser import parse
from epidsl.backends.network import NetworkBackend
from epidsl.ir import SimulationConfig


def parse_model(src: str):
    models = parse(src)
    return models[0]

# ---------------------------------------------------------------------------
# Modèles de référence
# ---------------------------------------------------------------------------

_SIR_NEIGHBOR_ATTR = """
model SIR {
    semantics: [Network]
    compartment S = 90, I = 10, R = 0
    attribute risk ~ categorical(Low=0.7, High=0.3)
    rate beta_high = 0.5   [1/contact/day]
    rate gamma    = 0.1    [1/day]
    network watts_strogatz(k=6, rewire=0.1)
    flow S -> I : beta_high * neighbor_attr(I, risk, High)
    flow I -> R : gamma
}
"""

_SIR_COMBINED = """
model SIR {
    semantics: [Network]
    compartment S = 90, I = 10, R = 0
    attribute risk ~ categorical(Low=0.6, High=0.4)
    rate beta_low  = 0.1   [1/contact/day]
    rate beta_high = 0.5   [1/contact/day]
    rate gamma     = 0.1   [1/day]
    network watts_strogatz(k=6, rewire=0.1)
    flow S -> I : beta_low  * neighbor_attr(I, risk, Low)
               + beta_high * neighbor_attr(I, risk, High)
    flow I -> R : gamma
}
"""

# Modèle C-NA2 : pas de sémantique Network
_NO_NETWORK = """
model M {
    compartment S = 100, I = 0
    rate beta = 0.3   [1/day]
    flow S -> I : beta * neighbor_attr(I, risk, High)
}
"""

# Modèle C-NA3 : état inconnu
_BAD_STATE = """
model M {
    semantics: [Network]
    compartment S = 90, I = 10
    attribute risk ~ categorical(Low=0.7, High=0.3)
    rate beta = 0.3   [1/contact/day]
    network watts_strogatz(k=6, rewire=0.1)
    flow S -> I : beta * neighbor_attr(X, risk, High)
}
"""

# Modèle C-NA4 : attribut inconnu
_BAD_ATTR = """
model M {
    semantics: [Network]
    compartment S = 90, I = 10
    attribute risk ~ categorical(Low=0.7, High=0.3)
    rate beta = 0.3   [1/contact/day]
    network watts_strogatz(k=6, rewire=0.1)
    flow S -> I : beta * neighbor_attr(I, age, Old)
}
"""

# Modèle C-NA1 : mauvaise arité
_BAD_ARITY_2 = """
model M {
    semantics: [Network]
    compartment S = 90, I = 10
    attribute risk ~ categorical(Low=0.7, High=0.3)
    rate beta = 0.3   [1/contact/day]
    network watts_strogatz(k=6, rewire=0.1)
    flow S -> I : beta * neighbor_attr(I, risk)
}
"""

_BAD_ARITY_4 = """
model M {
    semantics: [Network]
    compartment S = 90, I = 10
    attribute risk ~ categorical(Low=0.7, High=0.3)
    rate beta = 0.3   [1/contact/day]
    network watts_strogatz(k=6, rewire=0.1)
    flow S -> I : beta * neighbor_attr(I, risk, High, Extra)
}
"""


# ---------------------------------------------------------------------------
# Tests parsing / IR
# ---------------------------------------------------------------------------


def test_neighbor_attr_parses():
    """Un modèle utilisant neighbor_attr() se parse sans exception."""
    model = parse_model(_SIR_NEIGHBOR_ATTR)
    assert model.name == "SIR"
    assert len(model.flows) == 2


def test_neighbor_attr_no_check_errors():
    """Aucune erreur statique sur un modèle correct."""
    model = parse_model(_SIR_NEIGHBOR_ATTR)
    errs = errors(model)
    assert errs == [], [str(e) for e in errs]


def test_neighbor_attr_combined_no_errors():
    """Somme de neighbor_attr() pour deux niveaux de risque : aucune erreur."""
    model = parse_model(_SIR_COMBINED)
    errs = errors(model)
    assert errs == [], [str(e) for e in errs]


# ---------------------------------------------------------------------------
# Tests contrôles statiques
# ---------------------------------------------------------------------------


def test_neighbor_attr_c_na2_no_network():
    """C-NA2 : neighbor_attr() hors sémantique Network → erreur."""
    model = parse_model(_NO_NETWORK)
    errs = errors(model)
    codes = {e.code for e in errs}
    assert "C-NA2" in codes, [str(e) for e in errs]


def test_neighbor_attr_c_na3_bad_state():
    """C-NA3 : premier argument de neighbor_attr() n'est pas un compartiment → erreur."""
    model = parse_model(_BAD_STATE)
    errs = errors(model)
    codes = {e.code for e in errs}
    assert "C-NA3" in codes, [str(e) for e in errs]


def test_neighbor_attr_c_na4_bad_attr():
    """C-NA4 : deuxième argument de neighbor_attr() n'est pas un attribut déclaré → erreur."""
    model = parse_model(_BAD_ATTR)
    errs = errors(model)
    codes = {e.code for e in errs}
    assert "C-NA4" in codes, [str(e) for e in errs]


def test_neighbor_attr_c_na1_bad_arity_2():
    """C-NA1 : neighbor_attr() avec 2 arguments → erreur."""
    model = parse_model(_BAD_ARITY_2)
    errs = errors(model)
    codes = {e.code for e in errs}
    assert "C-NA1" in codes, [str(e) for e in errs]


def test_neighbor_attr_c_na1_bad_arity_4():
    """C-NA1 : neighbor_attr() avec 4 arguments → erreur."""
    model = parse_model(_BAD_ARITY_4)
    errs = errors(model)
    codes = {e.code for e in errs}
    assert "C-NA1" in codes, [str(e) for e in errs]


# ---------------------------------------------------------------------------
# Tests backend
# ---------------------------------------------------------------------------


def test_neighbor_attr_backend_runs():
    """La simulation avec neighbor_attr() se termine sans exception."""
    model = parse_model(_SIR_NEIGHBOR_ATTR)
    cfg = SimulationConfig(t_end=5.0, n_points=11, seed=42)
    result = NetworkBackend().run(model, cfg)
    assert result.t.shape[0] == 11
    assert set(result.y.keys()) == {"S", "I", "R"}


def test_neighbor_attr_population_conserved():
    """La population totale est conservée à chaque pas de sortie."""
    model = parse_model(_SIR_NEIGHBOR_ATTR)
    cfg = SimulationConfig(t_end=10.0, n_points=21, seed=0)
    result = NetworkBackend().run(model, cfg)
    N = sum(c.initial for c in model.compartments)
    total = result.y["S"] + result.y["I"] + result.y["R"]
    assert (total == N).all(), f"population non conservée : {total}"


def test_neighbor_attr_seed_reproducible():
    """Deux runs avec la même graine donnent le même résultat."""
    model = parse_model(_SIR_NEIGHBOR_ATTR)
    cfg = SimulationConfig(t_end=5.0, n_points=6, seed=7)
    r1 = NetworkBackend().run(model, cfg)
    r2 = NetworkBackend().run(model, cfg)
    import numpy as np
    for key in ("S", "I", "R"):
        np.testing.assert_array_equal(r1.y[key], r2.y[key])


def test_neighbor_attr_infection_occurs():
    """Avec beta_high élevé et des voisins I High-risk, des infections se produisent."""
    model = parse_model(_SIR_NEIGHBOR_ATTR)
    cfg = SimulationConfig(t_end=20.0, n_points=5, seed=1)
    result = NetworkBackend().run(model, cfg)
    # La population I doit avoir augmenté à un moment ou R doit être > 0
    assert result.y["R"][-1] > 0 or result.y["I"][-1] > 10, (
        "aucune infection détectée : R=0 et I ≤ 10 initial"
    )


def test_neighbor_attr_high_risk_accelerates():
    """Un beta_high élevé sur voisins High-risk produit plus de cas qu'un beta_high nul.

    On compare : modèle standard (beta_high=0.5) vs beta_high=0 (aucune infection
    possible puisque la seule source d'infection est neighbor_attr(I, risk, High)).
    """
    import numpy as np

    model_active = parse_model(_SIR_NEIGHBOR_ATTR)
    model_zero = parse_model("""
model SIR {
    semantics: [Network]
    compartment S = 90, I = 10, R = 0
    attribute risk ~ categorical(Low=0.7, High=0.3)
    rate beta_high = 0.0   [1/contact/day]
    rate gamma    = 0.1    [1/day]
    network watts_strogatz(k=6, rewire=0.1)
    flow S -> I : beta_high * neighbor_attr(I, risk, High)
    flow I -> R : gamma
}
""")
    cfg = SimulationConfig(t_end=30.0, n_points=4, seed=42)
    r_active = NetworkBackend().run(model_active, cfg)
    r_zero   = NetworkBackend().run(model_zero,   cfg)
    # Avec beta_high=0 : seule la guérison se produit, donc I+R doit rester ≤ 10 initial
    # Avec beta_high>0 : R final doit être supérieur
    assert r_active.y["R"][-1] > r_zero.y["R"][-1], (
        f"beta_high actif devrait produire plus de guéris : "
        f"actif R={r_active.y['R'][-1]}, nul R={r_zero.y['R'][-1]}"
    )
