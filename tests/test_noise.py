"""Tests for observation noise in observables."""

import numpy as np
import pytest

from epidsl.backends.gillespie import GillespieBackend
from epidsl.backends.ode import ODEBackend
from epidsl.backends.tau_leap import TauLeapBackend
from epidsl.calibration import Observations, calibrate, distance
from epidsl.ir import EpiModel, SimulationConfig
from epidsl.parser import parse


def model(body: str) -> EpiModel:
    return parse(f"model M {{ semantics: [ODE, Gillespie, TauLeap] {body} }}")[0]


# ---------------------------------------------------------------------------
# Parsing and IR
# ---------------------------------------------------------------------------


def test_noise_parsing() -> None:
    m = model("compartment X = 10 flow -> X : 1  observe O = X ~ poisson()")
    obs = m.observables[0]
    assert obs.noise is not None
    assert obs.noise.dist == "poisson"
    assert obs.noise.params == ()

    m2 = model("parameter s = 2 observe O = s ~ normal(2.0)")
    obs2 = m2.observables[0]
    assert obs2.noise is not None
    assert obs2.noise.dist == "normal"
    assert obs2.noise.params == (2.0,)

    m3 = model("observe O = 1 ~ poisson()")
    obs3 = m3.observables[0]
    assert obs3.noise is not None
    assert obs3.noise.dist == "poisson"
    assert obs3.noise.params == ()


def test_noise_no_noise_field_is_none() -> None:
    m = model("compartment X = 10  observe O = X")
    assert m.observables[0].noise is None


def test_noise_unknown_distribution_rejected() -> None:
    with pytest.raises(Exception, match="gamma"):
        parse("model M { semantics: [ODE] observe O = 1 ~ gamma(1.0) }")


# ---------------------------------------------------------------------------
# Check C13
# ---------------------------------------------------------------------------


def test_noise_check_c13() -> None:
    from epidsl.checks import check

    # normal() without its required sd parameter
    m1 = model("compartment X = 10  observe O = X ~ normal()")
    assert any(d.code == "C13" for d in check(m1))

    # negbinomial with non-positive dispersion
    m2 = model("compartment X = 10  observe O = X ~ negbinomial(-0.1)")
    assert any(d.code == "C13" for d in check(m2))

    # poisson with an extraneous parameter
    m3 = model("compartment X = 10  observe O = X ~ poisson(1.0)")
    assert any(d.code == "C13" for d in check(m3))

    # well-formed distributions produce no C13
    m4 = model("compartment X = 10  observe O = X ~ poisson()")
    assert not any(d.code == "C13" for d in check(m4))

    m5 = model("compartment X = 10  observe O = X ~ normal(0.5)")
    assert not any(d.code == "C13" for d in check(m5))

    m6 = model("compartment X = 10  observe O = X ~ negbinomial(2.0)")
    assert not any(d.code == "C13" for d in check(m6))


# ---------------------------------------------------------------------------
# Calibration with noise
# ---------------------------------------------------------------------------

_CALIB_SRC = """
model Epidemic {{
    compartment S = {S}, I = {I}, R = 0
    parameter beta = {beta0} prior uniform({lo}, {hi}) [1/day]
    rate gamma = 0.1 [1/day]
    flow S -> I : beta * S * I
    flow I -> R : gamma * I
    observe cases = incidence(S -> I) ~ {dist}
    semantics: [ODE]
}}
"""


def test_noise_calibration_poisson() -> None:
    m = parse(_CALIB_SRC.format(S=999, I=1, beta0="0.0002", lo="0.0001", hi="0.0005", dist="poisson()"))[0]
    obs = Observations(
        t=np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        y={"cases": np.array([0.0, 2.0, 5.0, 10.0, 18.0, 30.0])},
    )
    result = calibrate(ODEBackend(), m, obs, particles=10, populations=2, seed=42)
    assert result.names == ["beta"]
    assert result.particles.shape == (10, 1)
    assert all(result.particles[:, 0] > 0)


def test_noise_calibration_normal() -> None:
    m = parse(_CALIB_SRC.format(S=999, I=1, beta0="0.0002", lo="0.0001", hi="0.0005", dist="normal(5.0)"))[0]
    obs = Observations(
        t=np.array([0.0, 2.0, 5.0, 10.0]),
        y={"cases": np.array([0.0, 5.0, 25.0, 80.0])},
    )
    result = calibrate(ODEBackend(), m, obs, particles=10, populations=2, seed=0)
    assert result.names == ["beta"]
    assert result.particles.shape == (10, 1)


def test_noise_calibration_negbinomial() -> None:
    m = parse(_CALIB_SRC.format(S=999, I=1, beta0="0.0002", lo="0.0001", hi="0.0005", dist="negbinomial(2.0)"))[0]
    obs = Observations(
        t=np.array([0.0, 2.0, 5.0, 10.0]),
        y={"cases": np.array([0.0, 5.0, 25.0, 80.0])},
    )
    result = calibrate(ODEBackend(), m, obs, particles=10, populations=2, seed=1)
    assert result.names == ["beta"]
    assert result.particles.shape == (10, 1)


def test_noise_distance_pure_rmse_unchanged() -> None:
    """Without noise, distance() must still return RMSE (backward-compatible)."""
    from epidsl.backends import BackendResult

    t = np.array([0.0, 1.0, 2.0])
    result = BackendResult(
        t=t,
        y={"X": np.array([10.0, 20.0, 30.0])},
        observed={},
    )
    obs = Observations(t=t, y={"X": np.array([10.0, 20.0, 30.0])})
    assert distance(result, obs) == pytest.approx(0.0)

    obs2 = Observations(t=t, y={"X": np.array([10.0, 10.0, 10.0])})
    d = distance(result, obs2)
    assert d > 0.0


# ---------------------------------------------------------------------------
# Stochastic noise sampling
# ---------------------------------------------------------------------------


_NOISE_MODEL = """
model Noisy {{
    compartment S = 990, I = 10, R = 0
    rate beta = 0.0003 [1/day]
    rate gamma = 0.1 [1/day]
    flow S -> I : beta * S * I
    flow I -> R : gamma * I
    observe cases = incidence(S -> I) ~ {dist}
    observe active = I ~ {dist2}
    semantics: [ODE, Gillespie, TauLeap]
}}
"""


def _ode_values(dist: str, dist2: str) -> tuple[np.ndarray, np.ndarray]:
    m = parse(_NOISE_MODEL.format(dist=dist, dist2=dist2))[0]
    r = ODEBackend().run(m, SimulationConfig(t_end=10.0, n_points=11))
    return r.observed["cases"], r.observed["active"]


@pytest.mark.parametrize(
    "backend_cls,ids",
    [(GillespieBackend, "Gillespie"), (TauLeapBackend, "TauLeap")],
    ids=["Gillespie", "TauLeap"],
)
def test_stochastic_noise_poisson_mean_matches_ode(backend_cls: type, ids: str) -> None:
    """With Poisson noise, the mean over many runs ≈ the ODE value."""
    m = parse(_NOISE_MODEL.format(dist="poisson()", dist2="poisson()"))[0]
    ode_cases, ode_active = _ode_values("poisson()", "poisson()")

    runs = 80
    mean_cases = np.zeros(11)
    mean_active = np.zeros(11)
    for seed in range(runs):
        r = backend_cls().run(m, SimulationConfig(t_end=10.0, n_points=11, seed=seed))
        mean_cases += r.observed["cases"] / runs
        mean_active += r.observed["active"] / runs

    # Mean should be close to ODE (within 3 sigma for a Poisson draw)
    # At the peak, active ~ 50 → std of mean ≈ sqrt(50/80) ≈ 0.79
    np.testing.assert_allclose(mean_cases[1:], ode_cases[1:], atol=4.0, rtol=0.3)
    np.testing.assert_allclose(mean_active, ode_active, atol=5.0, rtol=0.3)


def test_stochastic_noise_values_are_integers_for_poisson() -> None:
    """Poisson draws are non-negative integers."""
    m = parse(_NOISE_MODEL.format(dist="poisson()", dist2="poisson()"))[0]
    for backend in [GillespieBackend(), TauLeapBackend()]:
        r = backend.run(m, SimulationConfig(t_end=10.0, n_points=11, seed=7))
        for name in ("cases", "active"):
            vals = r.observed[name]
            assert vals.min() >= 0.0
            np.testing.assert_array_equal(vals, np.round(vals))


def test_ode_noise_returns_deterministic_mean() -> None:
    """The ODE backend always returns the deterministic value regardless of noise."""
    m_noise = parse(_NOISE_MODEL.format(dist="poisson()", dist2="negbinomial(2.0)"))[0]
    m_plain = parse(
        "model Noisy { compartment S = 990, I = 10, R = 0 "
        "rate beta = 0.0003 [1/day] rate gamma = 0.1 [1/day] "
        "flow S -> I : beta * S * I  flow I -> R : gamma * I "
        "observe cases = incidence(S -> I)  observe active = I "
        "semantics: [ODE] }"
    )[0]
    cfg = SimulationConfig(t_end=10.0, n_points=11)
    r_noise = ODEBackend().run(m_noise, cfg)
    r_plain = ODEBackend().run(m_plain, cfg)
    np.testing.assert_allclose(r_noise.observed["cases"], r_plain.observed["cases"])
    np.testing.assert_allclose(r_noise.observed["active"], r_plain.observed["active"])


def test_same_seed_gives_same_noisy_trajectory() -> None:
    """With a fixed seed the noisy trajectory is reproducible."""
    m = parse(_NOISE_MODEL.format(dist="poisson()", dist2="negbinomial(3.0)"))[0]
    cfg = SimulationConfig(t_end=10.0, n_points=11, seed=99)
    for backend in [GillespieBackend(), TauLeapBackend()]:
        a = backend.run(m, cfg)
        b = backend.run(m, cfg)
        np.testing.assert_array_equal(a.observed["cases"], b.observed["cases"])
        np.testing.assert_array_equal(a.observed["active"], b.observed["active"])
