import csv
from pathlib import Path

import numpy as np
import pytest

from epidsl.backends import BackendResult
from epidsl.backends.gillespie import GillespieBackend
from epidsl.backends.ode import ODEBackend
from epidsl.calibration import (
    Observations,
    calibrate,
    distance,
    read_observations,
    with_constants,
)
from epidsl.checks import Severity, check
from epidsl.cli import main
from epidsl.ir import EpiModel, Prior, SimulationConfig
from epidsl.parser import EpiSyntaxError, parse

SIR = """
model SIR {
    semantics: [ODE, Gillespie]
    compartment S = 990, I = 10, R = 0
    parameter beta = 0.5 prior uniform(0.1, 1.0) [1/day]
    BLOCK
    flow S -> I : beta * S * I / N
    flow I -> R : gamma * I
}
"""
ONE = SIR.replace("BLOCK", "rate gamma = 0.1 [1/day]")
TWO = SIR.replace("BLOCK", "parameter gamma = 0.2 prior loguniform(0.02, 0.5) [1/day]")
FAST = ODEBackend(rtol=1e-6, atol=1e-6)


def model(source: str) -> EpiModel:
    return parse(source)[-1]


def synthetic(source: str, truth: dict[str, float], noise: float = 0.0) -> Observations:
    """Observations of I and R from the model run at ``truth``, with optional noise."""
    m = with_constants(model(source), truth)
    result = ODEBackend().run(m, SimulationConfig(t_end=80.0, n_points=21))
    rng = np.random.default_rng(0)
    return Observations(
        t=result.t,
        y={
            name: result.y[name] + noise * rng.standard_normal(result.t.shape)
            for name in ("I", "R")
        },
    )


# --- priors in the language ------------------------------------------------------------


def test_prior_is_parsed() -> None:
    beta, gamma = model(TWO).constants
    assert beta.prior == Prior("uniform", 0.1, 1.0)
    assert gamma.prior == Prior("loguniform", 0.02, 0.5)
    assert (beta.value, beta.unit) == (0.5, "1/day")
    assert model(ONE).constants[1].prior is None


def test_unknown_distribution_is_a_syntax_error() -> None:
    with pytest.raises(EpiSyntaxError, match="unknown distribution 'gamma'"):
        parse("model M { compartment A = 1, B = 0  parameter p = 1 prior gamma(1, 2)  flow A -> B : p * A }")


def test_rate_cannot_have_a_prior() -> None:
    with pytest.raises(EpiSyntaxError):
        parse("model M { compartment A = 1, B = 0  rate p = 1 prior uniform(0, 2)  flow A -> B : p * A }")


@pytest.mark.parametrize(
    ("declaration", "fragment"),
    [
        ("parameter p = 1 prior uniform(2, 1)", "lower bound must be below"),
        ("parameter p = 1 prior loguniform(0, 2)", "bounds must be positive"),
        ("parameter p = 1 prior normal(1, 0)", "standard deviation must be positive"),
        ("parameter p = 5 prior uniform(0, 2)", "value 5 lies outside its prior"),
        ("parameter p = 5 prior loguniform(1, 2)", "value 5 lies outside its prior"),
    ],
)
def test_c11_invalid_prior(declaration: str, fragment: str) -> None:
    m = model(f"model M {{ compartment A = 1, B = 0  {declaration} [1/day]  flow A -> B : p * A }}")
    (diagnostic,) = check(m)
    assert (diagnostic.code, diagnostic.severity) == ("C11", Severity.ERROR)
    assert fragment in diagnostic.message


def test_valid_priors_are_clean() -> None:
    for prior in ("uniform(0, 2)", "loguniform(0.5, 2)", "normal(3, 2)"):
        m = model(
            f"model M {{ compartment A = 1, B = 0  parameter p = 1 prior {prior} [1/day] "
            "flow A -> B : p * A }"
        )
        assert check(m) == []


def test_prior_density_and_sampling() -> None:
    rng = np.random.default_rng(0)
    uniform, loguniform, normal = (
        Prior("uniform", 2.0, 4.0),
        Prior("loguniform", 1.0, 100.0),
        Prior("normal", 1.0, 2.0),
    )

    assert uniform.pdf(3.0) == 0.5 and uniform.pdf(5.0) == 0.0
    assert loguniform.pdf(10.0) == pytest.approx(1 / (10 * np.log(100.0)))
    assert loguniform.pdf(0.5) == 0.0
    assert normal.pdf(1.0) == pytest.approx(1 / (2 * np.sqrt(2 * np.pi)))

    draws = np.array([uniform.sample(rng) for _ in range(4000)])
    assert draws.min() >= 2.0 and draws.max() <= 4.0
    assert draws.mean() == pytest.approx(3.0, abs=0.05)
    logs = np.log10([loguniform.sample(rng) for _ in range(4000)])
    assert logs.min() >= 0.0 and logs.max() <= 2.0
    assert logs.mean() == pytest.approx(1.0, abs=0.05)
    normals = np.array([normal.sample(rng) for _ in range(4000)])
    assert normals.mean() == pytest.approx(1.0, abs=0.15)
    assert normals.std() == pytest.approx(2.0, abs=0.1)


# --- building blocks -------------------------------------------------------------------


def test_with_constants_replaces_values_only() -> None:
    m = model(ONE)
    changed = with_constants(m, {"beta": 0.9})
    assert [(k.name, k.value) for k in changed.constants] == [("beta", 0.9), ("gamma", 0.1)]
    assert changed.constants[0].prior == m.constants[0].prior
    assert m.constants[0].value == 0.5
    with pytest.raises(ValueError, match="no constant named 'nope'"):
        with_constants(m, {"nope": 1.0})


def test_distance_is_zero_at_the_truth_and_grows_away_from_it() -> None:
    observations = synthetic(ONE, {"beta": 0.3})
    config = SimulationConfig(t_end=80.0, n_points=401)

    def at(beta: float) -> float:
        return distance(FAST.run(with_constants(model(ONE), {"beta": beta}), config), observations)

    assert at(0.3) < 1e-3
    assert at(0.3) < at(0.35) < at(0.5)
    assert at(0.3) < at(0.25) < at(0.15)


def test_distance_weighs_series_by_their_own_scale() -> None:
    t = np.array([0.0, 1.0])
    observations = Observations(t=t, y={"A": np.array([1000.0, 1000.0]), "B": np.array([1.0, 1.0])})

    def result(a: float, b: float) -> BackendResult:
        return BackendResult(t=t, y={"A": np.full(2, a), "B": np.full(2, b)})

    # A 10% error counts the same whatever the size of the series.
    assert distance(result(1100.0, 1.0), observations) == pytest.approx(
        distance(result(1000.0, 1.1), observations)
    )
    assert distance(result(1000.0, 1.0), observations) == 0.0


def test_read_observations(tmp_path: Path) -> None:
    path = tmp_path / "obs.csv"
    path.write_text("t,I,R\n0,10,0\n5,40.5,12\n")
    observations = read_observations(path)
    assert list(observations.t) == [0.0, 5.0]
    assert list(observations.y) == ["I", "R"]
    assert list(observations.y["I"]) == [10.0, 40.5]

    for text, fragment in [
        ("x,I\n0,1\n", "first column must be 't'"),
        ("t\n0\n", "at least one observed column"),
        ("t,I\n", "no rows"),
        ("t,I\n0,abc\n", "line 2"),
        ("t,I\n5,1\n0,2\n", "increasing"),
        ("t,I\n-1,1\n", "negative"),
    ]:
        path.write_text(text)
        with pytest.raises(ValueError, match=fragment):
            read_observations(path)


# --- calibration -----------------------------------------------------------------------


def test_recovers_one_parameter_from_noisy_data() -> None:
    observations = synthetic(ONE, {"beta": 0.3}, noise=3.0)
    result = calibrate(FAST, model(ONE), observations, particles=100, populations=5, seed=1)

    assert result.names == ["beta"]
    assert result.particles.shape == (100, 1)
    assert result.weights.sum() == pytest.approx(1.0)
    assert 50 < result.effective_sample_size <= 100
    low, high = result.quantile(0.025)["beta"], result.quantile(0.975)["beta"]
    assert low < 0.3 < high
    assert result.mean()["beta"] == pytest.approx(0.3, rel=0.05)
    # The posterior is far narrower than the prior, uniform on [0.1, 1].
    assert high - low < 0.1


def test_tolerance_shrinks_from_one_population_to_the_next() -> None:
    observations = synthetic(ONE, {"beta": 0.3}, noise=3.0)
    result = calibrate(FAST, model(ONE), observations, particles=60, populations=4, seed=2)

    assert len(result.tolerances) == 4
    assert result.tolerances[0] == np.inf
    assert result.tolerances[1] > result.tolerances[2] > result.tolerances[3] > 0
    assert result.simulations >= 4 * 60


def test_recovers_two_parameters() -> None:
    truth = {"beta": 0.4, "gamma": 0.08}
    observations = synthetic(TWO, truth)
    result = calibrate(FAST, model(TWO), observations, particles=120, populations=6, seed=3)

    assert result.names == ["beta", "gamma"]
    for name, value in truth.items():
        assert result.quantile(0.025)[name] < value < result.quantile(0.975)[name]
        assert result.mean()[name] == pytest.approx(value, rel=0.1)


def test_same_seed_gives_same_result() -> None:
    observations = synthetic(ONE, {"beta": 0.3})
    a = calibrate(FAST, model(ONE), observations, particles=30, populations=3, seed=4)
    b = calibrate(FAST, model(ONE), observations, particles=30, populations=3, seed=4)
    c = calibrate(FAST, model(ONE), observations, particles=30, populations=3, seed=5)
    assert np.array_equal(a.particles, b.particles) and np.array_equal(a.weights, b.weights)
    assert not np.array_equal(a.particles, c.particles)


def test_stochastic_calibration_is_reproducible() -> None:
    observations = synthetic(ONE, {"beta": 0.3})

    def run(seed: int) -> np.ndarray:
        backend = GillespieBackend()
        return calibrate(
            backend, model(ONE), observations, particles=20, populations=2, seed=seed
        ).particles

    assert np.array_equal(run(8), run(8))
    assert not np.array_equal(run(8), run(9))


class SameEveryTime:
    """A simulator whose output ignores the parameters: the data then say nothing."""

    def run(self, model: EpiModel, config: SimulationConfig) -> BackendResult:
        return BackendResult(t=np.array([0.0, config.t_end]), y={"A": np.array([10.0, 9.0])})


def test_weights_restore_the_prior_when_data_say_nothing() -> None:
    # Every proposal inside the prior is accepted, so the posterior must equal the
    # prior. Perturbation alone would reshape the particles; the importance weights are
    # what bring them back. (A real solver is not used here: its rounding error depends
    # on the parameter, and the tolerance would select on that noise.)
    source = (
        "model M { compartment A = 10 "
        "parameter p = 1 prior loguniform(0.01, 100) [1/day]  flow A -> : p * A }"
    )
    observations = Observations(t=np.array([0.0, 1.0]), y={"A": np.array([10.0, 9.0])})
    result = calibrate(
        SameEveryTime(), model(source), observations, particles=3000, populations=3, seed=0
    )

    logs = np.log10(result.particles[:, 0])
    mean = float(result.weights @ logs)
    variance = float(result.weights @ (logs - mean) ** 2)
    # log10(p) is uniform on [-2, 2]: mean 0, variance 16 / 12.
    assert mean == pytest.approx(0.0, abs=0.1)
    assert variance == pytest.approx(16 / 12, rel=0.1)
    assert result.weights.max() > 1.2 * result.weights.min()
    # The weights must not degenerate: this failed when a parameter spanning four
    # orders of magnitude was perturbed on a linear scale.
    assert result.effective_sample_size > 0.8 * 3000


def test_weights_restore_a_normal_prior() -> None:
    # Same idea with a prior whose density is not constant: dropping the prior from the
    # weights would flatten the result.
    source = (
        "model M { compartment A = 10 "
        "parameter p = 1 prior normal(1, 2) [1/day]  flow A -> : p * A }"
    )
    observations = Observations(t=np.array([0.0, 1.0]), y={"A": np.array([10.0, 9.0])})
    result = calibrate(
        SameEveryTime(), model(source), observations, particles=3000, populations=4, seed=0
    )

    values = result.particles[:, 0]
    mean = float(result.weights @ values)
    spread = float(np.sqrt(result.weights @ (values - mean) ** 2))
    assert mean == pytest.approx(1.0, abs=0.15)
    assert spread == pytest.approx(2.0, rel=0.08)
    # About 74% here: the perturbation is wider than the prior, so weights vary.
    assert result.effective_sample_size > 0.6 * 3000


def test_particles_stay_inside_the_prior() -> None:
    # The truth sits on the edge of the prior, so proposals often fall outside it.
    observations = synthetic(ONE, {"beta": 0.1})
    # Each population halves the tolerance, so seven of them shrink the prior's width
    # of 0.9 by about 2^6.
    result = calibrate(FAST, model(ONE), observations, particles=60, populations=7, seed=6)
    assert result.particles.min() >= 0.1 and result.particles.max() <= 1.0
    assert result.mean()["beta"] < 0.12


def test_stochastic_backend_can_be_calibrated() -> None:
    observations = synthetic(ONE, {"beta": 0.3})
    result = calibrate(
        GillespieBackend(), model(ONE), observations, particles=40, populations=3, seed=7
    )
    assert result.quantile(0.025)["beta"] < 0.3 < result.quantile(0.975)["beta"]
    assert result.mean()["beta"] == pytest.approx(0.3, rel=0.25)


def test_refusals() -> None:
    observations = synthetic(ONE, {"beta": 0.3})
    no_prior = model(ONE.replace(" prior uniform(0.1, 1.0)", ""))
    with pytest.raises(ValueError, match="no parameter with a prior"):
        calibrate(FAST, no_prior, observations)

    unknown = Observations(t=observations.t, y={"Z": observations.y["I"]})
    with pytest.raises(ValueError, match="'Z' is not a compartment"):
        calibrate(FAST, model(ONE), unknown)

    with pytest.raises(ValueError, match="at least 2 particles"):
        calibrate(FAST, model(ONE), observations, particles=1)
    with pytest.raises(ValueError, match="at least 1 population"):
        calibrate(FAST, model(ONE), observations, populations=0)

    broken = model(ONE.replace("gamma * I", "nope * I"))
    with pytest.raises(ValueError, match="C3"):
        calibrate(FAST, broken, observations)

    with pytest.raises(RuntimeError, match="more than 50 simulations"):
        calibrate(FAST, model(ONE), observations, particles=40, populations=4, max_simulations=50)


# --- command line ----------------------------------------------------------------------


def write_data(path: Path, observations: Observations) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["t", *observations.y])
        for i, t in enumerate(observations.t):
            writer.writerow([t, *(series[i] for series in observations.y.values())])


def test_cli_calibrate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source, data, out = tmp_path / "sir.epidsl", tmp_path / "obs.csv", tmp_path / "post.csv"
    source.write_text(ONE)
    write_data(data, synthetic(ONE, {"beta": 0.3}, noise=3.0))
    args = ["calibrate", str(source), "--data", str(data), "--particles", "60"]
    args += ["--populations", "4", "--seed", "1"]

    assert main([*args, "--out", str(out)]) == 0
    captured = capsys.readouterr()
    header, row = (line.split(",") for line in captured.out.splitlines())
    assert header == ["parameter", "mean", "q0.025", "q0.5", "q0.975"]
    assert row[0] == "beta"
    mean, low, median, high = map(float, row[1:])
    assert low < 0.3 < high and low < median < high
    assert mean == pytest.approx(0.3, rel=0.05)
    assert "simulations" in captured.err and "effective sample size" in captured.err

    particles = list(csv.reader(out.open()))
    assert particles[0] == ["weight", "beta"]
    assert len(particles) == 61
    assert sum(float(r[0]) for r in particles[1:]) == pytest.approx(1.0)

    assert main(args) == 0
    assert capsys.readouterr().out == captured.out


def test_cli_calibrate_reports_problems(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source, data = tmp_path / "sir.epidsl", tmp_path / "obs.csv"
    source.write_text(ONE)

    assert main(["calibrate", str(source), "--data", str(data)]) == 1
    assert str(data) in capsys.readouterr().err

    data.write_text("t,Z\n0,1\n1,2\n")
    assert main(["calibrate", str(source), "--data", str(data)]) == 1
    assert "'Z' is not a compartment" in capsys.readouterr().err

    source.write_text(ONE.replace(" prior uniform(0.1, 1.0)", ""))
    data.write_text("t,I\n0,10\n1,12\n")
    assert main(["calibrate", str(source), "--data", str(data)]) == 1
    assert "no parameter with a prior" in capsys.readouterr().err
