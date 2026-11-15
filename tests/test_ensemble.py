from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binom

from epidsl.backends.gillespie import GillespieBackend
from epidsl.cli import main
from epidsl.ensemble import run_ensemble
from epidsl.ir import EpiModel, SimulationConfig
from epidsl.parser import parse

EXAMPLES = Path(__file__).parent.parent / "examples"
SIR = str(EXAMPLES / "sir.epidsl")

DEATH = (
    "model D { semantics: [Gillespie] compartment X = 50, Y = 0 "
    "rate k = 1 [1/day]  flow X -> Y : k * X }"
)


def death() -> EpiModel:
    return parse(DEATH)[0]


def config(seed: int | None = 0) -> SimulationConfig:
    return SimulationConfig(t_end=1.0, n_points=3, seed=seed)


def test_shape_names_and_time_grid() -> None:
    ensemble = run_ensemble(GillespieBackend(), death(), config(), runs=7)

    assert ensemble.names == ["X", "Y"]
    assert ensemble.samples.shape == (7, 2, 3)
    assert list(ensemble.t) == [0.0, 0.5, 1.0]
    assert len(ensemble.seeds) == 7


def test_runs_use_distinct_reproducible_seeds() -> None:
    a = run_ensemble(GillespieBackend(), death(), config(seed=1), runs=20)
    b = run_ensemble(GillespieBackend(), death(), config(seed=1), runs=20)
    c = run_ensemble(GillespieBackend(), death(), config(seed=2), runs=20)

    assert len(set(a.seeds)) == 20
    assert a.seeds == b.seeds and np.array_equal(a.samples, b.samples)
    assert a.seeds != c.seeds and not np.array_equal(a.samples, c.samples)
    # The trajectories themselves differ from one run to the next.
    assert len({tuple(run[0]) for run in a.samples}) > 1


def test_each_run_is_a_plain_backend_run_with_its_seed() -> None:
    ensemble = run_ensemble(GillespieBackend(), death(), config(seed=3), runs=4)
    for i, seed in enumerate(ensemble.seeds):
        single = GillespieBackend().run(death(), config(seed=seed))
        assert np.array_equal(ensemble.samples[i, 0], single.y["X"])
        assert np.array_equal(ensemble.samples[i, 1], single.y["Y"])


def test_unseeded_ensembles_differ() -> None:
    a = run_ensemble(GillespieBackend(), death(), config(seed=None), runs=5)
    b = run_ensemble(GillespieBackend(), death(), config(seed=None), runs=5)
    assert a.seeds != b.seeds


def test_mean_and_quantiles_match_the_binomial_law() -> None:
    ensemble = run_ensemble(GillespieBackend(), death(), config(), runs=2000)
    n, p = 50, np.exp(-1.0)

    mean = ensemble.mean()
    assert mean["X"][0] == 50.0
    assert mean["X"][-1] == pytest.approx(n * p, abs=4 * np.sqrt(n * p * (1 - p) / 2000))
    np.testing.assert_allclose(mean["X"] + mean["Y"], 50.0)

    low, median, high = (ensemble.quantile(q)["X"][-1] for q in (0.025, 0.5, 0.975))
    assert low == pytest.approx(binom.ppf(0.025, n, p), abs=1)
    assert median == pytest.approx(binom.ppf(0.5, n, p), abs=1)
    assert high == pytest.approx(binom.ppf(0.975, n, p), abs=1)
    assert low < median < high


def test_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="at least one run"):
        run_ensemble(GillespieBackend(), death(), config(), runs=0)
    ensemble = run_ensemble(GillespieBackend(), death(), config(), runs=2)
    with pytest.raises(ValueError, match="between 0 and 1"):
        ensemble.quantile(1.5)


def rows(text: str) -> list[list[str]]:
    return [line.split(",") for line in text.splitlines()]


def test_cli_runs_writes_summary_columns(capsys: pytest.CaptureFixture[str]) -> None:
    args = ["run", SIR, "--t-end", "60", "--points", "4", "--method", "gillespie"]
    assert main([*args, "--runs", "30", "--seed", "1"]) == 0
    first = capsys.readouterr().out
    table = rows(first)

    assert table[0] == [
        "t",
        "S_mean",
        "S_q0.025",
        "S_q0.975",
        "I_mean",
        "I_q0.025",
        "I_q0.975",
        "R_mean",
        "R_q0.025",
        "R_q0.975",
    ]
    assert len(table) == 5
    for row in table[1:]:
        values = dict(zip(table[0], map(float, row), strict=True))
        assert values["S_mean"] + values["I_mean"] + values["R_mean"] == pytest.approx(1000.0)
        assert values["S_q0.025"] <= values["S_mean"] <= values["S_q0.975"]
    assert [float(x) for x in table[1]] == [
        0.0,
        990.0,
        990.0,
        990.0,
        10.0,
        10.0,
        10.0,
        0.0,
        0.0,
        0.0,
    ]
    assert float(table[-1][3]) > float(table[-1][2])

    assert main([*args, "--runs", "30", "--seed", "1"]) == 0
    assert capsys.readouterr().out == first


def test_cli_custom_quantiles(capsys: pytest.CaptureFixture[str]) -> None:
    args = ["run", SIR, "--t-end", "10", "--points", "2", "--method", "gillespie", "--seed", "1"]
    assert main([*args, "--runs", "5", "--quantiles", "0.5"]) == 0
    assert rows(capsys.readouterr().out)[0] == ["t", "S_mean", "S_q0.5", "I_mean", "I_q0.5"] + [
        "R_mean",
        "R_q0.5",
    ]


def test_cli_single_run_output_is_unchanged(capsys: pytest.CaptureFixture[str]) -> None:
    args = ["run", SIR, "--t-end", "10", "--points", "2", "--method", "gillespie", "--seed", "1"]
    assert main(args) == 0
    assert rows(capsys.readouterr().out)[0] == ["t", "S", "I", "R"]


def test_cli_epsilon_changes_tau_leap(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = tmp_path / "big.epidsl"
    src.write_text(
        "model Big { semantics: [TauLeap, Gillespie] compartment S = 99000, I = 1000, R = 0 "
        "rate b = 0.3 [1/day] rate g = 0.1 [1/day] "
        "flow S -> I : b * S * I / N  flow I -> R : g * I }"
    )
    args = ["run", str(src), "--t-end", "40", "--points", "3", "--seed", "1"]

    assert main([*args, "--method", "tau-leap", "--epsilon", "0.3"]) == 0
    coarse = capsys.readouterr().out
    assert main([*args, "--method", "tau-leap", "--epsilon", "0.01"]) == 0
    fine = capsys.readouterr().out
    assert coarse != fine

    assert main([*args, "--method", "gillespie", "--epsilon", "0.01"]) == 1
    assert "--epsilon applies only to --method tau-leap" in capsys.readouterr().err
    assert main([*args, "--method", "tau-leap", "--epsilon", "2"]) == 1
    assert "epsilon" in capsys.readouterr().err


def test_cli_refuses_runs_with_ode(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", SIR, "--t-end", "10", "--runs", "5"]) == 1
    assert "--runs needs a stochastic method" in capsys.readouterr().err


@pytest.mark.parametrize(
    "extra", [["--runs", "0"], ["--quantiles", "0.5,1.2"], ["--quantiles", "x"]]
)
def test_cli_rejects_bad_option_values(extra: list[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["run", SIR, "--t-end", "10", "--method", "gillespie", *extra])
    assert excinfo.value.code == 2
