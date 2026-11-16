"""Running simulations over several processes: speed may change, results may not."""

import os
from pathlib import Path

import numpy as np
import pytest

from epidsl.backends.gillespie import GillespieBackend
from epidsl.backends.ode import ODEBackend
from epidsl.calibration import Observations, calibrate, read_observations
from epidsl.cli import main
from epidsl.ensemble import run_ensemble
from epidsl.ir import EpiModel, SimulationConfig
from epidsl.parallel import Pool, resolve_workers, split
from epidsl.parser import parse
from epidsl.studio import api

EXAMPLES = Path(__file__).parent.parent / "examples"
SIR = (EXAMPLES / "sir.epidsl").read_text()


def model() -> EpiModel:
    return parse(SIR)[0]


def observations() -> Observations:
    return read_observations(EXAMPLES / "sir_observed.csv")


# --- splitting work --------------------------------------------------------------------


def test_split_keeps_order_and_balances() -> None:
    assert split([1, 2, 3, 4, 5], 2) == [[1, 2, 3], [4, 5]]
    assert split([1, 2, 3], 5) == [[1], [2], [3]]
    assert split([1, 2, 3], 1) == [[1, 2, 3]]
    assert split([], 4) == []
    joined = [item for chunk in split(list(range(97)), 7) for item in chunk]
    assert joined == list(range(97))
    assert max(len(chunk) for chunk in split(list(range(97)), 7)) <= 14


def test_resolve_workers() -> None:
    assert resolve_workers(3) == 3
    assert resolve_workers(0) == (os.cpu_count() or 1) >= 1
    with pytest.raises(ValueError, match="workers"):
        resolve_workers(-1)


def test_a_pool_gives_the_same_answer_with_any_worker_count() -> None:
    chunks = split([(SIR, "SIR", t) for t in range(1, 25)], 6)
    with Pool(1) as one, Pool(4) as many:
        alone, shared = one.map(_peaks_for_chunk, chunks), many.map(_peaks_for_chunk, chunks)
    assert alone == shared
    assert len(alone) == 24 and all(value > 0 for value in alone)


def test_a_pool_refuses_work_outside_its_block() -> None:
    pool = Pool(2)
    with pytest.raises(RuntimeError, match="with"):
        pool.map(_peaks_for_chunk, [[(SIR, "SIR", 10)]])


def _peaks_for_chunk(chunk: list[tuple[str, str, int]]) -> list[float]:
    """A picklable unit of work, so that the real process pool is exercised."""
    out = []
    for source, name, t_end in chunk:
        result = api.run({"source": source, "model": name, "t_end": t_end, "points": 5})
        out.append(result["series"][1]["mean"][-1])
    return out


# --- ensembles -------------------------------------------------------------------------


def test_workers_do_not_change_an_ensemble() -> None:
    config = SimulationConfig(t_end=40.0, n_points=9, seed=7)
    one = run_ensemble(GillespieBackend(), model(), config, runs=12, workers=1)
    many = run_ensemble(GillespieBackend(), model(), config, runs=12, workers=4)

    assert one.seeds == many.seeds
    assert np.array_equal(one.samples, many.samples)
    assert one.names == many.names


def test_an_ensemble_of_one_run_still_works_in_parallel() -> None:
    config = SimulationConfig(t_end=10.0, n_points=3, seed=1)
    ensemble = run_ensemble(GillespieBackend(), model(), config, runs=1, workers=4)
    assert ensemble.samples.shape == (1, 3, 3)


# --- calibration -----------------------------------------------------------------------


def calibrated(workers: int, seed: int = 3):
    return calibrate(
        ODEBackend(rtol=1e-6, atol=1e-6),
        model(),
        observations(),
        particles=24,
        populations=3,
        seed=seed,
        workers=workers,
    )


def test_workers_do_not_change_a_calibration() -> None:
    one = calibrated(1)
    many = calibrated(4)

    assert np.array_equal(one.particles, many.particles)
    assert np.array_equal(one.weights, many.weights)
    assert (one.simulations, one.tolerances) == (many.simulations, many.tolerances)
    assert one.names == many.names


def test_workers_do_not_change_a_stochastic_calibration() -> None:
    # Each proposal carries the seed of its own simulation, so a stochastic backend
    # gives the same answer however the proposals are shared out.
    def run(workers: int):
        return calibrate(
            GillespieBackend(),
            model(),
            observations(),
            particles=12,
            populations=2,
            seed=5,
            workers=workers,
        )

    assert np.array_equal(run(1).particles, run(4).particles)


def test_a_calibration_still_depends_on_its_seed() -> None:
    assert not np.array_equal(calibrated(4, seed=3).particles, calibrated(4, seed=4).particles)


def test_an_unseeded_calibration_is_reproducible_within_itself() -> None:
    # Two unseeded runs differ, but each is a valid posterior around the truth.
    first, second = calibrated(2, seed=None), calibrated(2, seed=None)
    assert not np.array_equal(first.particles, second.particles)
    for result in (first, second):
        assert result.quantile(0.025)["beta"] < 0.3 < result.quantile(0.975)["beta"]


def test_the_simulation_budget_counts_simulations_not_proposals() -> None:
    with pytest.raises(RuntimeError, match="more than 40 simulations"):
        calibrate(
            ODEBackend(),
            model(),
            observations(),
            particles=24,
            populations=3,
            seed=1,
            workers=2,
            max_simulations=40,
        )


def test_a_population_that_cannot_be_filled_gives_up(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("epidsl.calibration.MAX_PROPOSALS", 5)
    with pytest.raises(RuntimeError, match="5 proposals without filling a population"):
        calibrate(
            ODEBackend(), model(), observations(), particles=24, populations=2, seed=1, workers=2
        )


def test_workers_must_be_valid() -> None:
    with pytest.raises(ValueError, match="workers"):
        calibrate(ODEBackend(), model(), observations(), particles=4, populations=1, workers=-2)
    with pytest.raises(ValueError, match="workers"):
        run_ensemble(
            GillespieBackend(), model(), SimulationConfig(t_end=1.0, seed=1), runs=2, workers=-1
        )


# --- command line and studio -----------------------------------------------------------


def test_cli_workers(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = ["run", str(EXAMPLES / "sir.epidsl"), "--t-end", "20", "--points", "4"]
    args += ["--method", "gillespie", "--runs", "8", "--seed", "2"]
    assert main([*args, "--workers", "1"]) == 0
    one = capsys.readouterr().out
    assert main([*args, "--workers", "4"]) == 0
    assert capsys.readouterr().out == one

    out = tmp_path / "post.csv"
    cal = ["calibrate", str(EXAMPLES / "sir.epidsl"), "--data", str(EXAMPLES / "sir_observed.csv")]
    cal += ["--particles", "20", "--populations", "2", "--seed", "1", "--out", str(out)]
    assert main([*cal, "--workers", "3"]) == 0
    assert "beta" in capsys.readouterr().out


def test_cli_rejects_a_negative_worker_count() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["run", str(EXAMPLES / "sir.epidsl"), "--t-end", "10", "--workers", "-1"])
    assert excinfo.value.code == 2


def test_studio_passes_workers_through() -> None:
    payload = {
        "source": SIR,
        "model": "SIR",
        "t_end": 20,
        "points": 4,
        "method": "gillespie",
        "runs": 8,
        "seed": 2,
    }
    assert api.run({**payload, "workers": 3}) == api.run({**payload, "workers": 1})
    assert "workers" in api.run({**payload, "workers": 0.5})["error"]
    assert "workers" in api.run({**payload, "workers": 999})["error"]
