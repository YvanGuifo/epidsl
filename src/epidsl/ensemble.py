"""Ensembles: many independent runs of a stochastic backend, and their summaries."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from epidsl.backends import Backend
from epidsl.ir import EpiModel, SimulationConfig


@dataclass(frozen=True, slots=True)
class EnsembleResult:
    """Trajectories of several runs on a common time grid.

    ``samples[r, c, k]`` is compartment ``names[c]`` in run ``r`` at time ``t[k]``.
    ``seeds[r]`` reproduces run ``r`` when passed to the backend directly.
    """

    t: NDArray[np.float64]
    names: list[str]
    samples: NDArray[np.float64]
    seeds: list[int]

    def mean(self) -> dict[str, NDArray[np.float64]]:
        """Mean over runs, per compartment and time."""
        values = self.samples.mean(axis=0)
        return {name: values[c] for c, name in enumerate(self.names)}

    def quantile(self, q: float) -> dict[str, NDArray[np.float64]]:
        """Quantile ``q`` (between 0 and 1) over runs, per compartment and time."""
        if not 0 <= q <= 1:
            raise ValueError(f"a quantile must be between 0 and 1, got {q:g}")
        values = np.quantile(self.samples, q, axis=0)
        return {name: values[c] for c, name in enumerate(self.names)}


def run_ensemble(
    backend: Backend, model: EpiModel, config: SimulationConfig, runs: int
) -> EnsembleResult:
    """Run ``backend`` ``runs`` times with independent seeds derived from ``config.seed``.

    The same ``config.seed`` always gives the same ensemble; None draws fresh seeds.
    """
    if runs < 1:
        raise ValueError("an ensemble needs at least one run")
    seeds = [int(s) for s in np.random.SeedSequence(config.seed).generate_state(runs, np.uint64)]
    results = [backend.run(model, replace(config, seed=seed)) for seed in seeds]
    names = list(results[0].y)
    return EnsembleResult(
        t=results[0].t,
        names=names,
        samples=np.array([[result.y[name] for name in names] for result in results]),
        seeds=seeds,
    )
