"""Calibration: estimate the parameters that carry a prior from observed data.

The method is approximate Bayesian computation with sequential Monte Carlo
(ABC-SMC). It needs no likelihood, only the ability to simulate, so it works with
every backend:

1. Draw parameter values from the prior, simulate, and measure the distance between
   each simulation and the data.
2. Set the tolerance to a quantile of those distances, then build the next population
   by perturbing values drawn from the current one and keeping those whose simulation
   falls within the tolerance.
3. Repeat. Each population is a weighted sample that approximates the posterior more
   closely as the tolerance shrinks.

The weights are those of Toni et al. (2009, J. R. Soc. Interface 6, 187) and the
perturbation kernel that of Beaumont et al. (2009, Biometrika 96, 983); tolerances
are adaptive. See D15 in docs/DESIGN_DECISIONS.md for the differences.

A parameter with a ``loguniform`` prior is handled on a logarithmic scale throughout:
a perturbation of fixed size on a linear scale would be far too large at the low end
of a range that spans several orders of magnitude, and the weights would degenerate.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from epidsl.backends import Backend, BackendResult
from epidsl.checks import errors
from epidsl.ir import EpiModel, Prior, SimulationConfig
from epidsl.parallel import Pool, split


@dataclass(frozen=True, slots=True)
class Observations:
    """Observed values of some compartments at increasing times."""

    t: NDArray[np.float64]
    y: dict[str, NDArray[np.float64]]


def read_observations(path: Path) -> Observations:
    """Read a CSV file whose first column is ``t`` and whose others are compartments."""
    return parse_observations(path.read_text(encoding="utf-8"))


def parse_observations(text: str) -> Observations:
    """Read CSV text whose first column is ``t`` and whose others are compartments."""
    rows = list(csv.reader(io.StringIO(text, newline="")))
    if not rows or not rows[0] or rows[0][0].strip() != "t":
        raise ValueError("the first column must be 't'")
    names = [name.strip() for name in rows[0][1:]]
    if not names:
        raise ValueError("the file needs at least one observed column besides 't'")
    body = [row for row in rows[1:] if row]
    if not body:
        raise ValueError("the file has no rows of data")

    values: list[list[float]] = []
    for number, row in enumerate(body, start=2):
        try:
            if len(row) != len(names) + 1:
                raise ValueError
            values.append([float(cell) for cell in row])
        except ValueError:
            raise ValueError(f"line {number} is not {len(names) + 1} numbers") from None
    table = np.array(values, dtype=np.float64)
    t = table[:, 0]
    if t[0] < 0:
        raise ValueError("times must not be negative")
    if np.any(np.diff(t) <= 0):
        raise ValueError("times must be increasing")
    return Observations(t=t, y={name: table[:, i + 1] for i, name in enumerate(names)})


def with_constants(model: EpiModel, values: dict[str, float]) -> EpiModel:
    """Return ``model`` with the given constants set to new values."""
    known = {k.name for k in model.constants}
    for name in values:
        if name not in known:
            raise ValueError(f"model {model.name} has no constant named '{name}'")
    constants = tuple(
        replace(k, value=values[k.name]) if k.name in values else k for k in model.constants
    )
    return replace(model, constants=constants)


def simulate_at(
    backend: Backend, model: EpiModel, times: NDArray[np.float64], seed: int | None
) -> BackendResult:
    """Simulate from time zero and report exactly at ``times``.

    An incidence observable at the first of ``times`` counts from time zero.
    """
    starts_at_zero = times[0] == 0
    grid = times if starts_at_zero else np.concatenate([[0.0], times])
    result = backend.run(model, SimulationConfig.at(grid.tolist(), seed=seed))
    if starts_at_zero:
        return result
    return BackendResult(
        t=result.t[1:],
        y={name: series[1:] for name, series in result.y.items()},
        metadata=result.metadata,
        observed={name: series[1:] for name, series in result.observed.items()},
    )


def distance(result: BackendResult, observations: Observations) -> float:
    """Root mean squared relative error between a simulation and the observations.

    Each series is divided by its largest observed magnitude, so that large and small
    quantities count alike. A compartment is interpolated to the observation times.
    An observable is not, since an incidence depends on the spacing of the output
    times: the simulation must then report at the observation times themselves.
    """
    same_times = len(result.t) == len(observations.t) and np.array_equal(result.t, observations.t)
    total = 0.0
    count = 0
    for name, observed in observations.y.items():
        if name in result.y:
            simulated = np.interp(observations.t, result.t, result.y[name])
        elif name not in result.observed:
            raise ValueError(f"the simulation has no compartment or observable '{name}'")
        elif not same_times:
            raise ValueError(
                f"observable '{name}' can only be compared at the same times as the simulation"
            )
        else:
            simulated = result.observed[name]
        scale = float(np.abs(observed).max()) or 1.0
        total += float((((simulated - observed) / scale) ** 2).sum())
        count += len(observed)
    return math.sqrt(total / count)


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """A weighted sample of parameter values approximating the posterior.

    ``particles[i, j]`` is the value of parameter ``names[j]`` in particle ``i``, with
    weight ``weights[i]``. ``tolerances`` lists the tolerance of each population (the
    first is infinite) and ``simulations`` counts the simulations run.
    """

    names: list[str]
    particles: NDArray[np.float64]
    weights: NDArray[np.float64]
    tolerances: list[float]
    simulations: int

    @property
    def effective_sample_size(self) -> float:
        """Number of equally weighted particles the sample is worth (at most its size)."""
        return float(1.0 / (self.weights**2).sum())

    def mean(self) -> dict[str, float]:
        """Posterior mean of each parameter."""
        return {
            name: float(self.weights @ self.particles[:, j]) for j, name in enumerate(self.names)
        }

    def quantile(self, q: float) -> dict[str, float]:
        """Posterior quantile ``q`` (between 0 and 1) of each parameter."""
        if not 0 <= q <= 1:
            raise ValueError(f"a quantile must be between 0 and 1, got {q:g}")
        return {
            name: _weighted_quantile(self.particles[:, j], self.weights, q)
            for j, name in enumerate(self.names)
        }


def _weighted_quantile(
    values: NDArray[np.float64], weights: NDArray[np.float64], q: float
) -> float:
    order = np.argsort(values)
    # Cumulative weight at the centre of each particle, as for an unweighted sample.
    centres = np.cumsum(weights[order]) - 0.5 * weights[order]
    return float(np.interp(q, centres, values[order]))


@dataclass(frozen=True, slots=True)
class _Population:
    """The population a new proposal is perturbed from."""

    thetas: NDArray[np.float64]
    weights: NDArray[np.float64]
    spread: NDArray[np.float64]


def _to_internal(prior: Prior, x: float) -> float:
    """The coordinate the sampler works in: the logarithm for a loguniform prior."""
    return math.log(x) if prior.dist == "loguniform" else x


def _from_internal(prior: Prior, u: float) -> float:
    return math.exp(u) if prior.dist == "loguniform" else u


def _internal_density(prior: Prior, u: float) -> float:
    """Prior density in the sampler's coordinate."""
    if prior.dist == "loguniform":
        low, high = math.log(prior.a), math.log(prior.b)
        return 1.0 / (high - low) if low <= u <= high else 0.0
    return prior.pdf(u)


#: A population gives up after this many proposals. A proposal that falls outside the
#: prior costs no simulation, so only a pathological prior ever reaches this.
MAX_PROPOSALS = 1_000_000

#: One proposal: its parameters, in the sampler's coordinates, and the seed of the
#: simulation that measures it.
Candidate = tuple[NDArray[np.float64], int]


def _measure(
    job: tuple[Backend, EpiModel, list[str], list[Prior], Observations, list[Candidate]],
) -> list[float]:
    """One worker's share: the distance to the data of each proposal it was given."""
    backend, model, names, priors, observations, candidates = job
    measured = []
    for theta, run_seed in candidates:
        values = {
            name: _from_internal(prior, float(u))
            for name, prior, u in zip(names, priors, theta, strict=True)
        }
        try:
            simulated = simulate_at(
                backend, with_constants(model, values), observations.t, run_seed
            )
            measured.append(distance(simulated, observations))
        except RuntimeError:
            measured.append(math.inf)  # the simulation failed here: never accepted
    return measured


def calibrate(
    backend: Backend,
    model: EpiModel,
    observations: Observations,
    *,
    particles: int = 200,
    populations: int = 5,
    seed: int | None = None,
    keep: float = 0.5,
    max_simulations: int = 1_000_000,
    workers: int = 1,
) -> CalibrationResult:
    """Estimate every ``parameter`` of ``model`` that has a prior, by ABC-SMC.

    Each new tolerance is the ``keep`` quantile of the previous population's distances.
    Within a population the proposals are independent, so ``workers`` spreads them over
    that many processes — 0 meaning one per processor — without changing the result:
    each proposal draws from a generator fixed by its place, not by the order run.

    Raises ValueError on unusable inputs and RuntimeError when more than
    ``max_simulations`` simulations would be needed.
    """
    problems = errors(model)
    if problems:
        details = "; ".join(str(p) for p in problems)
        raise ValueError(f"model {model.name} has errors: {details}")
    targets = [k for k in model.constants if k.prior is not None]
    if not targets:
        raise ValueError(
            f"model {model.name} has no parameter with a prior; "
            "declare one as 'parameter p = 0.3 prior uniform(0.1, 1)'"
        )
    known = {c.name for c in model.compartments} | {o.name for o in model.observables}
    for name in observations.y:
        if name not in known:
            raise ValueError(
                f"observed column '{name}' is not a compartment or observable of model {model.name}"
            )
    if particles < 2:
        raise ValueError("calibration needs at least 2 particles")
    if populations < 1:
        raise ValueError("calibration needs at least 1 population")

    names = [k.name for k in targets]
    priors: list[Prior] = [k.prior for k in targets if k.prior is not None]
    master = np.random.SeedSequence(seed)
    simulations = 0

    def prior_density(theta: NDArray[np.float64]) -> float:
        return math.prod(
            _internal_density(prior, float(u)) for prior, u in zip(priors, theta, strict=True)
        )

    def propose(population: int, index: int, previous: _Population | None) -> Candidate | None:
        """The proposal at ``index``, or None when it falls outside the prior.

        Its generator depends on where the proposal sits, so neither the number of
        workers nor the order the proposals are measured in can reach the result.
        """
        rng = np.random.default_rng(
            np.random.SeedSequence(master.entropy, spawn_key=(population, index))
        )
        if previous is None:
            theta = np.array(
                [_to_internal(prior, prior.sample(rng)) for prior in priors], dtype=np.float64
            )
        else:
            origin = previous.thetas[rng.choice(particles, p=previous.weights)]
            theta = origin + previous.spread * rng.standard_normal(len(names))
            if prior_density(theta) == 0.0:
                return None
        return theta, int(rng.integers(0, 2**63))

    def measure(candidates: list[Candidate], pool: Pool) -> list[float]:
        nonlocal simulations
        if simulations + len(candidates) > max_simulations:
            raise RuntimeError(
                f"calibration of model {model.name} needs more than {max_simulations} simulations"
            )
        simulations += len(candidates)
        jobs = [
            (backend, model, names, priors, observations, chunk)
            for chunk in split(candidates, pool.workers * 4)
        ]
        return pool.map(_measure, jobs)

    def fill(
        population: int, previous: _Population | None, tolerance: float, pool: Pool
    ) -> list[tuple[NDArray[np.float64], float]]:
        """Measure proposals in batches until ``particles`` of them are close enough."""
        taken: list[tuple[NDArray[np.float64], float]] = []
        index = 0
        while len(taken) < particles:
            batch: list[Candidate] = []
            # Exactly as many as are still wanted, so that the count of simulations,
            # like the particles themselves, does not depend on how many workers run.
            while len(batch) < particles - len(taken):
                if index >= MAX_PROPOSALS:
                    raise RuntimeError(
                        f"calibration of model {model.name} made {MAX_PROPOSALS} proposals "
                        "without filling a population; its priors leave almost no room"
                    )
                candidate = propose(population, index, previous)
                index += 1
                if candidate is not None:
                    batch.append(candidate)
            for (theta, _), measured in zip(batch, measure(batch, pool), strict=True):
                if len(taken) == particles:
                    break
                if math.isfinite(measured) and measured <= tolerance:
                    taken.append((theta, measured))
        return taken

    with Pool(workers) as pool:
        # First population: straight from the prior, keeping every simulation that ran.
        taken = fill(0, None, math.inf, pool)
        thetas = np.array([theta for theta, _ in taken])
        distances = np.array([measured for _, measured in taken])
        weights = np.full(particles, 1.0 / particles)
        tolerances = [math.inf]

        for population in range(1, populations):
            tolerance = float(np.quantile(distances, keep))
            mean = weights @ thetas
            variance = weights @ (thetas - mean) ** 2
            # Perturbation with twice the weighted variance of the current population.
            spread = np.sqrt(2.0 * np.maximum(variance, 1e-300))
            taken = fill(population, _Population(thetas, weights, spread), tolerance, pool)

            new_weights = np.array(
                [
                    prior_density(theta)
                    / float(weights @ np.exp(-0.5 * (((theta - thetas) / spread) ** 2).sum(axis=1)))
                    for theta, _ in taken
                ]
            )
            thetas = np.array([theta for theta, _ in taken])
            distances = np.array([measured for _, measured in taken])
            weights = new_weights / new_weights.sum()
            tolerances.append(tolerance)

    for j, prior in enumerate(priors):
        thetas[:, j] = [_from_internal(prior, float(u)) for u in thetas[:, j]]
    return CalibrationResult(names, thetas, weights, tolerances, simulations)
