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

A parameter with a ``loguniform`` prior is handled on a logarithmic scale throughout:
a perturbation of fixed size on a linear scale would be far too large at the low end
of a range that spans several orders of magnitude, and the weights would degenerate.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from epidsl.backends import Backend, BackendResult
from epidsl.checks import errors
from epidsl.ir import EpiModel, Prior, SimulationConfig

#: Number of output points of each simulation, before interpolation to the data times.
_GRID_POINTS = 401


@dataclass(frozen=True, slots=True)
class Observations:
    """Observed values of some compartments at increasing times."""

    t: NDArray[np.float64]
    y: dict[str, NDArray[np.float64]]


def read_observations(path: Path) -> Observations:
    """Read a CSV file whose first column is ``t`` and whose others are compartments."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
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


def distance(result: BackendResult, observations: Observations) -> float:
    """Root mean squared relative error between a simulation and the observations.

    The simulation is interpolated to the observation times. Each series is divided
    by its largest observed magnitude, so that large and small compartments count alike.
    """
    total = 0.0
    count = 0
    for name, observed in observations.y.items():
        simulated = np.interp(observations.t, result.t, result.y[name])
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
) -> CalibrationResult:
    """Estimate every ``parameter`` of ``model`` that has a prior, by ABC-SMC.

    Each new tolerance is the ``keep`` quantile of the previous population's
    distances. Raises ValueError on unusable inputs and RuntimeError when more than
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
    compartments = {c.name for c in model.compartments}
    for name in observations.y:
        if name not in compartments:
            raise ValueError(f"observed column '{name}' is not a compartment of model {model.name}")
    if particles < 2:
        raise ValueError("calibration needs at least 2 particles")
    if populations < 1:
        raise ValueError("calibration needs at least 1 population")

    names = [k.name for k in targets]
    priors: list[Prior] = [k.prior for k in targets if k.prior is not None]
    rng = np.random.default_rng(seed)
    t_end = float(observations.t[-1])
    simulations = 0

    def simulate(theta: NDArray[np.float64]) -> float:
        nonlocal simulations
        if simulations == max_simulations:
            raise RuntimeError(
                f"calibration of model {model.name} needs more than {max_simulations} simulations"
            )
        simulations += 1
        config = SimulationConfig(
            t_end=t_end, n_points=_GRID_POINTS, seed=int(rng.integers(0, 2**63))
        )
        values = {
            name: _from_internal(prior, float(u))
            for name, prior, u in zip(names, priors, theta, strict=True)
        }
        candidate = with_constants(model, values)
        try:
            return distance(backend.run(candidate, config), observations)
        except RuntimeError:
            return math.inf  # the simulation failed for these values: never accepted

    def prior_density(theta: NDArray[np.float64]) -> float:
        return math.prod(
            _internal_density(prior, float(u)) for prior, u in zip(priors, theta, strict=True)
        )

    # First population: straight from the prior, keeping every simulation that ran.
    thetas = np.empty((particles, len(names)), dtype=np.float64)
    distances = np.empty(particles, dtype=np.float64)
    accepted = 0
    while accepted < particles:
        theta = np.array(
            [_to_internal(prior, prior.sample(rng)) for prior in priors], dtype=np.float64
        )
        d = simulate(theta)
        if math.isfinite(d):
            thetas[accepted], distances[accepted] = theta, d
            accepted += 1
    weights = np.full(particles, 1.0 / particles)
    tolerances = [math.inf]

    for _ in range(1, populations):
        tolerance = float(np.quantile(distances, keep))
        mean = weights @ thetas
        variance = weights @ (thetas - mean) ** 2
        # Perturbation with twice the weighted variance of the current population.
        spread = np.sqrt(2.0 * np.maximum(variance, 1e-300))

        new_thetas = np.empty_like(thetas)
        new_distances = np.empty_like(distances)
        new_weights = np.empty_like(weights)
        accepted = 0
        while accepted < particles:
            origin = thetas[rng.choice(particles, p=weights)]
            theta = origin + spread * rng.standard_normal(len(names))
            density = prior_density(theta)
            if density == 0.0:
                continue
            d = simulate(theta)
            if d > tolerance:
                continue
            kernel = np.exp(-0.5 * (((theta - thetas) / spread) ** 2).sum(axis=1))
            new_thetas[accepted], new_distances[accepted] = theta, d
            new_weights[accepted] = density / float(weights @ kernel)
            accepted += 1
        thetas, distances = new_thetas, new_distances
        weights = new_weights / new_weights.sum()
        tolerances.append(tolerance)

    for j, prior in enumerate(priors):
        thetas[:, j] = [_from_internal(prior, float(u)) for u in thetas[:, j]]
    return CalibrationResult(names, thetas, weights, tolerances, simulations)
