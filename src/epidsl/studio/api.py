"""What the studio's browser page asks for, as plain functions over JSON-like data.

Every function returns a dictionary with ``ok``; on failure it also carries
``error``, a message for the user. None of them raises on bad input.
"""

from __future__ import annotations

import math
from itertools import product
from typing import Any

import numpy as np

from epidsl.backends import Backend, BackendResult
from epidsl.backends.gillespie import GillespieBackend
from epidsl.backends.ode import ODEBackend
from epidsl.backends.tau_leap import TauLeapBackend
from epidsl.calibration import calibrate as run_calibration
from epidsl.calibration import parse_observations, with_constants
from epidsl.checks import Severity, check
from epidsl.ensemble import run_ensemble
from epidsl.errors import EpiError
from epidsl.expr import format_expr
from epidsl.ir import EpiModel, SimulationConfig
from epidsl.parser import parse

#: Upper bounds on what one request may ask for.
MAX_POINTS = 5000
MAX_RUNS = 2000
MAX_PARTICLES = 5000
MAX_POPULATIONS = 30

#: Shown when the studio is started without a file.
TEMPLATE = """model SIR {
    semantics: [ODE, Gillespie, TauLeap]

    compartment S = 990, I = 10, R = 0

    parameter beta = 0.3 prior uniform(0.05, 1) [1/day]
    rate gamma = 0.1 [1/day]

    flow S -> I : beta * S * I / N
    flow I -> R : gamma * I
}
"""


class _Refused(Exception):
    """A request that cannot be honoured; its text is shown to the user."""


def analyze(source: str) -> dict[str, Any]:
    """Parse ``source`` and describe each model for display."""
    try:
        models = parse(source)
    except EpiError as error:
        problem = {
            "kind": error.kind,
            "message": error.message,
            "line": error.line,
            "column": error.column,
        }
        return {"ok": False, "error": problem, "models": []}
    return {"ok": True, "models": [describe(model) for model in models]}


def describe(model: EpiModel) -> dict[str, Any]:
    """Describe ``model`` as a grid of cells joined by flows.

    The first component's compartments are the columns; the combinations of the other
    components' compartments are the rows. A model that is not composed has one row.
    """
    parts = model.parts or ((model.name, tuple(c.name for c in model.compartments)),)
    columns = list(parts[0][1])
    row_states = list(product(*(states for _, states in parts[1:])))
    position: dict[str, tuple[str, ...]] = {
        compartment.name: states
        for compartment, states in zip(
            model.compartments, product(*(states for _, states in parts)), strict=True
        )
    }

    def flow(source: str | None, target: str | None) -> tuple[str, int | None]:
        if source is None:
            return "inflow", None
        if target is None:
            return "outflow", None
        moved = [
            i
            for i, (a, b) in enumerate(zip(position[source], position[target], strict=True))
            if a != b
        ]
        return ("local", moved[0]) if len(moved) == 1 else ("sync", None)

    diagnostics = check(model)
    flows = []
    for f in model.flows:
        known = all(end is None or end in position for end in (f.source, f.target))
        kind, part = flow(f.source, f.target) if known else ("local", None)
        flows.append(
            {
                "source": f.source,
                "target": f.target,
                "rate": format_expr(f.expr),
                "kind": kind,
                "part": part,
                "line": f.line,
                "column": f.column,
            }
        )
    return {
        "name": model.name,
        "line": model.line,
        "semantics": [s.value for s in model.semantics],
        "grid": {
            "columns": columns,
            "rows": ["_".join(states) for states in row_states],
            "parts": [name for name, _ in parts],
        },
        "cells": [
            {
                "name": c.name,
                "initial": c.initial,
                "col": columns.index(position[c.name][0]),
                "row": row_states.index(position[c.name][1:]),
            }
            for c in model.compartments
        ],
        "flows": flows,
        "constants": [
            {
                "name": k.name,
                "kind": k.kind,
                "value": k.value,
                "unit": k.unit,
                "prior": None
                if k.prior is None
                else {"dist": k.prior.dist, "a": k.prior.a, "b": k.prior.b},
                "line": k.line,
            }
            for k in model.constants
        ],
        "observables": [
            {"name": o.name, "expr": format_expr(o.expr), "line": o.line} for o in model.observables
        ],
        "diagnostics": [
            {
                "code": d.code,
                "severity": d.severity.value,
                "message": d.message,
                "line": d.line,
                "column": d.column,
            }
            for d in diagnostics
        ],
        "usable": not any(d.severity is Severity.ERROR for d in diagnostics),
    }


def _number(payload: dict[str, Any], key: str, default: float | None = None) -> float:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise _Refused(f"{key} must be a number")
    return float(value)


def _count(payload: dict[str, Any], key: str, default: int, low: int, high: int) -> int:
    value = _number(payload, key, default)
    if value != int(value) or not low <= value <= high:
        raise _Refused(f"{key} must be a whole number between {low} and {high}")
    return int(value)


def _seed(payload: dict[str, Any]) -> int | None:
    return None if payload.get("seed") is None else _count(payload, "seed", 0, 0, 2**63 - 1)


def _model(payload: dict[str, Any]) -> EpiModel:
    source = payload.get("source")
    if not isinstance(source, str):
        raise _Refused("source must be text")
    try:
        models = parse(source)
    except EpiError as error:
        raise _Refused(str(error)) from None
    name = payload.get("model")
    for model in models:
        if model.name == name:
            return model
    raise _Refused(f"no model named '{name}'")


def _backend(payload: dict[str, Any]) -> tuple[str, Backend]:
    method = payload.get("method", "ode")
    epsilon = payload.get("epsilon")
    if epsilon is not None and method != "tau-leap":
        raise _Refused("epsilon applies only to the tau-leap method")
    if method == "ode":
        return method, ODEBackend()
    if method == "gillespie":
        return method, GillespieBackend()
    if method == "tau-leap":
        if epsilon is None:
            return method, TauLeapBackend()
        return method, TauLeapBackend(epsilon=_number(payload, "epsilon"))
    raise _Refused(f"unknown method '{method}'")


def _series(result: BackendResult) -> list[dict[str, Any]]:
    return [
        {"name": name, "kind": kind, "mean": values.tolist()}
        for kind, table in (("compartment", result.y), ("observable", result.observed))
        for name, values in table.items()
    ]


def run(payload: dict[str, Any]) -> dict[str, Any]:
    """Simulate a model; with ``runs`` above 1, return the mean and a 95% band."""
    try:
        model = _model(payload)
        method, backend = _backend(payload)
        t_end = _number(payload, "t_end")
        if t_end <= 0:
            raise _Refused("t_end must be positive")
        points = _count(payload, "points", 201, 2, MAX_POINTS)
        runs = _count(payload, "runs", 1, 1, MAX_RUNS)
        config = SimulationConfig(t_end=t_end, n_points=points, seed=_seed(payload))

        if runs == 1:
            result = backend.run(model, config)
            return {"ok": True, "t": result.t.tolist(), "series": _series(result)}
        if method == "ode":
            raise _Refused("several runs need a stochastic method (gillespie or tau-leap)")
        ensemble = run_ensemble(backend, model, config, runs)
        mean, low, high = ensemble.mean(), ensemble.quantile(0.025), ensemble.quantile(0.975)
        compartments = {c.name for c in model.compartments}
        series = [
            {
                "name": name,
                "kind": "compartment" if name in compartments else "observable",
                "mean": mean[name].tolist(),
                "low": low[name].tolist(),
                "high": high[name].tolist(),
            }
            for name in ensemble.names
        ]
        return {"ok": True, "t": ensemble.t.tolist(), "series": series, "runs": runs}
    except (_Refused, ValueError, RuntimeError) as error:
        return {"ok": False, "error": str(error)}


def calibrate(payload: dict[str, Any]) -> dict[str, Any]:
    """Estimate the parameters that have a prior from CSV observations.

    Returns the posterior summary, the data, and a simulation at the posterior mean
    reported at the observation times.
    """
    try:
        model = _model(payload)
        _, backend = _backend(payload)
        data = payload.get("data")
        if not isinstance(data, str):
            raise _Refused("data must be CSV text")
        particles = _count(payload, "particles", 200, 2, MAX_PARTICLES)
        populations = _count(payload, "populations", 5, 1, MAX_POPULATIONS)
        observations = parse_observations(data)
        result = run_calibration(
            backend,
            model,
            observations,
            particles=particles,
            populations=populations,
            seed=_seed(payload),
        )

        mean, low, median, high = (
            result.mean(),
            result.quantile(0.025),
            result.quantile(0.5),
            result.quantile(0.975),
        )
        priors = {k.name: k.prior for k in model.constants if k.prior is not None}
        fitted = with_constants(model, mean)
        # Reported at the observation times, from time zero: an incidence is a count
        # per output interval, so a finer grid would not be comparable with the data.
        times = (
            observations.t if observations.t[0] == 0 else np.concatenate([[0.0], observations.t])
        )
        simulation = backend.run(fitted, SimulationConfig.at(times.tolist(), seed=_seed(payload)))
        return {
            "ok": True,
            "posterior": [
                {
                    "name": name,
                    "mean": mean[name],
                    "low": low[name],
                    "median": median[name],
                    "high": high[name],
                    "prior": {"dist": priors[name].dist, "a": priors[name].a, "b": priors[name].b},
                }
                for name in result.names
            ],
            "simulations": result.simulations,
            "effective_sample_size": result.effective_sample_size,
            "tolerance": result.tolerances[-1] if math.isfinite(result.tolerances[-1]) else None,
            "data": {
                "t": observations.t.tolist(),
                "series": {name: values.tolist() for name, values in observations.y.items()},
            },
            "fit": {"t": simulation.t.tolist(), "series": _series(simulation)},
        }
    except (_Refused, ValueError, RuntimeError) as error:
        return {"ok": False, "error": str(error)}
