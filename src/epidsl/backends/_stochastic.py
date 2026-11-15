"""Pieces shared by the jump-process backends (Gillespie, tau-leap)."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from epidsl.backends import CompiledModel
from epidsl.expr import walk
from epidsl.ir import EpiModel, Sym


def require_whole_initial_values(model: EpiModel, label: str) -> None:
    """Jump processes count individuals; refuse a fractional initial value."""
    for compartment in model.compartments:
        if compartment.initial != math.floor(compartment.initial):
            raise ValueError(
                f"model {model.name}: initial value of '{compartment.name}' "
                f"({compartment.initial:g}) must be a whole number for {label} semantics"
            )


def require_time_independent_rates(model: EpiModel, label: str) -> None:
    """Both jump backends freeze rates between events or over a step; refuse ``t``."""
    for flow in model.flows:
        if any(isinstance(node, Sym) and node.name == "t" for node in walk(flow.expr)):
            raise ValueError(
                f"model {model.name}: time-dependent rates are not supported by the {label} backend"
            )


def evaluate_propensities(
    compiled: CompiledModel,
    state: NDArray[np.float64],
    t: float,
    out: NDArray[np.float64],
    model_name: str,
    label: str,
) -> None:
    """Fill ``out`` with the rate of every flow in ``state`` at time ``t``.

    A flow whose source is empty has rate zero, whatever its expression. Raises
    RuntimeError on a rate that cannot be evaluated or is negative or undefined.
    """
    for i, flow in enumerate(compiled.flows):
        if flow.source is not None and state[flow.source] <= 0:
            out[i] = 0.0
            continue
        try:
            value = flow.rate(state, t)
        except (ArithmeticError, ValueError) as error:
            raise RuntimeError(
                f"{label} simulation failed for model {model_name}: "
                f"a flow rate could not be evaluated ({error})"
            ) from error
        if not value >= 0:
            raise RuntimeError(
                f"{label} simulation failed for model {model_name}: "
                f"a flow rate is negative or undefined ({value:g}) at t = {t:g}"
            )
        out[i] = value


def choose_flow(propensities: NDArray[np.float64], total: float, rng: np.random.Generator) -> int:
    """Pick a flow index with probability proportional to its propensity."""
    threshold = rng.random() * total
    chosen = int(np.searchsorted(np.cumsum(propensities), threshold, side="right"))
    chosen = min(chosen, len(propensities) - 1)
    while propensities[chosen] == 0.0:  # guard against rounding at the upper end
        chosen -= 1
    return chosen
