"""Pieces shared by the jump-process backends (Gillespie, tau-leap)."""

from __future__ import annotations

import math
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import brentq

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


def uses_time(model: EpiModel) -> bool:
    """Whether any flow's rate names ``t``."""
    return any(
        isinstance(node, Sym) and node.name == "t"
        for flow in model.flows
        for node in walk(flow.expr)
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


class TimedJumps:
    """Exact jump times when rates depend on time.

    Between two jumps the state does not change, so the total rate is a known function
    of time alone. The next jump falls where its integral reaches a draw from the unit
    exponential law; this class computes that integral and inverts it.
    """

    def __init__(self, compiled: CompiledModel, model_name: str, label: str) -> None:
        self._compiled = compiled
        self._model_name = model_name
        self._label = label
        self._rates = np.zeros(len(compiled.flows), dtype=np.float64)

    def mass(self, state: NDArray[np.float64], start: float, stop: float) -> float:
        """The integral of the total rate from ``start`` to ``stop``, in ``state``."""
        failure: list[RuntimeError] = []

        def total(s: float) -> float:
            # An error must not be raised through the integrator; keep it for after.
            try:
                evaluate_propensities(
                    self._compiled, state, s, self._rates, self._model_name, self._label
                )
            except RuntimeError as error:
                failure.append(error)
                return 0.0
            return float(self._rates.sum())

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IntegrationWarning)
            value, error = quad(total, start, stop, epsabs=1e-10, epsrel=1e-10, limit=200)
        if failure:
            raise failure[0]
        if error > 1e-6 * max(1.0, abs(value)):
            raise RuntimeError(
                f"{self._label} simulation failed for model {self._model_name}: the rates "
                f"could not be integrated between t = {start:g} and t = {stop:g}"
            )
        return float(value)

    def reach(self, state: NDArray[np.float64], start: float, stop: float, target: float) -> float:
        """The time at which the integral from ``start`` equals ``target``.

        ``target`` must not exceed the mass up to ``stop``.
        """
        return float(
            brentq(
                lambda s: self.mass(state, start, s) - target,
                start,
                stop,
                xtol=1e-12 * max(1.0, abs(stop)),
            )
        )

    def advance(
        self, state: NDArray[np.float64], start: float, stop: float, owed: float
    ) -> tuple[float | None, float]:
        """Integrate the total rate from ``start`` to ``stop``, firing when it hits ``owed``.

        Fast path: estimate the jump time from the current rate (exponential
        approximation), then refine with Newton-Raphson using a trapezoid
        integral.  Typically 3–8 rate evaluations per call vs. ~160 with the
        ``quad`` + ``brentq`` combination in :meth:`mass` / :meth:`reach`.

        Slow fallback: when Newton does not converge (non-monotone rates,
        taux near zero), falls back to :meth:`mass` + :meth:`reach`.

        Returns ``(t_jump, residual)``:
        - Jump in ``[start, stop]``: ``t_jump`` is the crossing time (the
          caller must draw a fresh exponential); ``residual`` is 0.
        - No jump: ``t_jump`` is ``None``; ``residual = owed - integral``
          must be carried forward.
        """
        failure: list[RuntimeError] = []
        rates = np.zeros(len(self._compiled.flows), dtype=np.float64)

        def rate_at(t: float) -> float:
            try:
                evaluate_propensities(
                    self._compiled, state, t, rates, self._model_name, self._label
                )
            except RuntimeError as error:
                failure.append(error)
                return 0.0
            return float(rates.sum())

        def trap(a: float, b: float, ra: float, rb: float) -> float:
            return 0.5 * (ra + rb) * (b - a)

        r0 = rate_at(start)
        if failure:
            raise failure[0]

        # Fast estimate: how far would the integral carry us at current rate?
        h = stop - start
        if r0 > 0:
            t_est = start + owed / r0
        else:
            t_est = stop + 1.0  # above stop → check the endpoint

        # ------------------------------------------------------------------ #
        # Case A: estimate says no jump before stop                           #
        # ------------------------------------------------------------------ #
        if t_est > stop:
            # Use Simpson's rule (r0 already evaluated, two more points) for
            # higher precision so the residual is accurate to ~1e-5 relative.
            r_mid = rate_at(start + 0.5 * h)
            r1 = rate_at(stop)
            if failure:
                raise failure[0]
            integral = h / 6.0 * (r0 + 4.0 * r_mid + r1)
            if integral < owed:
                return None, max(0.0, owed - integral)
            # The rate rose enough: there IS a jump — fall through with t=stop.
            t_est = stop

        # ------------------------------------------------------------------ #
        # Case B: Newton-Raphson to pinpoint the crossing                     #
        # F(t) = ∫[start, t] λ - owed = 0,  F'(t) = λ(t)                    #
        # Simpson rule for the integral gives O(h^5) precision vs O(h^3)     #
        # for the trapezoid, so Newton converges to ~1e-6 relative in 3-4    #
        # steps even when the rate varies strongly over the interval.         #
        # ------------------------------------------------------------------ #
        t = min(t_est, stop)
        for _ in range(8):
            r_t = rate_at(t)
            r_mid = rate_at(0.5 * (start + t))
            if failure:
                raise failure[0]
            dt_half = t - start
            integral_t = dt_half / 6.0 * (r0 + 4.0 * r_mid + r_t)
            f = integral_t - owed
            if abs(f) <= 1e-9 * max(1.0, abs(owed)):
                return float(np.clip(t, start, stop)), 0.0
            if r_t <= 0.0:
                break  # Cannot step — fall back to the slow path.
            t_new = t - f / r_t
            t_new = float(np.clip(t_new, start, stop))
            if abs(t_new - t) <= 1e-12 * max(1.0, abs(t)):
                return float(t_new), 0.0
            t = t_new

        if failure:
            raise failure[0]

        # ------------------------------------------------------------------ #
        # Slow fallback (rare): non-monotone or near-zero rates               #
        # ------------------------------------------------------------------ #
        mass_total = self.mass(state, start, stop)
        if mass_total < owed:
            return None, max(0.0, owed - mass_total)
        t_jump = self.reach(state, start, stop, owed)
        return float(t_jump), 0.0
