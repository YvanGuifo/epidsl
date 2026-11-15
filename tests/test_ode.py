from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq

from epidsl.backends.ode import ODEBackend
from epidsl.ir import SimulationConfig
from epidsl.parser import parse

EXAMPLES = Path(__file__).parent.parent / "examples"


def run(source: str, t_end: float, n_points: int = 201):
    (model,) = parse(source)
    return ODEBackend().run(model, SimulationConfig(t_end=t_end, n_points=n_points))


def test_exponential_decay_matches_closed_form() -> None:
    result = run("model D { compartment X = 100  rate k = 0.5 [1/day]  flow X -> : k * X }", 10.0)
    np.testing.assert_allclose(result.y["X"], 100 * np.exp(-0.5 * result.t), rtol=1e-6)


def test_si_model_matches_logistic_solution() -> None:
    source = (
        "model SI { compartment S = 99, I = 1  rate b = 0.4 [1/day]  flow S -> I : b * S * I / N }"
    )
    result = run(source, 40.0)
    n, i0, b = 100.0, 1.0, 0.4
    expected = n * i0 * np.exp(b * result.t) / (n - i0 + i0 * np.exp(b * result.t))
    np.testing.assert_allclose(result.y["I"], expected, rtol=1e-6)


def test_sir_conserves_population_and_matches_final_size() -> None:
    result = run((EXAMPLES / "sir.epidsl").read_text(), 2000.0)
    s, i, r = result.y["S"], result.y["I"], result.y["R"]
    n, s0, r0 = 1000.0, 990.0, 3.0

    np.testing.assert_allclose(s + i + r, n, rtol=1e-9)
    assert i[-1] < 1e-6

    # Final-size relation: ln(S_inf / S0) = -R0 * (N - S_inf) / N
    s_inf = brentq(lambda x: np.log(x / s0) + r0 * (n - x) / n, 1e-9, s0 / r0)
    assert s[-1] == pytest.approx(s_inf, rel=1e-6)


def test_open_population_reaches_endemic_equilibrium() -> None:
    result = run((EXAMPLES / "sir_demography.epidsl").read_text(), 5000.0)
    beta, gamma, mu, n = 0.5, 0.1, 0.01, 1000.0

    total = result.y["S"] + result.y["I"] + result.y["R"]
    np.testing.assert_allclose(total, n, rtol=1e-8)
    # S* = N / R0 with R0 = beta / (gamma + mu)
    assert result.y["S"][-1] == pytest.approx(n * (gamma + mu) / beta, rel=1e-5)


def test_time_and_functions_are_available_in_rates() -> None:
    result = run("model F { compartment X = 0  flow -> X : exp(-t) + max(0, -1) }", 5.0)
    np.testing.assert_allclose(result.y["X"], 1 - np.exp(-result.t), atol=1e-7)


def test_result_shape_and_metadata() -> None:
    result = run("model D { compartment X = 1  flow X -> : X }", 1.0, n_points=11)
    assert result.t.shape == (11,)
    assert result.t[0] == 0.0 and result.t[-1] == 1.0
    assert list(result.y) == ["X"]
    assert result.y["X"][0] == 1.0
    assert result.metadata["backend"] == "ODE"


def test_refuses_model_with_errors() -> None:
    (model,) = parse("model B { compartment X  flow X -> : nope }")
    with pytest.raises(ValueError, match="C3"):
        ODEBackend().run(model, SimulationConfig(t_end=1.0))


def test_refuses_model_not_declared_for_ode() -> None:
    (model,) = parse("model G { semantics: [Gillespie] compartment X = 1  flow X -> : X }")
    with pytest.raises(ValueError, match="ODE"):
        ODEBackend().run(model, SimulationConfig(t_end=1.0))


def test_rate_that_cannot_be_evaluated_is_reported() -> None:
    # Division by zero from the first step. The error must surface as a RuntimeError
    # and must not be raised through the solver, which aborts under some SciPy versions.
    (model,) = parse("model Z { compartment X = 1, Y  flow X -> Y : X / Y }")
    with pytest.raises(RuntimeError, match="a flow rate could not be evaluated"):
        ODEBackend().run(model, SimulationConfig(t_end=1.0))


def test_solver_failure_is_reported() -> None:
    # dx/dt = x^2 blows up at t = 1.
    (model,) = parse("model Boom { compartment X = 1  flow -> X : X ^ 2 }")
    with pytest.raises(RuntimeError, match="ODE solver failed"):
        ODEBackend().run(model, SimulationConfig(t_end=5.0))
