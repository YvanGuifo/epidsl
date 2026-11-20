"""Network (contact-based) backend for EpiDSL.

Each individual is a node in a contact graph.  Flows whose rate contains
``contacts(X)`` are evaluated per-agent: ``contacts(X)`` resolves to the number
of that agent's neighbours currently in state X.  All other flows are evaluated
exactly as in the Gillespie backend (population-level constants, t, N).

The simulation is an event-driven Gillespie algorithm on the graph:
  1. Compute rate_u for every agent u based on its current state and neighbourhood.
  2. Total rate R = Σ rate_u.
  3. Draw waiting time dt ~ Exp(R) and advance time.
  4. Pick the event proportionally to rate_u; update agent state.
  5. Recompute rates for u and all neighbours of u (O(degree) update).

Supported generators: erdos_renyi(p), watts_strogatz(k, rewire),
                       barabasi_albert(m), complete().
"""

from __future__ import annotations

from collections.abc import Mapping

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from epidsl.backends import BackendResult
from epidsl.checks import errors
from epidsl.expr import FUNCTIONS, walk
from epidsl.ir import BinOp, Call, EpiModel, Expr, Neg, Num, Semantics, SimulationConfig, Sym, AttributeDecl

_KNOWN_GENERATORS = frozenset({"erdos_renyi", "watts_strogatz", "barabasi_albert", "complete"})


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _build_graph(model: EpiModel, n: int, rng: np.random.Generator) -> nx.Graph:
    decl = model.network
    if decl is None:
        raise ValueError("model has no network declaration")
    gen = decl.generator
    params = dict(decl.params)
    seed = int(rng.integers(0, 2**31))
    if gen == "erdos_renyi":
        p = float(params.get("p", 0.01))
        return nx.erdos_renyi_graph(n, p, seed=seed)
    if gen == "watts_strogatz":
        k = int(params.get("k", 6))
        rewire = float(params.get("rewire", 0.1))
        return nx.watts_strogatz_graph(n, k, rewire, seed=seed)
    if gen == "barabasi_albert":
        m = int(params.get("m", 3))
        return nx.barabasi_albert_graph(n, m, seed=seed)
    if gen == "complete":
        return nx.complete_graph(n)
    raise ValueError(f"unknown network generator '{gen}'")


# ---------------------------------------------------------------------------
# Expression evaluator with contacts() support
# ---------------------------------------------------------------------------


def _assign_attributes(
    attributes: tuple,
    n: int,
    rng: np.random.Generator,
) -> dict[str, NDArray[np.int32]]:
    """Draw categorical attribute values for N agents.

    Returns a dict mapping attribute name → int array of category indices (one per agent).
    """
    result: dict[str, NDArray[np.int32]] = {}
    for decl in attributes:
        labels = [label for label, _ in decl.categories]
        probs = np.array([p for _, p in decl.categories], dtype=np.float64)
        probs /= probs.sum()  # normalise
        indices = rng.choice(len(labels), size=n, p=probs).astype(np.int32)
        result[decl.name] = indices
    return result


def _eval_expr(
    expr: Expr,
    constants: Mapping[str, float],
    compartment_index: Mapping[str, int],
    state_vec: NDArray[np.float64],
    t: float,
    contact_counts: Mapping[str, int],
    agent_attr_values: Mapping[str, str] | None = None,
) -> float:
    """Recursively evaluate expr; contact_counts maps compartment name → neighbour count.

    agent_attr_values maps attribute name → current agent's label (e.g. {"risk": "High"}).
    """
    def _rec(e: Expr) -> float:
        return _eval_expr(e, constants, compartment_index, state_vec, t, contact_counts, agent_attr_values)

    match expr:
        case Num(value):
            return value
        case Sym("t"):
            return t
        case Sym("N"):
            return float(state_vec.sum())
        case Sym(name) if name in compartment_index:
            return float(state_vec[compartment_index[name]])
        case Sym(name):
            return constants[name]
        case Neg(operand):
            return -_rec(operand)
        case BinOp("+", left, right):
            return _rec(left) + _rec(right)
        case BinOp("-", left, right):
            return _rec(left) - _rec(right)
        case BinOp("*", left, right):
            return _rec(left) * _rec(right)
        case BinOp("/", left, right):
            return _rec(left) / _rec(right)
        case BinOp("^", left, right):
            return _rec(left) ** _rec(right)
        case Call("contacts", (Sym(comp_name),)):
            return float(contact_counts.get(comp_name, 0))
        case Call("attr", (Sym(attr_name), Sym(value_name))):
            if agent_attr_values is None:
                return 0.0
            return 1.0 if agent_attr_values.get(attr_name) == value_name else 0.0
        case Call(func, args):
            _, impl = FUNCTIONS[func]
            return float(impl(*(_rec(a) for a in args)))
    raise TypeError(f"unsupported expression node: {expr!r}")


def _has_contacts(expr: Expr) -> bool:
    """Return True if expr contains a contacts() call."""
    return any(isinstance(n, Call) and n.func == "contacts" for n in walk(expr))


# ---------------------------------------------------------------------------
# Per-agent rate computation
# ---------------------------------------------------------------------------


def _agent_rate(
    agent: int,
    agent_state: int,
    flow_source: int | None,
    flow_target: int | None,
    flow_expr: Expr,
    constants: Mapping[str, float],
    compartment_index: Mapping[str, int],
    compartment_names: list[str],
    agent_states: NDArray[np.int32],
    neighbours: list[list[int]],
    state_vec: NDArray[np.float64],
    t: float,
    agent_attr_values: Mapping[str, str] | None = None,
) -> float:
    """Return the event rate for agent ``agent`` firing flow source→target.

    Zero if the agent is not in ``flow_source``.
    """
    if flow_source is not None and agent_state != flow_source:
        return 0.0
    contact_counts = {
        name: int(np.sum(agent_states[neighbours[agent]] == i))
        for i, name in enumerate(compartment_names)
    }
    return max(
        0.0,
        _eval_expr(flow_expr, constants, compartment_index, state_vec, t, contact_counts, agent_attr_values),
    )


# ---------------------------------------------------------------------------
# Main backend
# ---------------------------------------------------------------------------


class NetworkBackend:
    """Gillespie simulation on a contact network."""

    def run(self, model: EpiModel, config: SimulationConfig) -> BackendResult:
        problems = errors(model)
        if problems:
            details = "; ".join(str(p) for p in problems)
            raise ValueError(f"model {model.name} has errors: {details}")
        if Semantics.NETWORK not in model.semantics:
            declared = ", ".join(s.value for s in model.semantics)
            raise ValueError(
                f"model {model.name} does not declare Network semantics (declared: {declared})"
            )

        rng = np.random.default_rng(config.seed)
        compartment_names = [c.name for c in model.compartments]
        compartment_index = {name: i for i, name in enumerate(compartment_names)}
        constants = {k.name: k.value for k in model.constants}
        n_compartments = len(compartment_names)

        # Total population
        N = int(sum(c.initial for c in model.compartments))

        # Build contact graph
        G = _build_graph(model, N, rng)
        neighbours: list[list[int]] = [list(G.neighbors(u)) for u in range(N)]

        # Assign initial states: draw agents for each compartment
        agent_states = np.empty(N, dtype=np.int32)
        assigned = 0
        for i, c in enumerate(model.compartments):
            count = int(c.initial)
            agent_states[assigned:assigned + count] = i
            assigned += count
        rng.shuffle(agent_states)

        # Assign discrete attributes to agents
        attr_index_arrays = _assign_attributes(model.attributes, N, rng)
        # Build label lookup: attr_name → {index → label}
        attr_label_maps: dict[str, list[str]] = {
            decl.name: [label for label, _ in decl.categories]
            for decl in model.attributes
        }

        # Flow metadata
        flow_sources = [
            None if f.source is None else compartment_index[f.source]
            for f in model.flows
        ]
        flow_targets = [
            None if f.target is None else compartment_index[f.target]
            for f in model.flows
        ]
        flow_exprs = [f.expr for f in model.flows]

        # Output grid
        output_times = np.linspace(0.0, config.t_end, config.n_points)
        trajectories: dict[str, list[float]] = {name: [] for name in compartment_names}

        def _record(t_now: float) -> None:
            counts = np.bincount(agent_states, minlength=n_compartments)
            for i, name in enumerate(compartment_names):
                trajectories[name].append(float(counts[i]))

        # State vector (compartment counts, for non-contacts expressions)
        def _state_vec() -> NDArray[np.float64]:
            return np.bincount(agent_states, minlength=n_compartments).astype(np.float64)

        t = 0.0
        output_idx = 0

        # Record initial state
        _record(0.0)
        output_idx = 1

        max_steps = N * config.n_points * 1000  # safety cap
        step = 0

        while t < config.t_end and step < max_steps:
            sv = _state_vec()

            # Compute all rates: (agent, flow_index) -> rate
            # For efficiency, only compute rates for agents in the right state
            total_rate = 0.0
            agent_flow_rates: list[tuple[int, int, float]] = []

            for f_idx, (f_src, f_tgt, f_expr) in enumerate(
                zip(flow_sources, flow_targets, flow_exprs)
            ):
                if f_src is not None:
                    # Only agents in state f_src can fire this flow
                    eligible = np.where(agent_states == f_src)[0]
                else:
                    eligible = np.arange(N)

                for u in eligible:
                    agent_attrs = {
                        name: attr_label_maps[name][int(attr_index_arrays[name][u])]
                        for name in attr_label_maps
                    } if attr_label_maps else None
                    rate = _agent_rate(
                        u,
                        int(agent_states[u]),
                        f_src,
                        f_tgt,
                        f_expr,
                        constants,
                        compartment_index,
                        compartment_names,
                        agent_states,
                        neighbours,
                        sv,
                        t,
                        agent_attrs,
                    )
                    if rate > 0.0:
                        agent_flow_rates.append((u, f_idx, rate))
                        total_rate += rate

            if total_rate <= 0.0:
                # No events possible: epidemic ended
                break

            # Draw next event time
            dt = rng.exponential(1.0 / total_rate)
            t_next = t + dt

            # Record at all output times before t_next
            while output_idx < len(output_times) and output_times[output_idx] <= t_next:
                _record(output_times[output_idx])
                output_idx += 1

            # Select event
            rates_arr = np.array([r for _, _, r in agent_flow_rates])
            chosen = int(rng.choice(len(agent_flow_rates), p=rates_arr / rates_arr.sum()))
            u, f_idx, _ = agent_flow_rates[chosen]

            # Apply event
            f_tgt = flow_targets[f_idx]
            f_src = flow_sources[f_idx]
            if f_src is not None:
                agent_states[u] = f_tgt if f_tgt is not None else f_src
            elif f_tgt is not None:
                agent_states[u] = f_tgt

            t = t_next
            step += 1

        # Fill remaining output times
        while output_idx < len(output_times):
            _record(output_times[output_idx])
            output_idx += 1

        t_arr = output_times
        y = {name: np.array(trajectories[name]) for name in compartment_names}

        return BackendResult(t=t_arr, y=y)
