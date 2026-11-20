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

D42 — Vectorised Gillespie: rate computation uses scipy.sparse CSR matrix-vector
products instead of a Python loop over agents.  contacts(X), attr(), and
neighbor_attr() are all evaluated simultaneously for all N agents via
  adj @ mask_X  (contacts),
  mask_attr_val  (attr),
  adj @ (mask_X & mask_attr)  (neighbor_attr).
"""

from __future__ import annotations

from collections.abc import Mapping

import networkx as nx
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from epidsl.backends import BackendResult
from epidsl.checks import errors
from epidsl.expr import FUNCTIONS, walk
from epidsl.ir import (
    AttributeDecl,
    BinOp,
    Call,
    EpiModel,
    Expr,
    Neg,
    Num,
    Semantics,
    SimulationConfig,
    Sym,
)

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
# Scalar expression evaluator (used for interact flows, fallback)
# ---------------------------------------------------------------------------


def _eval_expr(
    expr: Expr,
    constants: Mapping[str, float],
    compartment_index: Mapping[str, int],
    state_vec: NDArray[np.float64],
    t: float,
    contact_counts: Mapping[str, int],
    agent_attr_values: Mapping[str, str] | None = None,
    neighbor_attr_counts: Mapping[tuple[str, str, str], int] | None = None,
) -> float:
    """Recursively evaluate expr; contact_counts maps compartment name → neighbour count."""

    def _rec(e: Expr) -> float:
        return _eval_expr(
            e,
            constants,
            compartment_index,
            state_vec,
            t,
            contact_counts,
            agent_attr_values,
            neighbor_attr_counts,
        )

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
            return _rec(left) ** _rec(right)  # type: ignore[no-any-return]
        case Call("contacts", (Sym(comp_name),)):
            return float(contact_counts.get(comp_name, 0))
        case Call("attr", (Sym(attr_name), Sym(value_name))):
            if agent_attr_values is None:
                return 0.0
            return 1.0 if agent_attr_values.get(attr_name) == value_name else 0.0
        case Call("neighbor_attr", (Sym(state_name), Sym(attr_name), Sym(value_name))):
            if neighbor_attr_counts is None:
                return 0.0
            return float(neighbor_attr_counts.get((state_name, attr_name, value_name), 0))
        case Call(func, args):
            _, impl = FUNCTIONS[func]
            return float(impl(*(_rec(a) for a in args)))
    raise TypeError(f"unsupported expression node: {expr!r}")


# ---------------------------------------------------------------------------
# Vectorised expression evaluator (D42)
# ---------------------------------------------------------------------------


def _eval_expr_vec(
    expr: Expr,
    constants: Mapping[str, float],
    compartment_index: Mapping[str, int],
    state_vec: NDArray[np.float64],
    t: float,
    agent_states: NDArray[np.int32],
    adj: sp.csr_array,
    attr_index_arrays: dict[str, NDArray[np.int32]],
    attr_label_maps: dict[str, list[str]],
) -> NDArray[np.float64] | float:
    """Evaluate expr for all N agents simultaneously.

    Returns an N-float array when the expression depends on per-agent state,
    or a Python float when it is a population-level constant.

    contacts(X)              → adj @ (agent_states == idx_X)
    attr(name, val)          → (attr_arrays[name] == idx_val)
    neighbor_attr(X, nm, v)  → adj @ ((agent_states == idx_X) & (attr == idx_v))
    """

    def _rec(e: Expr) -> NDArray[np.float64] | float:
        return _eval_expr_vec(
            e,
            constants,
            compartment_index,
            state_vec,
            t,
            agent_states,
            adj,
            attr_index_arrays,
            attr_label_maps,
        )

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
            lv = _rec(left)
            rv = _rec(right)
            return lv**rv
        case Call("contacts", (Sym(comp_name),)):
            idx = compartment_index[comp_name]
            mask = (agent_states == idx).astype(np.float64)
            result = adj @ mask
            return np.asarray(result).ravel()
        case Call("attr", (Sym(attr_name), Sym(value_name))):
            label_list = attr_label_maps.get(attr_name, [])
            if value_name not in label_list:
                return np.zeros(len(agent_states), dtype=np.float64)
            idx_v = label_list.index(value_name)
            return (attr_index_arrays[attr_name] == idx_v).astype(np.float64)  # type: ignore[no-any-return]
        case Call("neighbor_attr", (Sym(state_name), Sym(attr_name), Sym(value_name))):
            idx_s = compartment_index[state_name]
            label_list = attr_label_maps.get(attr_name, [])
            if value_name not in label_list:
                return np.zeros(len(agent_states), dtype=np.float64)
            idx_v = label_list.index(value_name)
            combined = (agent_states == idx_s) & (attr_index_arrays[attr_name] == idx_v)
            mask = combined.astype(np.float64)
            result = adj @ mask
            return np.asarray(result).ravel()
        case Call(func, args):
            _, impl = FUNCTIONS[func]
            # For built-in math functions: evaluate each arg, apply func element-wise
            evaled = [_rec(a) for a in args]
            return float(impl(*evaled))
    raise TypeError(f"unsupported expression node: {expr!r}")


def _has_contacts(expr: Expr) -> bool:
    """Return True if expr contains a contacts() call."""
    return any(isinstance(n, Call) and n.func == "contacts" for n in walk(expr))


# ---------------------------------------------------------------------------
# Per-agent helpers (used for interact flows only)
# ---------------------------------------------------------------------------


def _compute_neighbor_attr_counts(
    agent: int,
    neighbours: list[list[int]],
    agent_states: NDArray[np.int32],
    compartment_names: list[str],
    attr_index_arrays: dict[str, NDArray[np.int32]],
    attr_label_maps: dict[str, list[str]],
) -> dict[tuple[str, str, str], int]:
    nbr_list = neighbours[agent]
    if not nbr_list:
        return {}
    nbr_arr = np.array(nbr_list, dtype=np.int32)
    nbr_states = agent_states[nbr_arr]
    result: dict[tuple[str, str, str], int] = {}
    for state_idx, state_name in enumerate(compartment_names):
        state_mask = nbr_states == state_idx
        for attr_name, label_list in attr_label_maps.items():
            attr_vals = attr_index_arrays[attr_name][nbr_arr]
            for vi, value_name in enumerate(label_list):
                cnt = int(np.sum(state_mask & (attr_vals == vi)))
                result[(state_name, attr_name, value_name)] = cnt
    return result


# ---------------------------------------------------------------------------
# Attribute assignment
# ---------------------------------------------------------------------------


def _assign_attributes(
    attributes: tuple[AttributeDecl, ...],
    n: int,
    rng: np.random.Generator,
) -> dict[str, NDArray[np.int32]]:
    result: dict[str, NDArray[np.int32]] = {}
    for decl in attributes:
        labels = [label for label, _ in decl.categories]
        probs = np.array([p for _, p in decl.categories], dtype=np.float64)
        probs /= probs.sum()
        indices = rng.choice(len(labels), size=n, p=probs).astype(np.int32)
        result[decl.name] = indices
    return result


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

        N = int(sum(c.initial for c in model.compartments))

        # Build contact graph + sparse adjacency matrix for vectorised ops
        G = _build_graph(model, N, rng)
        neighbours: list[list[int]] = [list(G.neighbors(u)) for u in range(N)]
        adj: sp.csr_array = nx.to_scipy_sparse_array(G, format="csr", dtype=np.float64)

        # Assign initial states
        agent_states = np.empty(N, dtype=np.int32)
        assigned = 0
        for i, c in enumerate(model.compartments):
            count = int(c.initial)
            agent_states[assigned : assigned + count] = i
            assigned += count
        rng.shuffle(agent_states)

        # Assign discrete attributes
        attr_index_arrays = _assign_attributes(model.attributes, N, rng)
        attr_label_maps: dict[str, list[str]] = {
            decl.name: [label for label, _ in decl.categories] for decl in model.attributes
        }

        # Regular flow metadata
        flow_sources = [
            None if f.source is None else compartment_index[f.source] for f in model.flows
        ]
        flow_targets = [
            None if f.target is None else compartment_index[f.target] for f in model.flows
        ]
        flow_exprs = [f.expr for f in model.flows]

        # Interact flow metadata
        iflow_focal_src = [compartment_index[f.focal_source] for f in model.interact_flows]
        iflow_nbr_src = [compartment_index[f.neighbor_source] for f in model.interact_flows]
        iflow_focal_tgt = [compartment_index[f.focal_target] for f in model.interact_flows]
        iflow_nbr_tgt = [compartment_index[f.neighbor_target] for f in model.interact_flows]
        iflow_exprs = [f.expr for f in model.interact_flows]

        # Output grid
        output_times = np.linspace(0.0, config.t_end, config.n_points)
        trajectories: dict[str, list[float]] = {name: [] for name in compartment_names}

        def _record(t_now: float) -> None:
            counts = np.bincount(agent_states, minlength=n_compartments)
            for i, name in enumerate(compartment_names):
                trajectories[name].append(float(counts[i]))

        def _state_vec() -> NDArray[np.float64]:
            return np.bincount(agent_states, minlength=n_compartments).astype(np.float64)

        t = 0.0
        output_idx = 0

        _record(0.0)
        output_idx = 1

        max_steps = N * config.n_points * 1000
        step = 0

        while t < config.t_end and step < max_steps:
            sv = _state_vec()

            # ----------------------------------------------------------------
            # Vectorised rate computation — regular flows
            # ----------------------------------------------------------------
            # For each flow we compute an N-float array of per-agent rates,
            # then pick only the non-zero entries for the event list.

            reg_agents_parts: list[NDArray[np.intp]] = []
            reg_fidx_parts: list[NDArray[np.int32]] = []
            reg_rates_parts: list[NDArray[np.float64]] = []

            for f_idx, (f_src, _f_tgt, f_expr) in enumerate(
                zip(flow_sources, flow_targets, flow_exprs, strict=False)
            ):
                rates_vec = _eval_expr_vec(
                    f_expr,
                    constants,
                    compartment_index,
                    sv,
                    t,
                    agent_states,
                    adj,
                    attr_index_arrays,
                    attr_label_maps,
                )
                if isinstance(rates_vec, float):
                    rates_vec = np.full(N, rates_vec, dtype=np.float64)

                # Apply source compartment mask
                if f_src is not None:
                    rates_vec = rates_vec * (agent_states == f_src)

                rates_vec = np.maximum(0.0, rates_vec)
                nz = np.where(rates_vec > 0.0)[0]
                if len(nz):
                    reg_agents_parts.append(nz)
                    reg_fidx_parts.append(np.full(len(nz), f_idx, dtype=np.int32))
                    reg_rates_parts.append(rates_vec[nz])

            # ----------------------------------------------------------------
            # Interact flows (rate computation vectorised; event list per agent)
            # ----------------------------------------------------------------
            # event_key for interact: ('I', i_idx, b_nbrs list)
            interact_agents: list[int] = []
            interact_keys: list[tuple[str, int, list[int]]] = []
            interact_rates: list[float] = []

            iflow_zip = zip(
                iflow_focal_src,
                iflow_nbr_src,
                iflow_focal_tgt,
                iflow_nbr_tgt,
                iflow_exprs,
                strict=False,
            )
            for i_idx, (fs, ns, _, _, i_expr) in enumerate(iflow_zip):
                # Vectorised: count of ns-state neighbours for every agent
                n_ns_vec = adj @ (agent_states == ns).astype(np.float64)
                if isinstance(n_ns_vec, np.matrix):
                    n_ns_vec = np.asarray(n_ns_vec).ravel()

                per_pair_rate_vec = _eval_expr_vec(
                    i_expr,
                    constants,
                    compartment_index,
                    sv,
                    t,
                    agent_states,
                    adj,
                    attr_index_arrays,
                    attr_label_maps,
                )
                if isinstance(per_pair_rate_vec, float):
                    per_pair_rate_vec = np.full(N, per_pair_rate_vec, dtype=np.float64)

                focal_eligible = (agent_states == fs).astype(np.float64)
                focal_rates_vec = focal_eligible * np.maximum(0.0, per_pair_rate_vec) * n_ns_vec
                focal_rates_vec = np.maximum(0.0, focal_rates_vec)

                # Only loop over agents that actually have ns-neighbours
                nz = np.where(focal_rates_vec > 0.0)[0]
                for u in nz:
                    b_nbrs = [v for v in neighbours[int(u)] if agent_states[v] == ns]
                    if b_nbrs:
                        interact_agents.append(int(u))
                        interact_keys.append(("I", i_idx, b_nbrs))
                        interact_rates.append(float(focal_rates_vec[u]))

            # ----------------------------------------------------------------
            # Merge all events and select one
            # ----------------------------------------------------------------
            total_rate = 0.0

            if reg_agents_parts:
                reg_agents = np.concatenate(reg_agents_parts)
                reg_fidxs = np.concatenate(reg_fidx_parts)
                reg_rates = np.concatenate(reg_rates_parts)
                total_rate += float(reg_rates.sum())
            else:
                reg_agents = np.empty(0, dtype=np.intp)
                reg_fidxs = np.empty(0, dtype=np.int32)
                reg_rates = np.empty(0, dtype=np.float64)

            interact_rates_arr = np.array(interact_rates, dtype=np.float64)
            total_rate += float(interact_rates_arr.sum()) if len(interact_rates_arr) else 0.0

            if total_rate <= 0.0:
                break

            # Draw next event time
            dt = rng.exponential(1.0 / total_rate)
            t_next = t + dt

            # Record at all output times before t_next
            while output_idx < len(output_times) and output_times[output_idx] <= t_next:
                _record(output_times[output_idx])
                output_idx += 1

            # Select event: build a combined cumsum and draw
            n_reg = len(reg_rates)
            n_int = len(interact_rates_arr)
            if n_int:
                all_rates_combined = np.concatenate([reg_rates, interact_rates_arr])
            else:
                all_rates_combined = reg_rates
            cumsum = np.cumsum(all_rates_combined)
            r = rng.uniform(0.0, total_rate)
            chosen = int(np.searchsorted(cumsum, r, side="left"))
            chosen = min(chosen, len(all_rates_combined) - 1)

            if chosen < n_reg:
                # Regular flow
                u = int(reg_agents[chosen])
                f_idx_chosen = int(reg_fidxs[chosen])
                f_tgt_chosen = flow_targets[f_idx_chosen]
                f_src_chosen = flow_sources[f_idx_chosen]
                if f_src_chosen is not None:
                    agent_states[u] = f_tgt_chosen if f_tgt_chosen is not None else f_src_chosen
                elif f_tgt_chosen is not None:
                    agent_states[u] = f_tgt_chosen
            else:
                # Interact flow
                i_chosen = chosen - n_reg
                u = interact_agents[i_chosen]
                _, i_idx, b_nbrs = interact_keys[i_chosen]
                agent_states[u] = iflow_focal_tgt[i_idx]
                v = b_nbrs[int(rng.integers(len(b_nbrs)))]
                agent_states[v] = iflow_nbr_tgt[i_idx]

            t = t_next
            step += 1

        # Fill remaining output times
        while output_idx < len(output_times):
            _record(output_times[output_idx])
            output_idx += 1

        t_arr = output_times
        y = {name: np.array(trajectories[name]) for name in compartment_names}

        return BackendResult(t=t_arr, y=y)
