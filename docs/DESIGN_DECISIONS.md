# EpiDSL — Design Decisions

**Version:** 1.3
**Date:** 2026-11-15
**Author:** Dr. A. Yvan GUIFO FODJO (EFREI Paris / EFREI Research Lab)
**Status:** Adopted — implemented: grammar, IR, checks C1–C8, ODE backend, CLI, `compose` with `sync`, Gillespie backend

> References, licences and competitor claims in this document have **not yet been
> independently verified** (see "To verify" at the end). Do not copy them into a
> paper or grant proposal before that is done.

---

## Overview

EpiDSL is a standalone domain-specific language for epidemiological modelling that
compiles to a typed Python intermediate representation and targets multiple
simulation backends (ODE, Gillespie SSA, tau-leap, ABM). Its intended scientific
differentiator is a **SAN composition algebra** (Stochastic Automata Networks)
inherited from Kendrick (Bui et al. 2019), enabling formally composable multi-model
assemblies.

---

## Design Decisions

### D1 — Syntax: braces

**Decision:** Brace-delimited blocks `{ }`. Whitespace and newlines are insignificant.

**Alternatives considered:** Python-style indentation; `begin/end` keywords.

**Rationale:** Indentation-sensitive parsing complicates multi-line rate expressions.
Explicit delimiters make bracket matching, folding and incremental re-parsing simple
for a future language server.

**Implementation note (2026-11-15):** the grammar is parsed with Lark's **LALR**
parser rather than Earley. LALR rejects an ambiguous grammar when it is built, so
ambiguity is caught during development instead of at parse time, and parsing is
linear. Statements are introduced by keywords; a trailing `;` is optional.

### D2 — Composition keyword: `compose`

**Decision:** `compose Name { models: [A, B, ...]  sync ... }`. The model list uses
the same list syntax as `semantics`.

**Alternatives considered:** a `network` keyword (ambiguous with ABM network
topology); operator syntax `SIR ⊗ Demography` (not ASCII-safe).

### D3 — Units: optional, with a warning

**Decision:** Units are optional. Check C5 warns when a `rate` or `parameter` has
none; `--strict` promotes warnings to errors.

**Rationale:** Mandatory units would block quick prototyping; no units would rule out
later dimensional checking. Units are currently stored as text and not interpreted.

### D4 — Package name: `epidsl`

**Decision:** Provisional name `epidsl`, file extension `.epidsl`.

**Action required:** confirm the name is free on PyPI before the first public release.

### D5 — Comments

**Decision:** `--` for line comments; `/* ... */` for block comments.

**Rationale:** `#` is kept free for a possible group/index operator in
age-structured models.

**Known consequence:** `a --b` is read as `a` followed by a comment, not as a
double negation. Write `a - -b`.

### D6 — Semantics declaration

**Decision:** `semantics: [ODE, Gillespie, TauLeap]` list syntax; default `[ODE]`.

**Rationale:** A model documents which backends it is mathematically valid for. A
backend refuses a model that does not declare it.

### D7 — Open populations (2026-11-15)

**Decision:** A flow may omit its source (`flow -> S : expr`, an inflow) or its
target (`flow S -> : expr`, an outflow).

**Rationale:** Births and deaths need no artificial "outside" compartment, and the
missing endpoint is visible in the source, which a future mass-conservation check
can rely on.

### D8 — `parameter` versus `rate` (2026-11-15)

**Decision:** Both declare a named constant. `parameter` marks it as a calibration
target (a prior will attach to it later); `rate` is fixed.

### D9 — Static checks C1–C8 (2026-11-15)

| Code | Severity | Condition |
|------|----------|-----------|
| C1 | error | A name is declared more than once. |
| C2 | error | A flow endpoint is not a compartment. |
| C3 | error | Undefined symbol, unknown function, or wrong number of arguments. |
| C4 | error | A flow with the same compartment at both ends, or with no endpoint. |
| C5 | warning | A `rate` or `parameter` has no unit. |
| C6 | error | Negative initial value, or a non-finite declared value. |
| C7 | warning | A compartment, rate or parameter is never used. |
| C8 | error | A declaration reuses a reserved name (`N`, `t`, a function name). |

### D10 — Composition semantics (2026-11-15)

**Decision:** `compose` is the product of its models read as automata, flattened
into an ordinary model before any backend sees it.

- **States:** one compartment per combination of local states, named by joining
  local names with `_` in model order.
- **Initial distribution:** independent — the product of the marginals divided by
  `N` per extra model. The models must therefore have the same positive total.
- **Local transitions:** each flow is copied into every combination of the other
  models' states (Kronecker sum).
- **Synchronised transitions:** `sync a -> b, c -> d : expr` moves several models at
  once, at most one move per model, and is added to the local transitions.
- **Rates (functional rates):** in a rate, the name of the flow's source denotes the
  specific source compartment; any other state name, or `_`-joined group of state
  names, denotes the marginal total over the models it does not mention.

**Rationale:**
- Flattening keeps the IR and every backend unchanged, and makes composition
  testable on its own: with no `sync` rule the automata are independent, so each
  marginal of the product must reproduce its component and the joint must stay a
  product. The test suite checks exactly that.
- The rate rule is exact when a rate is linear in its source, which is the
  per-capita reading of a transition. A rate that is non-linear in its source (for
  example `k * S * S`) is lifted as written, on the stratum's own count.
- Composition is associative: `[A, B, C]` equals composing `[A, B]` first (OQ3).

**Limits, to revisit:**
- A model with an inflow cannot be composed: the stratum newcomers enter is
  unspecified. This excludes births for now.
- A rate cannot yet depend on another model's state without a `sync` move (for
  example an age-specific `beta`, or admitting only infected people to hospital).
  This is the next extension to design.
- Whether this reading matches Kendrick's composition operator has not been
  checked against the paper.

### D11 — Gillespie backend (2026-11-15)

**Decision:** Gillespie's direct method, one trajectory per run, reported on the
same regular time grid as the ODE backend (the state is constant between jumps).
Each flow is a jump of one individual whose propensity is the flow's expression.

- Initial values must be whole numbers; a fractional value is refused rather than
  rounded, because rounding would change the total population. A composed model
  whose independent initial distribution is fractional is therefore refused.
- A flow has propensity zero while its source is empty, whatever its expression.
- A negative or undefined propensity stops the run with an error.
- Rates that use `t` are refused: the direct method assumes rates are constant
  between jumps. Supporting them needs thinning or a modified next-reaction method.
- An event budget (10 million by default) stops models whose rates grow without limit.

**Validation:** the pure death process against its binomial law, and the
immigration–death process against its Poisson stationary law (mean and variance,
400 seeded runs each).

**Not done:** ensembles (several runs with summary statistics) in the CLI, and any
performance work — the loop is plain Python.

Planned: C9 (mass conservation) and C10 (requested semantics not declared; today
this is a run-time refusal by the backend). Dimensional consistency is a separate
check from C9 and has no number yet.

---

## Roadmap

1. **Done:** grammar, typed IR, checks C1–C8, ODE backend (SciPy LSODA), CLI.
2. **Done:** `compose` and `sync`. Next for composition: state-dependent rates
   and inflows (see D10).
3. **Done:** Gillespie SSA (D11). Next: tau-leap, validated against Gillespie.
4. Network ABM backend; calibration (ABC-SMC); language server; PyPI release.

Research follow-ups carried over from the October–November 2026 watch: an LLM
code-generation benchmark task suite for EpiDSL; a Bayesian calibration backend
built on `parameter`; a positioning note against Lingua Franca for the ANR
pre-proposal; optional carbon tracking of simulation runs.

---

## Open Questions

| ID | Question | Decision needed by |
|----|----------|--------------------|
| OQ1 | Is the PyPI name `epidsl` available? | Before first release |
| OQ2 | C9 (mass conservation): error or warning for inflows and outflows? | When C9 is implemented |
| OQ3 | How should a rate depend on another model's state, and where do inflows enter a product? | Before age-structured examples |
| OQ4 | Must source files be UTF-8 (Unicode in unit annotations such as `day⁻¹`)? | Language server |

---

## To verify

Carried over from version 1.0 of this document and still unchecked:

- **Kendrick:** Bui et al. 2019, *Journal of Computational Science*,
  DOI 10.1016/j.jocs.2019.03.006 — confirm title, authors and that the paper
  supports the SAN composition claim.
- **Lingua Franca:** Lohstroh et al. 2021 — venue (EMSOFT proceedings or ACM TECS)
  and DOI (10.1145/3477022 was cited).
- **DSL-Bench:** cited as Zhou et al., arXiv:2502.11168.
- **Bayesian calibration for SIS:** cited as Duong 2026, arXiv:2611.14468.
- **Competitor table** (Starsim, Lingua Franca, AlgebraicJulia / Catalyst.jl,
  Epydemix, EMULSION, EpiAgent): paradigm, composition, calibration method and
  licence of each. In particular, Catalyst.jl and AlgebraicJulia are separate
  projects, and the existence of an ABM tool named EpiAgent is unconfirmed.
- Gillespie 1977 (DOI 10.1021/j100540a008), Gillespie 2001
  (DOI 10.1063/1.1378322), Toni et al. 2009 (DOI 10.1098/rsif.2008.0172),
  Henderson et al. 2020 (JMLR 21:248).

---

*Maintained alongside the source. Update it when a decision is made or an open
question is resolved.*
