# EpiDSL — Design Decisions

**Version:** 1.10
**Date:** 2026-11-15
**Author:** Dr. A. Yvan GUIFO FODJO (EFREI Paris / EFREI Research Lab)
**Status:** Adopted — implemented: grammar, IR, checks C1–C8, ODE backend, CLI, `compose` with `sync`, Gillespie and tau-leap backends

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

**Checked 2026-11-15:** `epidsl` (and `epi-dsl`) return 404 on PyPI, so the name was
free on that date. It is not reserved: only publishing a release reserves it.

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
- Inflows and state-dependent rates: addressed by D14.
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

**Not done:** any performance work — the loop is plain Python.

### D12 — Tau-leap backend (2026-11-15)

**Decision:** explicit Poisson tau-leaping with an adaptive step, falling back to
single exact Gillespie steps when a leap would cover fewer than 10 expected events.

- **Step size.** With accuracy parameter `epsilon` (default 0.03), the step is the
  largest for which, for every compartment: the expected net change and its
  standard deviation stay below `max(epsilon * size, 1)`, and no more than a
  fraction `epsilon` of the compartment is expected to leave. The first two bounds
  follow the idea of Cao, Gillespie and Petzold's step selection; the third was
  added because without it the stationary variance of an immigration–death process
  is overestimated (a test covers this). *The exact correspondence with the
  published procedure has not been checked against the paper.*
- **Negative counts.** A leap that would empty a compartment below zero is
  discarded and retried with half the step.
- **Output.** Steps are cut at output times, so reported states are states of the
  simulated process, not interpolations.
- **Time-dependent rates** are refused, as in the Gillespie backend: a rate is
  frozen over a step, and a first version that allowed `t` overestimated a decaying
  inflow by 5%.

**Validation:** the mean of 30 trajectories converges to the ODE as `epsilon`
shrinks (measured gap 0.6% of the population at 0.03, 0.2% at 0.01, against 0.1%
for Gillespie); binomial law of the pure death process; Poisson stationary variance
of the immigration–death process; small populations are simulated exactly.

**Test-suite check:** seven deliberate faults (three in tau-leap, two in
Gillespie, two in composition) were each caught by at least one test.

### D13 — Ensembles (2026-11-15)

**Decision:** `run_ensemble(backend, model, config, runs)` runs any backend several
times and returns all trajectories; `epidsl run --runs K` writes their mean and
quantiles.

- **Seeds.** Per-run seeds are drawn from NumPy's `SeedSequence(config.seed)`, so
  one master seed reproduces the whole ensemble and each run can be reproduced
  alone from its recorded seed.
- **Output.** The CLI writes a summary (mean and quantiles per compartment), not
  the individual runs. The Python API keeps every run.
- **ODE.** `--runs` is refused with the ODE method.

**Rationale:** calibration will compare summaries of many simulations with data, so
ensembles and reproducible seeding come first.

**Not done:** running the simulations in parallel; writing individual runs from the CLI.

### D14 — Stratified constants and entry states (2026-11-15)

**Decision (chosen by the author among three options each):**

- `rate c by B = { x: v1, y: v2 }` (or `parameter`) inside `compose` gives constant
  `c` one value per compartment of model `B`. It is replaced by the constants `c_x`,
  `c_y`; a transition uses the one of the cell it leaves, and an inflow the one of
  the cell it enters. This is a SAN functional rate: the rate of one automaton
  depends on the state of another.
- `enter B at x` states which compartment of `B` newcomers from another model's
  inflow enter. A model with an inflow is composable only when every other model
  has an entry state; nothing is assumed otherwise.

**Alternatives not taken:** redefining a lifted flow per stratum (general, verbose);
splitting inflows in proportion to the other model's current distribution (wrong
for age).

**Rationale:** named constants per stratum, rather than inlined values, keep units
and let calibration target `beta_young` directly.

**Validation:** equal values reproduce the unstratified model; with susceptibility
by age and no ageing, `S_young(t)/S_young(0) = (S_old(t)/S_old(0))^(beta_young/beta_old)`
holds along the trajectory; with births entering `young`, the status marginals
reproduce the uncomposed model and the young population reaches
`mu * N / (aging + mu)`.

**Limits:**
- A rate depends on the state of the individual who moves, not on who they meet: a
  contact matrix (age by age) is not expressible yet.
- A constant can be stratified by one model only.
- Restricting a transition to some strata (admitting only infected people to
  hospital) still has no syntax; a stratified rate set to zero is the workaround.

### D15 — Calibration (2026-11-15)

**Decision:** `parameter p = v prior dist(a, b)` attaches a prior (`uniform`,
`loguniform`, `normal`); `epidsl calibrate FILE --data CSV` estimates every such
parameter by ABC-SMC. Check C11 rejects an ill-defined prior or a value outside it.

- **Why ABC:** it needs only simulations, so one implementation serves the ODE,
  Gillespie and tau-leap backends.
- **Distance:** root mean squared error after interpolating the simulation to the
  observation times, each series divided by its largest observed magnitude.
- **Tolerances:** adaptive — each is the median distance of the previous population.
- **Perturbation and weights:** Gaussian perturbation with twice the weighted
  variance of the current population, and importance weights
  `prior / sum(weights * kernel)`. *Written from memory of Toni et al. 2009 and
  Beaumont et al. 2009; not checked against the papers.*
- **Scale:** a `loguniform` parameter is sampled, perturbed and weighted on a
  logarithmic scale.

**Findings while building it:**
- With a `loguniform` prior over four orders of magnitude and a linear-scale
  perturbation, the effective sample size fell from 3000 to under 200. The
  logarithmic scale fixed it (now above 80%), and a test guards it.
- An ODE solver's rounding error depends on the parameters. With data that carry no
  information, the adaptive tolerance selected on that noise and the posterior
  drifted. This is a property of ABC with a deterministic simulator and uninformative
  data, not a defect of the weights; it is why the weight tests use a simulator
  whose output is exactly constant.

**Validation:** recovery of one parameter from noisy data and of two parameters at
once (95% interval contains the truth, mean within 5% and 10%); the prior is
restored when the data say nothing (uniform-in-log and normal priors); particles
stay within the prior; a stochastic backend; reproducibility with a seed.

**Test-suite check:** thirteen deliberate faults in the calibration code were each
caught. Two were missed at first (simulation seeds; prior density in the weights)
and tests were added for them.

**Limits:**
- Observation model: addressed by D16.
- Priors on stratified parameters: addressed by D17.
- The posterior is approximate: its width depends on the final tolerance, hence on
  `--populations`.
- Simulations run one after another.

### D16 — Observables and incidence (2026-11-15)

**Decision:** `observe name = expr` declares a quantity computed from the state at
each output time. `incidence(A -> B)` inside it is the number of individuals moved
by the flows from `A` to `B` since the previous output time. Check C12 rejects
`incidence` in a flow rate and an incidence that matches no flow.

- **Backends.** The ODE backend integrates one extra variable per counted flow;
  the jump backends count firings. Only flows that an observable needs are counted.
- **Output times.** `SimulationConfig.at(times)` reports at arbitrary increasing
  times. Calibration uses it to simulate exactly at the observation times, starting
  from zero: an interval count cannot be interpolated.
- **Composition.** Observables are lifted: a state name becomes its marginal total,
  and an incidence counts every transition of the product in which that model makes
  the move, synchronised transitions included.
- **Reporting.** Under-reporting is an ordinary parameter (`rho * incidence(...)`),
  so it can carry a prior and be estimated.

**Validation:** in all three backends the incidence of `S -> I` equals the decrease
of `S` over each interval, for regular and irregular output times; arrivals minus
departures equal the change in population; a lifted incidence in a model with a
`sync` rule equals the decrease of the marginal; transmission rate and reporting
fraction are recovered together from weekly cases.

**Test-suite check:** eleven of twelve deliberate faults were caught; the twelfth
(counting every flow rather than only those needed) changes cost, not results.

**Limits:**
- No measurement noise (binomial reporting, negative-binomial overdispersion): an
  observable is a deterministic function of the simulation.
- Observables by stratum: addressed by D17.
- Reporting delays are not expressible.

### D17 — Constants, observables and priors at the level of a composition (2026-11-15)

**Decision:**
- A `compose` block may declare `observe`, `rate` and `parameter` statements of its
  own. They are written against the composed model: cell names, generated
  constants, and group names (a group name is a total; in `incidence` it stands for
  all the cells of the group).
- A parameter stratified with `by` takes the prior written after its values, or
  inherits the original's. A stratified `rate` never has a prior.

**Rationale:** this closes the loop for the central use case — an age-structured
model calibrated on cases by age. A reporting fraction per stratum is a plain
constant of the block (`rho_young`), not a constant of a component stratified
afterwards: the component has no use for it on its own, and a check (C7) says so.

**Validation:** stratum incidence equals the decrease of that stratum's source (ODE
and Gillespie); group names give totals; `beta_young` and `beta_old` are recovered
together from cases by age. Ten deliberate faults were each caught.

**Limits:**
- An observable of a component still cannot use a constant that the block
  stratifies; declare the observable in the block instead.

### D18 — Release preparation (2026-11-15)

**Done, and verified by running it:**
- The wheel and source distribution build; the wheel contains the grammar, the
  type marker, the licence and the README, and the installed `epidsl` command
  works from a clean environment outside the repository.
- The test suite passes on Python 3.11 and 3.14, and on the oldest dependencies
  the metadata allows (NumPy 2.0.0, SciPy 1.13.0, Lark 1.3.1).
- `LICENSE` (MIT), `CHANGELOG.md`, package metadata, `epidsl --version`.
- CI gained a job on the oldest dependencies and a job that builds and installs
  the package.
- The copyright holder in `LICENSE` and the author name and e-mail address in
  `pyproject.toml`, which become public metadata, were confirmed by the author.

**Finding:** on SciPy 1.13.0, an exception raised by a rate during integration
(overflow of a diverging model, division by zero) aborted the whole Python process
instead of propagating. The ODE backend no longer raises through the solver: it
records the error, returns NaN so the solver stops, and reports the error
afterwards. The declared lower bound on SciPy had never been tested before.

**Not done, and needing the author:**
- Nothing is published, and CI has never run: both need a remote repository.
- No version has been tagged; the changelog section is still "Unreleased".

Planned: C9 (mass conservation) and C10 (requested semantics not declared; today
this is a run-time refusal by the backend). Dimensional consistency is a separate
check from C9 and has no number yet.

---

## Roadmap

1. **Done:** grammar, typed IR, checks C1–C8, ODE backend (SciPy LSODA), CLI.
2. **Done:** `compose`, `sync`, stratified constants and entry states (D10, D14).
   Next for composition: contact matrices (OQ3).
3. **Done:** Gillespie SSA (D11) and tau-leap (D12). Ensembles of runs (D13). Open:
   time-dependent rates in the stochastic backends.
4. **Done:** calibration by ABC-SMC (D15), observables and incidence (D16).
5. **Done:** release preparation (D18).
6. Network ABM backend; language server; first PyPI release.

Research follow-ups carried over from the October–November 2026 watch: an LLM
code-generation benchmark task suite for EpiDSL; a Bayesian calibration backend
built on `parameter`; a positioning note against Lingua Franca for the ANR
pre-proposal; optional carbon tracking of simulation runs.

---

## Open Questions

| ID | Question | Decision needed by |
|----|----------|--------------------|
| OQ2 | C9 (mass conservation): error or warning for inflows and outflows? | When C9 is implemented |
| OQ3 | How should a contact matrix (a rate depending on both parties' strata) be written? | Before realistic age-structured models |
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
