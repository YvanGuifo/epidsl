# EpiDSL — Design Decisions

**Version:** 1.15
**Date:** 2026-11-15
**Author:** Dr. A. Yvan GUIFO FODJO (EFREI Paris / EFREI Research Lab)
**Status:** Adopted — implemented: grammar, IR, checks C1–C8, ODE backend, CLI, `compose` with `sync`, Gillespie and tau-leap backends

> The references were checked against their registries on 2026-11-15; three
> identifiers from version 1.0 were wrong (see "References" at the end). Use
> [references.bib](references.bib), not the BibTeX of version 1.0.

---

## Overview

EpiDSL is a standalone domain-specific language for epidemiological modelling that
compiles to a typed Python intermediate representation and targets multiple
simulation backends (ODE, Gillespie SSA, tau-leap, ABM). Its intended scientific
differentiator is a **SAN composition algebra** (Stochastic Automata Networks)
inherited from Kendrick (Bui et al. 2019, BMC Bioinformatics), enabling formally
composable multi-model assemblies.

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
- **Relation to Kendrick, checked on 2026-11-15 against Bui et al. 2019.** The paper
  defines concerns as stochastic automata combined by a tensor sum, following
  stochastic automata networks (Plateau and Stewart). It names two ways automata
  interact: *functional rates* (a rate depends on the state of other automata) and
  *triggered transitions* (a transition in one automaton forces one in others), and
  states that Kendrick supports the first and did not need the second. In EpiDSL,
  the product is the tensor sum, `by` (D14) gives functional rates, and `sync` is a
  synchronised transition, which Kendrick does not provide. Only the passages on
  the mathematical model were read; the correspondence of the rate rule itself
  (source name as cell, other names as marginals) was not checked.

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
  fraction `epsilon` of the compartment is expected to leave.
- **Relation to Cao, Gillespie and Petzold 2006, checked against the paper on
  2026-11-15.** Taken from the paper: the form of the first two bounds,
  `max(eps * x / g, 1) / |mu|` and `max(eps * x / g, 1)^2 / sigma^2`; switching to
  exact steps when the step is below 10 mean inter-event times; halving a step that
  gives a negative population; and `epsilon = 0.03`. Differences:
  - The paper sets `g` from the order of the reactions a species takes part in (1
    for first order, 2 for second order, with corrections). EpiDSL uses `g = 1`
    throughout, because a rate is an arbitrary expression whose order is not known.
    For mass-action infection, a given `epsilon` therefore allows steps up to twice
    as long as in the paper; halve `epsilon` to compare.
  - The paper treats *critical* reactions (within about 10 firings of exhausting a
    reactant) separately, by exact simulation. EpiDSL has no such partition; the
    third bound above, which is not in the paper, plays that part. It was added
    because without it the stationary variance of an immigration–death process was
    overestimated (a test covers this).
  - The paper takes 100 exact steps before trying to leap again; EpiDSL takes one.
    This changes cost, not results.
  - The paper's bounds range over reactant species; EpiDSL's over all compartments.
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
- **Perturbation and weights, checked against both papers on 2026-11-15.** The
  weights are those of Toni et al. 2009, `prior / sum(weights * kernel)`, and a
  proposal outside the prior is discarded without simulating, as there. The kernel
  is that of Beaumont et al. 2009: component-wise Gaussian with twice the weighted
  empirical variance of the current population. Differences:
  - Toni et al. fix the tolerances in advance and apply the first to the draws from
    the prior. EpiDSL accepts every draw from the prior, then sets each tolerance to
    the median distance of the previous population (Beaumont et al. mention choosing
    tolerances from quantiles of earlier iterations).
  - Toni et al. reject a simulation at a distance equal to the tolerance; EpiDSL
    accepts it.
  - For stochastic models Toni et al. describe a variant with several simulations
    per particle; EpiDSL always uses one.
- **Scale:** a `loguniform` parameter is sampled, perturbed and weighted on a
  logarithmic scale. This is not in either paper.

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
- Parallelism: addressed by D21.

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

### D19 — The studio (2026-11-16)

**Decision:** a visual workbench, `epidsl studio FILE`, served on this machine and
shown in the browser. Its centrepiece is the **product grid**: the compartments of a
composed model laid out with the first model in columns and the others in rows, the
flows drawn as arrows, and every arrow linked to the line of source that produced it.

**What sets it apart.** Composition is the contribution of this project, so it is the
thing on screen: one flow written in a model appears as one arrow per stratum, a
constant stratified with `by` shows as different rates on parallel arrows, and a
`sync` rule as a dashed diagonal. The text file remains the model; the studio edits
and saves that file and never stores a model of its own, so a model stays readable,
diffable and writable by a program.

**How it is built.**
- Standard library HTTP server and a page in plain HTML, CSS and JavaScript: no new
  dependency, no build step, nothing fetched from the internet.
- `epidsl.studio.api` holds the logic as plain functions (`analyze`, `run`,
  `calibrate`) and is tested without a browser. A composed model now records its
  components (`EpiModel.parts`) so that the grid can be laid out.
- The server listens on 127.0.0.1 only and refuses a request whose `Host` is not
  itself or whose `Origin` is another site, so a page open elsewhere in the browser
  cannot drive it. It writes one file only: the one it was started with.

**Finding while building it.** The first version drew the calibrated model on a fine
time grid over weekly case counts: the curve peaked near 60 where the data reached
1000, because an incidence is a count per output interval. The fit is now reported
at the observation times, a test compares its peak with the data's, and a manual run
removes the observations from the chart, since its time grid differs.

**Verified:** 43 tests on the API and the server; twelve deliberate faults each
caught. The page itself was checked by hand in a browser — loading, the arrow-to-line
link, live errors, ODE and Gillespie runs, calibration, dark mode, a narrow screen —
and has no automated test.

**Limits:**
- No automated test of the page's JavaScript.
- The editor is a plain text area: no syntax colouring or completion.
- A calibration blocks its request until it ends, with no progress or cancel.
- With three or more models the rows are combinations of the later models; there is
  no way yet to choose which model goes where.
- The studio's wording is in English only.

### D20 — Contact matrices (2026-11-16)

**Decision (chosen by the author among three options):** naming the model twice,
`parameter beta by Age, Age = { young, young: .., young, old: .., ... }`, gives one
constant per ordered pair of states (`beta_young_old`). The flow stays written once
in the component model.

**Semantics.** A rate that uses a matrix is the sum, over the states `b` of that
model, of the rate in which the matrix takes the pair (the state the flow leaves,
`b`) and every group of compartments is restricted to the cells in state `b`. `N`,
time, ordinary constants and the flow's own source are untouched. A group that names
no cell in state `b` contributes zero.

**Alternatives not taken:** an explicit `mix(c, I)` function, and flows rewritten by
hand in the compose block. Both would have forced the component's flow to be
rewritten, which is what composition exists to avoid.

**Restrictions, each with its own error:** a matrix is indexed by one model twice
(`by Age, Age`, not two models); every ordered pair needs a value; one rate may use
matrices of one model only; and a rate that uses a matrix must name a group of
compartments, since otherwise the matrix would collapse to a row sum.

**Validation:** a matrix whose rows are constant reproduces the one-way
stratification exactly; with the off-diagonal entries at zero, each group is a closed
SIR of its own and the final-size relation holds within it; moving the same total
contact from within groups to across them changes the outcome in the expected
direction. Nine deliberate faults were each caught.

**Finding: four entries are weakly identified from two case series.** On
`examples/sir_contact_matrix.epidsl`, calibrating the four entries from weekly cases
by age (300 particles, 12 populations, 109 000 simulations, about 7 minutes) left
wide intervals that cover the truth but whose means are far from it. The distance
surface is nearly flat: 0.0130 at the true matrix against 0.0150 at the estimate,
and *lower* a quarter of the way between them. This is the model being
under-determined, not the sampler failing; contact matrices are normally measured in
surveys rather than inferred from case counts.

**Limits:**
- The generated rate is the sum, not the factored form
  (`(b_yy * I_young + b_yo * I_old) * S_young / N`); the two are the same value.
- A matrix cannot cross two different models (OQ3 answered: one model only).
- A matrix cannot be used in an observable declared in a compose block; name the
  generated constants there instead.

### D21 — Running over several processors (2026-11-16)

**Decision:** `workers` on `run_ensemble` and `calibrate`, and `--workers` on the
two commands, spread the simulations over that many processes; 0 means one per
processor. The guarantee is that **the number of workers changes how long a run
takes and nothing else**: a seeded run gives the same particles, weights,
tolerances and count of simulations whatever it is set to.

**How that guarantee is kept.** Each proposal of a calibration draws from a
generator fixed by its place — `SeedSequence(master, spawn_key=(population, index))`
— rather than from one shared generator consumed in the order the work happens to
run. Each proposal carries the seed of its own simulation. A batch asks for exactly
as many proposals as are still wanted, so even the count of simulations does not
follow the worker count. An ensemble derives all its seeds before any run starts.

**Measured on an eight-processor machine:** the calibration of
`examples/sir_contact_matrix.epidsl` (300 particles, 12 populations, 86 000
simulations) takes 406 seconds on one processor and 81 seconds on eight — five
times faster, same answer.

**Finding: a pool per batch made it seven times slower.** The first version opened a
`ProcessPoolExecutor` for each batch of proposals. On macOS a worker is spawned, not
forked, so each one loads NumPy and SciPy afresh — about a second each, paid dozens
of times. The same calibration took 11 seconds on one processor and 72 on four. One
pool now lives for the whole calculation (`epidsl.parallel.Pool`), and the same run
takes 5 seconds on four.

**Compatibility.** The order in which proposals are drawn changed with this
restructuring, so a calibration seeded the same way gives a different (equally
valid) sample from the one version 1.14 gave.

**Limits:**
- A single simulation still runs on one processor; only independent runs are shared.
- Workers are worth it only when there is enough work: starting them costs about a
  second.
- The studio offers the setting but runs the calculation in its own process, so a
  long calibration still blocks that request.

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
6. **Done:** the studio (D19).
7. **Done:** contact matrices (D20).
8. **Done:** running over several processors (D21).
9. Network ABM backend; language server; first PyPI release.

Research follow-ups carried over from the October–November 2026 watch: an LLM
code-generation benchmark task suite for EpiDSL; a Bayesian calibration backend
built on `parameter`; a positioning note against Lingua Franca for the ANR
pre-proposal; optional carbon tracking of simulation runs.

---

## Open Questions

| ID | Question | Decision needed by |
|----|----------|--------------------|
| OQ2 | C9 (mass conservation): error or warning for inflows and outflows? | When C9 is implemented |
| OQ4 | Must source files be UTF-8 (Unicode in unit annotations such as `day⁻¹`)? | Language server |

---

## References

Checked on 2026-11-15 by resolving each identifier at doi.org, Crossref, arXiv,
Europe PMC or jmlr.org. Entries are in [references.bib](references.bib).

### Corrections to version 1.0

| Cited in version 1.0 | Finding | Correct reference |
|---|---|---|
| Kendrick: *Journal of Computational Science* 36, 101001 (2019), DOI 10.1016/j.jocs.2019.03.006 | The DOI does not exist (Crossref 404). No such article was found. | Bui et al. 2019, *BMC Bioinformatics* 20:312, DOI 10.1186/s12859-019-2843-0. The title cited in 1.0 is that of the RIVF 2015 paper, DOI 10.1109/RIVF.2015.7049888. |
| Lingua Franca: EMSOFT 2021, DOI 10.1145/3477022, "Towards a…" | The DOI does not exist. | Lohstroh, Menard, Bateni, Lee 2021, *ACM Transactions on Embedded Computing Systems* 20(4), DOI 10.1145/3448128, "Toward a…". |
| DSL-Bench: arXiv:2502.11168 | That identifier is an unrelated computer-vision paper (spatio-temporal video grounding). | Zhou, Ye, Gao, Wei, Zheng 2026, arXiv:2611.05168. Per its abstract: code generation on 18 DSLs with 7 LLMs. The task types and venue stated in 1.0 are not confirmed. |

### Confirmed

- Gillespie 1977 (DOI 10.1021/j100540a008) and Gillespie 2001 (DOI 10.1063/1.1378322).
- Toni et al. 2009 (DOI 10.1098/rsif.2008.0172); published online in 2008.
- Duong 2026, arXiv:2611.14468. Its subject is Bayesian inference for a heterogeneous
  SIS reaction–diffusion model; "probabilistic programming", stated in 1.0, does not
  appear in the part of the abstract that was read.
- Henderson et al. 2020: title confirmed on jmlr.org; volume and pages not re-checked.
- Page et al. 2021 (DOI 10.1136/bmj.n160) exists but is cited nowhere in this project.
- Added: Cao, Gillespie, Petzold 2006 (DOI 10.1063/1.2159468) and Beaumont et al.
  2009 (DOI 10.1093/biomet/asp052), which D12 and D15 refer to. The code was
  compared with these two papers and with Toni et al. 2009; D12 and D15 list what
  is taken from them and what differs. The versions read were the authors' preprints
  (arXiv:0901.1925, arXiv:0805.2256, and the copy of Cao et al. on a UCSB server).

### Other tools

Licence and description as published by each project (GitHub, PyPI), 2026-11-15.
The feature comparison of version 1.0 (composition, stochastic methods, calibration)
was not re-checked and is not reproduced here.

| Tool | What it is | Licence | Version 1.0 said |
|---|---|---|---|
| Starsim | Agent-based disease modelling framework (Python) | MIT | MIT |
| Lingua Franca | Concurrent programming language with code generation | BSD 2-Clause | BSD-2 |
| Catalyst.jl | Chemical reaction network modelling (Julia). A SciML project, not AlgebraicJulia. | MIT | MIT |
| AlgebraicPetri.jl | Compositional Petri net models (Julia), AlgebraicJulia | MIT | grouped with Catalyst.jl |
| Epydemix | Epidemic modelling, simulation and calibration (Python) | GPL-3.0 | Apache-2 — wrong |
| EMULSION | Epidemiological multi-level simulation framework (Python) | Apache-2.0 | LGPL — wrong |
| EpiAgent | On PyPI, a foundation model for single-cell epigenomic data, not an epidemic simulator | MIT | listed as an agent-based epidemic tool — no such tool was found |

---

*Maintained alongside the source. Update it when a decision is made or an open
question is resolved.*
