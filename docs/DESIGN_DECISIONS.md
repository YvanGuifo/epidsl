# EpiDSL — Design Decisions

**Version:** 1.16
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
- Rates that use `t`: see D22.
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
- **Time-dependent rates** were refused at first — freezing the rate over a step
  overestimated a decaying inflow by 5% — and are now handled as described in D22.

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

### D22 — Rates that depend on time, in the stochastic backends (2026-11-16)

**Decision:** a rate may name `t` under every semantics. This replaces the refusal
recorded in D11 and D12.

- **Gillespie stays exact.** Between two jumps the state is fixed, so the total rate
  is a known function of time. The next jump falls where its integral reaches a draw
  from the unit exponential law; the integral is computed by adaptive quadrature and
  inverted by root finding. What is left of the draw is carried across output times.
  An error raised while integrating is kept and reported afterwards, never raised
  through the integrator (see D18).
- **Tau-leap.** A leap is shortened until time alone moves the rates by less than
  `epsilon`, and fires at the average of the rates at its two ends. Its exact steps
  use the same integration as Gillespie. A total rate of zero no longer ends the
  run when a rate may wake up later.
- **Models without `t` take the previous code path, draw for draw**; a test pins
  seeded results from before this change.

**Validation:** the Poisson law of a decaying inflow and the binomial law of a
seasonal death rate, in both backends; a rate that is zero until a date and then
grows; many output times between jumps; a seasonally forced epidemic against the
ODE in both backends; and the case that failed before (D12), a fast-decaying inflow
in tau-leap, now within 0.5% where freezing the rate gave 5%. Eight deliberate
faults were each caught, three of them only after two tests were added.

**Cost, measured on `examples/sir_seasonal.epidsl` (120 days, 30 000 events):**
Gillespie takes 11 seconds, against 0.15 seconds for the same model without a
season — about 75 times more per event, since every jump integrates the rate.
Tau-leap takes 0.15 seconds.

**Limits:**
- The integration assumes rates the quadrature can follow; a rate with a jump in
  time (there is no `if` yet, but `max` gives kinks) is handled, at more cost.
- The exact method could be made several times faster with a cheaper inversion;
  not done.

Planned: C9 (mass conservation) and C10 (requested semantics not declared; today
this is a run-time refusal by the backend). Dimensional consistency is a separate
check from C9 and has no number yet.

---

### D27 — Fast inversion for time-variable rates in Gillespie (2026-11-20)

**Decision:** replace the ``quad`` + ``brentq`` combination in
``TimedJumps.mass`` / ``TimedJumps.reach`` with a new ``TimedJumps.advance``
method that locates the next jump using Newton-Raphson with a Simpson-rule
integral.

- **Problem.** Each jump in a model with time-dependent rates previously
  required ~160 rate evaluations: ``brentq`` made ~15 calls to ``mass()``,
  each of which called SciPy's ``quad`` for ~10 evaluations of the total
  rate.  On ``examples/sir_seasonal.epidsl`` (120 days, 30 000 events) this
  took 11 s.
- **Fast path (Newton-Raphson with Simpson integral).**
  1. Evaluate λ(start) (1 call) and estimate the jump time as
     t₀ = start + owed/λ(start).
  2. If t₀ > stop, verify with Simpson on [start, stop] (2 more calls); if
     the integral is still below ``owed``, return the residual immediately
     (3 calls total for the common "no jump" case).
  3. Otherwise run Newton-Raphson with F(t) = ∫[start, t] λ − owed and
     F′(t) = λ(t), where the integral is evaluated by Simpson with one
     midpoint (2 rate calls per iteration).  Converges in 3–5 iterations
     for the smooth rates typical of epidemiological models (~10–12 calls).
- **Slow fallback.** When Newton stalls (rate near zero, non-monotone
  region), ``advance`` falls back to ``mass`` + ``reach`` (unchanged).
- **Gillespie loop change.** The inner loop of ``GillespieBackend.run``
  was simplified: the separate ``mass < owed`` branch and the ``reach`` call
  were replaced by a single ``jumps.advance(state, t, ahead, owed)`` call.
- **Tau-leap unchanged.** Its ``TimedJumps`` usage was already minimal; no
  change is needed there.

**Cost, measured on ``examples/sir_seasonal.epidsl`` (120 days, 30 000
events, same seed 42):**

| Implementation | Time per run |
|---|---|
| quad + brentq (before) | 11 s |
| Newton + Simpson (after) | 0.78 s |
| **Speedup** | **×14** |

Models without ``t`` use the unchanged path (``while jumps is None`` branch)
and are unaffected.

**Validation:** 8 new tests cover ``TimedJumps.advance`` directly (no-jump
residual, jump time for constant rate, jump time for exponential rate) and
end-to-end (Poisson law, binomial law, seed reproducibility, benchmark);
three deliberate faults (ignore all jumps, force immediate jump, invert
Newton sign) were each caught by at least one test.

---

## Roadmap

1. **Done:** grammar, typed IR, checks C1–C8, ODE backend (SciPy LSODA), CLI.
2. **Done:** `compose`, `sync`, stratified constants and entry states (D10, D14).
   Next for composition: contact matrices (OQ3).
3. **Done:** Gillespie SSA (D11) and tau-leap (D12). Ensembles of runs (D13).
4. **Done:** calibration by ABC-SMC (D15), observables and incidence (D16).
5. **Done:** release preparation (D18).
6. **Done:** the studio (D19).
7. **Done:** contact matrices (D20).
8. **Done:** running over several processors (D21); time-dependent rates in the
   stochastic backends (D22).
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

### D23 — Observation noise: tilde syntax, NLL calibration, stochastic sampling (2026-11-19)

**Decision:** An observable may declare a noise distribution with `~ dist(params)`:
`poisson()`, `normal(sd)`, or `negbinomial(phi)`. The distribution affects
three things:

1. **Calibration distance.** The ABC-SMC distance in `calibration.py` replaces
   the RMSE for noisy observables by the negative log-likelihood (NLL) of the
   noise distribution, normalised per data point. When both noisy and noise-free
   observables are observed, the NLL contribution and the RMSE contribution are
   summed so that neither dominates the other. The sum is still a valid pseudometric
   for ABC (it has a minimum exactly when the simulation equals the data under both
   criteria) and ABC only requires a consistent ordering.

2. **ODE backend.** `observe()` in `backends/__init__.py` always returns the
   deterministic mean regardless of the noise field when no `rng` is passed.
   Deterministic output is the right default for a deterministic solver: it is what
   the user typically wants to plot or compare against data.

3. **Stochastic backends.** When Gillespie or tau-leap pass `rng=rng` to
   `observe()`, each output time gets one independent draw from the noise
   distribution, centred on the deterministic value. This models measurement error
   on top of demographic stochasticity and is useful for generating synthetic
   observed data or for ensemble visualisation.

**Why these three distributions?**
- Poisson: standard for integer counts (cases reported per interval); variance = mean.
- Negative binomial (phi): overdispersed counts; variance = mean + mean²/phi; reduces
  to Poisson as phi → ∞. Commonly used for infectious disease surveillance data.
- Normal(sd): continuous measurements or large-count approximations where Gaussian
  error is a reasonable assumption.

**Why NLL and not RMSE for calibration?**
RMSE treats all deviations equally regardless of the expected magnitude. For count
data the variance grows with the mean, so a deviation of 10 cases when the expected
count is 1 000 is very different from the same deviation when the count is 10. NLL
accounts for this naturally and gives better-calibrated posteriors. Using NLL is
equivalent to Bayesian ABC with a likelihood-based acceptance kernel (Wilkinson 2013,
Stat. Appl. Genet. Mol. Biol. 12, 129–141).

**Parametrisation of negative binomial:**
Variance = μ + μ²/φ. The numpy parametrisation is `negative_binomial(n=φ, p=φ/(φ+μ))`.

**Validation:** mean NLL-based ABC posterior for a Poisson SIR recovers the
true β within 2 standard deviations from 10 particles over 2 populations with
seed 42 (checked in `tests/test_noise.py::test_noise_calibration_poisson`). Tests also
confirm: (i) stochastic Poisson draws have integer values and mean ≈ ODE value;
(ii) ODE returns deterministic values regardless of noise field; (iii) trajectories are
reproducible under the same seed.

---

## D24 — Metapopulation migration via `migrate`

### Context

Epidemiological models often represent several geographically or socially distinct
sub-populations (patches) between which individuals travel.  EpiDSL already supported
stratified models via the SAN composition algebra; the missing piece was a concise way
to express *cross-patch flows* (migration) without listing every directed edge by hand.

### Decision: `migrate` statement with a matrix constant

A new `migrate` statement in `compose` blocks generates inter-patch flows automatically:

```
compose MetaSIR {
    models: [SIR, City]

    rate travel by City, City = {
        Paris, Paris: 0, Paris, Lyon: 0.005, ...
    } [1/day]

    migrate S, I, R : travel
}
```

For every pair of distinct patch states (src, dst) and every listed compartment C, one
flow `C_src -> C_dst` is added with rate `travel_src_dst * C_src`.  This is a
per-capita emigration rate: in the ODE, `dC_src/dt -= travel_src_dst * C_src` and
`dC_dst/dt += travel_src_dst * C_src`.

### Why this design?

**Option A (explicit cross-strata flows)** — writing each flow by hand — is correct but
verbose for N patches: O(N²) lines for a symmetric matrix.

**Option B (declarative `network` block)** — adds three new keywords (`network`, `edge`,
`migrate`).

**Option C (matrix constant + `migrate`)** — reuses the existing matrix-constant
mechanism (`by X, X`) and adds only one keyword (`migrate`).  It composes cleanly with
the existing strata machinery and keeps the grammar minimal.

### Inline matrix constants

As a by-product of this feature, `stratified_rate` and `stratified_parameter` in a
`compose` block may now declare **new** constants (not present in any composed model).
Previously the constant had to be pre-declared in one of the models; this restriction
was lifted because migration matrices have no meaningful "base" value in the component
models.

### Warnings on diagonal rates and patch compartments

The checker emits C7 ("unused") for diagonal rates (`travel_X_X`, always 0) and for
compartments of the pure-patch model (they have no local flows).  These warnings are
cosmetic: the composition uses them correctly.  Users who want a warning-free run can
set all diagonal entries equal to a named zero constant or use `--strict false`.

### Gillespie correctness

Each migration event `C_src -> C_dst` at rate `mu * C_src` is a standard Gillespie
reaction and is handled without changes to the stochastic backends.  Population totals
are conserved exactly (integer counts).  Tested in
`tests/test_network.py::test_migrate_gillespie_total_integer_conservation`.

### Limitations

- Only one matrix per `migrate` statement; multiple matrices require separate statements.
- Patch model must have the same total population as the base model (existing SAN
  constraint); patch compartment sizes encode the fraction of population per patch.
- No distance-based or time-varying travel rates (possible future work).

---

## D25 — Static checks C9 and C10: mass conservation and backend semantics (2026-11-20)

### Context

Two categories of modelling error had no static diagnostic before this tranche:

1. **Open flows without a counterpart.** A model with inflows (`flow -> X`) but no
   outflows, or outflows (`flow X ->`) but no inflows, has an unbounded growth or decay
   that the author may not have intended.  This was silently accepted at parse time and
   only became visible at runtime when the trajectory exploded or collapsed.

2. **Requesting a backend the model does not declare.** When a model's `semantics` list
   does not include the method requested on the CLI (e.g. `epidsl run --method gillespie`
   on an ODE-only model), the runtime raised a `ValueError` deep inside the simulation
   engine — after potentially loading data, running checks, and printing other diagnostics.
   There was no early, structured diagnostic.

### Decision: C9 — mass conservation warning

`check()` emits a **warning** when a model has open inflows but no outflows, or open
outflows but no inflows.  Demographic models that intentionally have both are not warned.

```
warning C9: model has inflows but no outflows; total population grows without bound
warning C9: model has outflows but no inflows; total population decays to zero
```

A **warning** (not an error) was chosen because open flows are a valid modelling
primitive (immigration, mortality without replacement).  The author can suppress the
warning with `--strict false` or by acknowledging it.  Strict mode (`--strict`) turns
it into an error.

### Decision: C10 — backend not declared in semantics (error)

A new function `check_semantics(model, requested: Semantics) -> Diagnostic | None` is
called by `_run()` and `_calibrate()` in `cli.py` immediately after loading and checking
the model, before the simulation starts.  It returns a structured **error** diagnostic
(not a `ValueError`) when the requested backend is not in the model's `semantics` list:

```
error C10: model 'SIR' does not declare gillespie semantics (declared: ode)
```

**Why a separate function, not part of `check()`?**  `epidsl check` does not know which
backend the user will request; that information only exists at `run`/`calibrate` time.
Putting C10 inside `check()` would require passing the backend to a function that is
otherwise backend-agnostic.  The separate `check_semantics` keeps the concerns apart
and is easy to call from any entry point that knows the requested backend.

### Why errors are reported before the simulation starts

Both C9 and C10 follow the principle of "fail early with a clear message".  The
diagnostic carries source position (line and column), so the author can jump directly
to the offending declaration.  A `ValueError` from deep in a backend gives no position
and forces the author to infer the cause from a Python traceback.

---

## D26 — Language Server Protocol server (2026-11-20)

### Context

Authoring EpiDSL models in a text editor without language support means:
- syntax errors are only discovered at the CLI, after a save-and-run loop;
- there is no completion for keywords, compartment names, or rate names;
- finding where a compartment or parameter is declared requires a manual search.

### Decision: `pygls` server with three capabilities

A new `src/epidsl/lsp.py` module implements a Language Server Protocol server
using `pygls` ≥ 2.1.1 and `lsprotocol`.

Three capabilities are exposed:

1. **Diagnostics** (`textDocument/publishDiagnostics`).  On `didOpen` and
   `didChange`, the server calls `parse()` + `check()` and converts each
   `Diagnostic` to its LSP counterpart.  Parse errors (syntax errors from Lark)
   are returned as a single error with `code: "parse"`.  Positions are translated
   from EpiDSL's 1-based (line, column) to LSP's 0-based.

2. **Completion** (`textDocument/completion`).  Returns all DSL keywords, built-in
   symbols (`N`, `t`), math functions, and every compartment and constant declared
   in the file.  Completions are global-scope only (no context-sensitive filtering).
   The file is re-parsed on every completion request; on parse errors, only keywords
   and built-ins are returned.

3. **Goto definition** (`textDocument/definition`).  Extracts the identifier under
   the cursor by walking left and right until a non-word character is found, then
   searches compartments, constants, and observables for a matching declaration.
   Returns the declaration's LSP `Location`, or `null` when nothing matches.

### Design: pure functions + server wiring

The core logic lives in three pure functions — `_check_document`, `_get_completions`,
`_get_definition` — that take plain strings and return `lsprotocol` values.  The
`pygls` handlers are thin wrappers that extract the document text from the workspace
and call these functions.  This separation makes the functions testable without a
running LSP server or JSON-RPC transport.

### Transport

The server runs over stdio (`epidsl server`) so any editor that supports the LSP
protocol can use it without a TCP port or external process manager.

### Limitations

- Completion is not context-sensitive (all items are always returned).
- Definition does not follow composed models (no cross-model lookup).
- Rename and hover are not implemented.

---

## D28 — Sparse migration matrices (2026-11-20)

### Context

A `rate by M, M` matrix used in a `migrate` statement must have all N² entries
explicitly listed, including the N diagonal zeros and all zero off-diagonal
entries. For a 10-patch network this is 100 entries; for a 50-patch ring
topology (100 non-zero edges) it is 2 500 entries, 2 400 of which are zero.
This makes sparse graphs impractical to write and hard to read.

### Decision: omitted pairs default to 0 for migrate matrices

When a matrix constant is the target of a `migrate` statement, entries that
are not listed in the `by M, M = { ... }` block default to 0.  Any pair that
is explicitly listed still requires a valid compartment name; a typo in an
explicit entry continues to raise a composition error.

The sparse default applies **only** to migration matrices (those named in a
`migrate` statement). Other matrix constants — contact-rate matrices such as
`parameter beta by Age, Age` — still require all entries, since a silent 0
would introduce biologically incorrect transmission with no warning.

**Implementation.** `_stratify` in `compose.py` receives the set of rate names
referenced by `migrate` statements. For matrices in that set, `given.setdefault(key,
0.0)` fills in missing pairs before constants are generated. For all other
matrices, the existing "has no value for …" error is preserved.

### Consequences

- Dense syntax (explicit zeros) remains valid; no existing model is broken.
- `examples/metapop_sir.epidsl` updated to omit the six diagonal zeros.
- Contact-rate matrices and other non-migration matrices still require complete
  declarations.

---

## D29 — Reporting delay on observables

### Context

Epidemiological time series are rarely observed instantaneously.  Hospital
admission data, lab-confirmed cases, or death reports all lag the true
infection event by a characteristic delay (1–21 days depending on the
disease and the reporting system).  Without modelling this lag, calibration
fits the wrong phase of the epidemic curve and produces biased parameter
estimates.

### Options

| Option | Mechanics | Drawbacks |
|---|---|---|
| External post-processing | Shift the output array in calling code | Invisible to the DSL; calibration still fits the unshifted series |
| Delay-differential equation | True DDE solver | Requires a new solver, far heavier than a time shift |
| **Time-shift of the output series** | Shift observable array by `d` steps, zero-pad head | No solver change needed; correct for calibration when observations are aligned |

### Decision: `observe NAME = expr delay(d)`

An optional `delay(d)` clause on an `observe` statement post-processes
the observable time series before it is returned to the caller (CLI, calibration
engine, Studio).  The semantics:

- the value at output time `t[k]` in the **delayed** series equals the
  value at `t[k - d]` in the undelayed series;
- time steps before `t = d` are set to 0 (no information before the
  simulation starts);
- `d` is interpreted in the model's time unit (usually days);
- the number of steps shifted is `round(d / dt)` where `dt` is the
  output time step.

**Negative delays** are rejected by check C14 at model-check time.

The clause is order-independent with the noise distribution:
`observe X = expr ~ poisson() delay(7)` and
`observe X = expr delay(7) ~ poisson()` are equivalent (grammar
places `noise` before `delay`; both are optional).

**Implementation.** The shift is applied in `backends/__init__.py::observe()`
after noise sampling, so a delayed stochastic observable shifts the drawn
samples rather than the deterministic mean.  The shift is `O(n)` time and
`O(n)` space.

### Consequences

- No change to any existing model (default `delay = 0.0` is a no-op).
- Calibration with `delay(d)` aligns the simulated curve to data shifted by
  `d` days, without manual pre-processing of the data.
- Check C14 produces a structured diagnostic for `delay(-1)`, consistent
  with the other check codes.

---

## D30 — Contact matrices for stratified transmission

### Context

Age-structured models commonly use a *contact matrix* C where C_{i,j} is the
rate at which a susceptible of group i meets an infectious of group j.  The
force of infection on group i is Σ_j C_{i,j} · I_j · S_i.

### Decision

`parameter beta by Model, Model = { i, j: v, ... }` in a `compose` block
declares a 2-index matrix constant.  When a flow uses such a constant, the
rate is automatically expanded to the contact sum over all states of the second
index model.  All four existing backends receive the expanded plain model and
need no modification.

Two cases are supported:
- **Case A.** `beta` is declared (as a scalar) in the component model and
  overridden by the matrix in `compose`.
- **Case B.** `beta` is declared inline only in `compose`, with no prior
  declaration.  `flatten()` passes `stratified_names` as `allowed_undefined`
  to the static checker so C3 does not fire on the component model before
  stratification is applied.

### Alternatives considered

- **Explicit sum syntax** (`beta * S * sum(I_j by Age)`): more explicit but
  requires new grammar and a new IR node; rejected because the matrix constant
  already carries the same information and the implicit sum is the standard
  epidemiological convention.
- **External matrix file**: out of scope for the DSL; the inline dict is
  sufficient for models with up to ~10 groups.

### Diagonal entries

Contact matrices require all n² entries, including the diagonal (self-group
transmission).  Migration matrices omit the diagonal by convention (no
self-migration).  The `_stratify()` function gates the diagonal-skip on
`name in migrate_rates` to keep these two cases separate.

### Consequences

- No change to any existing model (the `by M, M` syntax is new).
- The static check for undefined symbols (C3) is extended with an
  `allowed_undefined` parameter, used only during composition.
- `examples/sir_age.epidsl` demonstrates the feature.

---

## D35 — Smooth cubic interpolation `smoothstep(t, t0, t1)`

**Context.** `ramp(t, t0, t1)` is C⁰: its derivative jumps at `t0` and `t1`,
creating kinks that LSODA must cross. For interventions that ramp up and down
gradually (a lockdown implemented through daily communications, a school-closure
policy that takes effect day by day), a smoother transition is more physically
realistic and produces cleaner ODE trajectories.

**Decision.** Add `smoothstep(t, t0, t1)` as a built-in function:
- `t <= t0` → 0
- `t >= t1` → 1
- `t0 < t < t1` → `3s² – 2s³`, where `s = (t – t0) / (t1 – t0)`

This is the standard cubic Hermite `smoothstep` from computer graphics (GLSL,
HLSL) and numerical libraries. Its first derivative is 0 at both endpoints, so
it is C¹ everywhere: no kink at `t0` or `t1`.

**ODE breakpoints.** The function is C¹ continuous (smooth derivative); there are
no discontinuities, only inflection points. LSODA degrades gracefully near the
inflection point without injecting extra `t_eval` entries. `_discontinuity_breakpoints()`
is **not** extended.

**Cubic formula note.** `3s²–2s³ = s²(3–2s)` factored for numerical stability.
At `s=0.5`: `3×0.25–2×0.125 = 0.5` (matches the linear ramp at the midpoint by
symmetry — `test_smoothstep_midpoint` confirms this). At `s=0.25`:
`3×0.0625–2×0.015625 = 0.15625` — strictly less than the linear value `0.25`,
confirming the cubic is slower at first (ease-in/ease-out shape).

**Measured value.** Growth rate before ramp (days 5–10) vs after (days 35–40)
on the reference model: before > after, confirmed by `test_smoothstep_ode_epidemic_slows_during_ramp`.

**Stochastic backends.** Same as `ramp`: rates are evaluated at each event time.

**Alternatives considered.**
- *`ramp` (linear)*: C⁰, adequate when the kink is tolerable.
- *Logistic `1/(1+exp(-k*(t-t_mid)))`*: C∞ but never exactly 0 or 1; two extra
  parameters (k and t_mid instead of t0, t1); asymmetric around the midpoint.
- *Quintic smoothstep `6s⁵–15s⁴+10s³`*: C² everywhere, but overkill for
  epidemiological use cases and harder to explain.

---

## D34 — Value-bounding function `clamp(x, lo, hi)`

**Context.** Rate expressions can produce values outside their physical range when
composed with time-dependent drivers. For example, `vacc_rate * t` grows without
bound; using it directly as a coverage fraction can exceed 1 or drop below 0,
causing negative effective transmission rates or spurious oscillations. `min` and
`max` can be nested to achieve the same effect, but `clamp(vacc_rate * t, 0, 1)`
is clearer in intent and matches the vocabulary used in shader programming and
numerical libraries.

**Decision.** Add `clamp(x, lo, hi)` as a built-in function:
- `x < lo` → `lo`
- `x > hi` → `hi`
- `lo <= x <= hi` → `x`

Registered in `FUNCTIONS` with arity 3. Equivalent to `max(lo, min(hi, x))`,
but explicit and readable.

**ODE breakpoints.** `clamp` is continuous everywhere (piecewise linear with two
kinks at `x = lo` and `x = hi`). Unlike `step` or `pulse`, its argument is a
general expression whose kinks in time are not statically determined from the AST.
`_discontinuity_breakpoints()` is **not** extended: LSODA degrades gracefully at
kinks, and injecting breakpoints would require solving `x(t) = lo` and `x(t) = hi`
analytically, which is out of scope for a static pass.

**Stochastic backends.** As with all utility functions, Gillespie and tau-leap
evaluate the rate at each event time. No extra logic needed.

**Alternatives considered.**
- *`min(1, max(0, expr))`*: works but requires the user to know the nesting order
  and is inconsistent across lo ≠ 0, hi ≠ 1 cases.
- *Injecting kink breakpoints from `clamp`*: would require symbolically solving
  `expr(t) = lo` and `expr(t) = hi`, feasible only for linear `expr`.
- *A saturating logistic `1/(1+exp(-k*x))`*: smooth but introduces an extra
  parameter and never reaches exactly 0 or 1.

---

## D33 — Rectangular pulse function `pulse(t, t0, t1)`

**Context.** Some interventions have a clear start and end: a school closure, a
short vaccination campaign, a temporary price shock. `step()` (D31) can model the
start but not the end; composing two steps (`step(t,t0) - step(t,t1)`) is
error-prone and relies on user knowledge of the algebra. A dedicated built-in
reduces cognitive load and matches the vocabulary of epidemiological scenario
planning.

**Decision.** Add `pulse(t, t0, t1)` as a built-in function:
- `t < t0` → 0
- `t0 <= t < t1` → 1  (closed on the left, open on the right)
- `t >= t1` → 0

Registered in `FUNCTIONS` with arity 3. The boundary convention — closed left,
open right — is the standard half-open interval `[t0, t1)` used in mathematics.
It makes consecutive pulses `pulse(t, 0, 10) + pulse(t, 10, 20)` cover `[0, 20)`
without overlap or gap at t=10.

**ODE breakpoint injection.** A pulse has two jump discontinuities, at `t0` and
`t1`. `_discontinuity_breakpoints()` treats `pulse` identically to `ramp`:
`node.func in ("ramp", "pulse")` with `args[1:]` to inject both. The trimming
step keeps the output at exactly `n_points`. This shared branch avoids any change
to the trimming logic.

**Stochastic backends.** As with `step()` and `ramp()`, Gillespie and tau-leap
evaluate the rate at each event time, crossing both boundaries naturally. No extra
logic needed.

**Measured output.** On the reference model (N=1 000, `pulse(t, 20, 50)`, phi=0.9,
t_end=60, n_points=61): R0_before=3.0, R0_during=0.3. With an off-grid pulse
(t0=20.5, t1=49.5, coarse grid n_points=7) the output has exactly 7 points after
trimming, confirmed by `test_pulse_ode_breakpoints_injected`.

**Alternatives considered.**
- *User composing `step(t,t0) - step(t,t1)`*: algebraically correct but
  requires the user to understand the convention and is error-prone for
  `1 - phi * (step(t,t0) - step(t,t1))` where sign errors occur.
- *A family of pulse shapes (trapezoidal, Gaussian)*: out of scope for the
  base language; `ramp()` already handles the trapezoidal entry and exit.
- *`pulse(t, t0, duration)` with duration instead of t1*: less natural for
  calendar-based scenarios where start and end dates are primary.

---

## D32 — Linear ramp function `ramp(t, t0, t1)`

**Context.** Some interventions are not instantaneous. Lockdown stringency may
increase over several days; vaccination coverage grows progressively. `step()`
(D31) cannot model these — it jumps discontinuously. Users need a smooth, linear
transition between two regimes.

**Decision.** Add `ramp(t, t0, t1)` as a built-in function:
- `t <= t0` → 0
- `t >= t1` → 1
- `t0 < t < t1` → `(t - t0) / (t1 - t0)`

Registered in `FUNCTIONS` with arity 3. The boundary convention is `0` at t0
(the ramp is not yet active) and `1` at t1 (fully engaged), consistently with
`step()` being closed on the right.

**ODE breakpoint injection.** The ramp function has kinks (derivative
discontinuities) at both `t0` and `t1`, not a jump. LSODA degrades gracefully at
kinks, but injecting both as breakpoints in `t_eval` avoids unnecessarily small
steps at the transition boundaries. `_discontinuity_breakpoints()` (which
generalises `_step_thresholds()` from D31) iterates over `args[1:]` for a ramp
node, adding both t0 and t1 when they are literals or named constants. The output
trimming step (unchanged from D31) strips both extra points, keeping the output
shape at exactly `n_points`.

**Reuse.** `_resolve_num()` was extracted as a helper shared by step and ramp
handling. `_discontinuity_breakpoints()` replaces the old `_step_thresholds()`
so future time-varying functions need only add one branch there.

**Stochastic backends.** As with `step()`, Gillespie and tau-leap evaluate the
rate at each event time and cross the threshold naturally. No extra logic needed.

**Alternatives considered.**
- *Sigmoidal softening `1/(1+exp(-k*(t-t_mid)))`*: smooth everywhere but
  introduces two extra parameters and never reaches exactly 0 or 1.
- *User composing `step(t,t0) - step(t,t1)`*: gives a pulse, not a ramp.
  Requires the user to understand the algebra, error-prone.
- *Piecewise-linear event scheduling*: would require a richer IR and cannot
  be expressed in the current expression grammar.

---

## D31 — Heaviside step function `step(t, t0)`

**Context.** Many epidemiological interventions are discrete events: a lockdown
starts on day 30, a vaccination campaign begins on week 10. Users need a clean
way to express "rate changes at time t0" without splitting the model into two
separate models or introducing a smoothed sigmoid.

**Decision.** Add `step(t, t0)` as a built-in function returning 0 for `t < t0`
and 1 for `t >= t0`. It is registered in `FUNCTIONS` with arity 2, so the static
checker C3 validates its argument count. The boundary convention is closed on the
right (`t >= t0`), consistent with the standard Heaviside definition.

**ODE breakpoint injection.** LSODA is a continuous integrator; a sharp
discontinuity at `t0` can cause it to take many tiny steps or accumulate error
near the boundary. `_step_thresholds()` extracts every literal or constant `t0`
from `step(t, t0)` calls in flow expressions. Those thresholds are added to
`t_eval` via `np.unique(np.concatenate([user_grid, thresholds]))`, then the
solution is trimmed back to the user-requested grid with `np.searchsorted`.
Thresholds outside `(t_start, t_end)` are skipped. Complex expressions
(e.g. `step(t, a+1)`) are silently ignored and the user relies on LSODA's
adaptive steps.

**Alternatives considered.**
- *Smooth sigmoid `1 / (1 + exp(-k*(t-t0)))`*: avoids the discontinuity but
  introduces an extra parameter `k` and makes threshold semantics fuzzy.
- *Pre-scheduled events (callbacks in `solve_ivp`)*: more precise but requires
  a richer IR and breaks the Gillespie and tau-leap backends.
- *Two-segment ODE (run before and after t0)*: exact but non-compositional; the
  user would need to chain two runs manually.

**Stochastic backends.** Gillespie and tau-leap evaluate `step(t, t0)` at every
event time, so they cross the threshold naturally at the first event after `t0`.
No extra logic is needed.

**Output shape invariant.** The output always has exactly `n_points` values,
regardless of how many breakpoints are injected. The `test_step_ode_threshold_not_on_grid`
test (t0=30.5, coarse grid with dt=10) explicitly checks `len(result.t) == 7`
to guard against trimming regressions.

---

*Maintained alongside the source. Update it when a decision is made or an open
question is resolved.*
