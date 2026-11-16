# EpiDSL

A domain-specific language for epidemiological modelling. A model is written once as
compartments and flows, checked statically, and compiled to a typed intermediate
representation that simulation backends consume.

**Status:** early development. Deterministic (ODE) and stochastic (Gillespie,
tau-leap) simulation, model composition, calibration and a visual workbench work
end to end (see [Roadmap](#roadmap)).

## Install

EpiDSL is not published on PyPI yet. Work from a checkout: it requires Python 3.11
or later and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Supported and tested: Python 3.11 to 3.14, NumPy 2.0 or later, SciPy 1.13 or later.

## Write a model

```
model SIR {
    semantics: [ODE]

    compartment S = 990, I = 10, R = 0

    parameter beta = 0.3 [1/day]   -- transmission rate
    rate gamma = 0.1 [1/day]       -- recovery rate

    flow S -> I : beta * S * I / N
    flow I -> R : gamma * I
}
```

## Check and run it

```bash
uv run epidsl check examples/sir.epidsl
```

```bash
uv run epidsl run examples/sir.epidsl --t-end 160 --out sir.csv
```

`run` writes one CSV row per output time, with a column `t` and one column per
compartment. Without `--out` the CSV goes to standard output.

## Open the studio

The studio shows a model, its simulations and its calibration in the browser. The
text file stays the model: the studio reads it, checks it as you type, and saves it
back.

```bash
uv run epidsl studio examples/sir_age_structured.epidsl
```

It prints a local address and opens it. Stop it with Ctrl+C. Started without a
file, it opens a small template that cannot be saved.

- **Source.** Edit the model on the left. Lines with a problem are marked, and the
  list below explains each one; click an entry to jump to its line.
- **Model.** Compartments are laid out as a grid. For a composed model the first
  model gives the columns and the others the rows, so each cell is one combination
  of states. Arrows are the flows, coloured by the model they come from; a dashed
  arrow is a synchronised transition. Select an arrow to read its rate and to
  highlight the line of source that produced it. One line can produce several
  arrows: that is composition made visible.
- **Simulation.** Choose a method the model declares, then **Run** (Ctrl+Enter).
  ODE runs again by itself as you edit. With a stochastic method and several runs,
  the chart shows the mean and a 95% band. Click a name under the chart to show or
  hide it; observables are dashed and hidden at first.
- **Calibration.** Paste or load the observations, then **Calibrate**. Each
  parameter is shown on a track that spans its prior, with its 95% interval and
  median, and the chart shows the model at the estimated values with the data.

The studio listens on this computer only, loads nothing from the internet, and
answers only its own page. It uses port 8642, or a free one if that is taken;
`--port` chooses one and `--no-browser` does not open the browser.

## Run a stochastic simulation

Two methods simulate one random trajectory of whole individuals. The model must
list the method in its `semantics` (`Gillespie` or `TauLeap`).

| Method | What it does | Use it when |
|--------|--------------|-------------|
| `--method gillespie` | Exact: fires one individual event at a time. | The population is small, or you need the exact law. |
| `--method tau-leap` | Approximate: fires many events per time step. | The population is large and exact simulation is too slow. |

```bash
uv run epidsl run examples/sir.epidsl --t-end 160 --method gillespie --seed 1
```

`--seed` makes the run reproducible; without it each run differs.

To summarise many runs, add `--runs`. The CSV then holds, for each compartment, the
mean over the runs and the quantiles given by `--quantiles` (by default 0.025 and
0.975, a 95% band), in columns such as `I_mean`, `I_q0.025` and `I_q0.975`:

```bash
uv run epidsl run examples/sir.epidsl --t-end 160 --method gillespie --runs 200 --seed 1
```

With `--runs`, `--seed` fixes the whole set of runs. `--runs` is refused with the
ODE method, which always gives the same trajectory.

Both methods require whole-number initial values. A flow never fires from an empty
compartment, and a rate that evaluates to a negative number stops the run with an
error.

A rate may use `t`, for a season or a dated change:

```
flow S -> I : beta * (1 + swing * cos(6.283185307 * t / year)) * S * I / N
```

```bash
uv run epidsl run examples/sir_seasonal.epidsl --t-end 730 --method tau-leap --seed 1
```

Gillespie stays exact with such a rate, by integrating it to find each waiting time,
but that costs far more per event: on the example above, 11 seconds for 30 000
events against 0.15 seconds for the same model without a season. Tau-leap handles it
at its usual speed (0.15 seconds here) and is the practical choice beyond a few
thousand individuals.

Tau-leap chooses its own step sizes, following Cao, Gillespie and Petzold (2006),
and switches to exact steps while few individuals are involved. Its error shrinks with its accuracy parameter `epsilon`
(default 0.03, set with `--epsilon`): on a 100 000
person SIR epidemic the mean trajectory is within about 0.6% of the population
of the ODE solution at 0.03, and 0.2% at 0.01.

## Describe what is observed

Surveillance rarely sees compartments directly. It sees new cases per reporting
period, and only some of them. An `observe` statement defines such a quantity:

```
parameter rho = 0.5 prior uniform(0.05, 1) [1]    -- fraction of infections reported

observe cases = rho * incidence(S -> I)
observe prevalence = I / N
```

`incidence(S -> I)` is the number of people who moved from `S` to `I` since the
previous output time, so with `--points` set for weekly output it is a weekly
count. It is zero at the first output time. `incidence(-> S)` counts arrivals and
`incidence(I ->)` departures. When several flows share the same endpoints, they
are counted together.

Observables are written after the compartments in the output of `epidsl run`,
are summarised by `--runs`, and can be used as columns of the data given to
`epidsl calibrate`. In a composed model, each model's observables are kept: a
compartment name stands for its total over the other models, and an incidence
counts every transition in which that model makes the move.

```bash
uv run epidsl calibrate examples/sir_reporting.epidsl --data examples/sir_weekly_cases.csv --seed 1
```

```
parameter,mean,q0.025,q0.5,q0.975
beta,0.3038061462673391,0.252614404758703,0.3029811560539657,0.3620092080755828
rho,0.37209780074709964,0.16092360120134233,0.35308363011578686,0.6462532658218411
```

The data are synthetic weekly counts produced with `beta = 0.3` and `rho = 0.4`.
The reporting fraction is less sharply determined than the transmission rate;
more `--populations` narrow both intervals.

An observable is a deterministic function of the simulation. Measurement noise,
such as the chance of each case being reported, is not modelled.

## Calibrate parameters from data

Calibration estimates the parameters you are unsure of from observed counts. Give
each such parameter a prior, the range of values you consider plausible:

```
parameter beta = 0.3 prior uniform(0.05, 1) [1/day]
```

Put the observations in a CSV file whose first column is `t` and whose other
columns are named after compartments or observables. Any subset will do:

```
t,I
0,10
10,69
20,200
```

Then run:

```bash
uv run epidsl calibrate examples/sir.epidsl --data examples/sir_observed.csv --seed 1
```

The output has one row per parameter, with its estimated mean and the 2.5%, 50%
and 97.5% quantiles of its plausible values given the data:

```
parameter,mean,q0.025,q0.5,q0.975
beta,0.2866752974896084,0.264269375017657,0.28701743058810203,0.31155539756701467
```

The example data are synthetic: the model was run with `beta = 0.3` and noise was
added, so the estimate can be compared with the value that produced them.

A summary line on standard error reports the number of simulations, the final
tolerance and the effective sample size. `--out` writes the full weighted sample.

| Prior | Meaning |
|-------|---------|
| `uniform(a, b)` | Any value between `a` and `b`, equally likely. |
| `loguniform(a, b)` | Between `a` and `b`, with each order of magnitude equally likely. Both must be positive. |
| `normal(m, s)` | Centred on `m` with standard deviation `s`. |

The method is ABC-SMC (approximate Bayesian computation, sequential Monte Carlo;
Toni et al. 2009, with the adaptive kernel of Beaumont et al. 2009):
it repeatedly simulates the model with candidate values and keeps those whose
simulation is closest to the data, tightening the tolerance at each of
`--populations` rounds. It works with any `--method`. Each simulation starts at
time zero and reports exactly at the times in the data. The distance is the root
mean squared error, each series being divided by its largest observed value.

More populations or particles give a sharper estimate and take longer. A low
effective sample size relative to `--particles` means the estimate rests on few
particles: increase `--particles`.

### Use several processors

A calibration runs thousands of independent simulations, and an ensemble runs many.
`--workers` spreads them over that many processes, and `--workers 0` uses one per
processor:

```bash
uv run epidsl calibrate examples/sir_contact_matrix.epidsl --model ContactMatrix --data examples/contact_weekly_cases.csv --particles 300 --populations 12 --seed 1 --workers 0
```

That calibration takes 406 seconds on one processor and 81 seconds on eight — five
times faster, for the same answer. The number of workers changes how long a run
takes and nothing else: the seeds and the order of the work are fixed before any of
it is handed out, so a seeded run gives the same particles, the same tolerances and
the same count of simulations whatever `--workers` is set to.

Workers only pay off when there is enough work for them: starting them costs about
a second, since each loads NumPy and SciPy afresh.

## Compose models

A `compose` block builds the product of several models. Each model is read as an
automaton whose states are its compartments, in the manner of a stochastic automata
network; the result has one compartment per combination of states.

```
model Age {
    compartment young = 300, old = 700
    rate aging = 0.002 [1/day]
    flow young -> old : aging * young
}

compose AgeSIR {
    models: [SIR, Age]
}
```

`AgeSIR` has the compartments `S_young`, `S_old`, `I_young`, `I_old`, `R_young`
and `R_old`. Run it by name:

```bash
uv run epidsl run examples/sir_age.epidsl --model AgeSIR --t-end 160
```

The rules of composition:

- **Compartments** are named by joining the local names with `_`, in the order of
  the `models` list. Initial values are the independent joint distribution: the
  product of each model's initial values, divided by `N` for each extra model.
- **Flows** of each model are copied into every combination of the other models'
  states.
- **Rates** keep their meaning. The name of a flow's source denotes the specific
  source compartment (`S` becomes `S_young`); every other compartment name denotes
  the total over the other models (`I` becomes `I_young + I_old`). Infection in
  `AgeSIR` therefore mixes homogeneously across age groups.
- **Constants** of all models are merged. **Semantics** is the intersection.

A `sync` rule adds a transition that moves several models at once. Here a severe
infection changes a person's status and location in the same instant:

```
compose Hospital {
    models: [Disease, Place]
    sync S -> I, home -> hospital : severe * beta * S_home * I / N
}
```

In a `sync` rate, the source is named by joining the moves' sources (`S_home`).
With more than two models, a joined name that leaves some models out is a total
over them, and the rule is copied into each of their states.

### Make a rate depend on another model

Inside `compose`, `by` gives a constant of one model a value for each state of
another:

```
compose AgeStructured {
    models: [Epidemic, Age]
    parameter beta by Age = { young: 0.6, old: 0.3 }
}
```

`beta` is replaced by the constants `beta_young` and `beta_old`, and each
transition uses the one of the compartment it leaves: infection of `S_young` runs
at `beta_young * S_young * I / N`. Every state of the `by` model needs a value. The
unit is kept from the original constant unless a new one is given.

This expresses a rate that depends on the state of the individual who moves, such
as susceptibility by age.

### Say who meets whom

Naming the model twice gives one value per ordered pair of states — a contact
matrix. The row is the group being infected, the column the group it meets:

```
compose ContactMatrix {
    models: [Epidemic, Age]

    parameter beta by Age, Age = {
        young, young: 0.9,
        young, old:   0.15,
        old,   young: 0.15,
        old,   old:   0.35
    }
}
```

The flow is still written once, in the component model, and is still
`flow S -> I : beta * S * I / N`. A rate that uses a matrix becomes the sum over the
states of `Age` of that rate, with the matrix on the pair (the state the flow
leaves, the state it meets) and each group of compartments restricted to that
state. The infection of the young is therefore generated as:

```
beta_young_young * S_young * I_young / N + beta_young_old * S_young * I_old / N
```

`N`, time and ordinary constants are untouched. A rate that uses a matrix must name
a group of compartments for the sum to range over, and one rate may use matrices of
one model only. A matrix is indexed by one model twice (`by Age, Age`); two
different models are not supported.

```bash
uv run epidsl run examples/sir_contact_matrix.epidsl --model ContactMatrix --t-end 120
```

Estimating the entries of a matrix from case data is possible but often weakly
determined: on the example above, four entries estimated from two weekly case
series give wide intervals, because very different matrices fit the data almost
equally well. Contact matrices are usually measured, not inferred.

### Say where newcomers enter

A model with an inflow, such as births, can be composed once every other model
states which of its compartments newcomers enter:

```
compose AgeStructured {
    models: [Epidemic, Age]
    enter Age at young
}
```

The inflow `flow -> S : mu * N` of `Epidemic` becomes `flow -> S_young : mu * N`.

### Observe and calibrate by stratum

A `compose` block can declare its own constants and observables. They are written
with the names of the composed model: a compartment such as `S_young`, a generated
constant such as `beta_young`, or a group name such as `I` or `young`, which stands
for a total.

```
compose AgeStructured {
    models: [Epidemic, Age]

    parameter beta by Age = { young: 0.6, old: 0.3 }
    rate rho_young = 0.8 [1]
    rate rho_old = 0.4 [1]

    observe cases_young = rho_young * incidence(S_young -> I_young)
    observe cases_old = rho_old * incidence(S_old -> I_old)
}
```

In `incidence`, a group name counts every transition between the two groups:
`incidence(S -> I)` is all new infections and `incidence(young -> old)` all ageing.

A parameter stratified with `by` keeps the prior of the original, or takes the one
written after its values (`... } prior uniform(0.1, 1)`). Each generated parameter
is then estimated separately:

```bash
uv run epidsl calibrate examples/sir_age_structured.epidsl --model AgeStructured --data examples/age_weekly_cases.csv --seed 1
```

```
parameter,mean,q0.025,q0.5,q0.975
beta_young,0.5977532401017741,0.32833100358229744,0.6015675486652726,0.8736703893495928
beta_old,0.3064468147009518,0.19243156341367465,0.2976122185897274,0.43279312186920177
```

The data are synthetic weekly cases by age group, produced with `beta_young = 0.6`
and `beta_old = 0.3`. With the default five populations the intervals are wide.

The full model is in
[examples/sir_age_structured.epidsl](examples/sir_age_structured.epidsl).

A composition is rejected when: a model is unknown or has errors; two models share
a compartment or constant name; the models' total populations differ or are zero;
they have no semantics in common; a joined or generated name is ambiguous or
already used; or a
model has an inflow (`flow -> A`) and another model has no `enter` statement.
Fewer than two models, and a malformed `sync`, `by` or `enter` statement, are
rejected too.

A composed model is an ordinary model: it can be listed in another `compose`
block, and `models: [A, B, C]` gives the same model as composing `A` and `B`
first.

## Language reference

| Statement | Meaning |
|-----------|---------|
| `model Name { ... }` | Declares a model. A file may hold several. |
| `compartment A = 10, B` | Declares compartments with initial values (default 0). |
| `parameter p = 0.3 [1/day]` | A named constant intended for calibration. The unit is optional. |
| `parameter p = 0.3 prior uniform(0.1, 1) [1/day]` | The same, with the prior that `epidsl calibrate` uses. |
| `rate g = 0.1 [1/day]` | A fixed named constant. The unit is optional. |
| `flow A -> B : expr` | Moves population from `A` to `B` at rate `expr`. |
| `flow -> A : expr` | Inflow (for example births). |
| `flow A -> : expr` | Outflow (for example deaths). |
| `observe x = expr` | A quantity derived from the state, written with the output. May use `incidence(A -> B)`. |
| `compose Name { ... }` | Declares the product of several models. |
| `models: [A, B]` | Inside `compose`: the models to compose, two or more. |
| `sync a -> b, c -> d : expr` | Inside `compose`: a transition that moves two or more models at once, one move per model. |
| `rate c by B = { x: 1, y: 2 }` | Inside `compose`: gives constant `c` one value per compartment of model `B`. Also with `parameter`. |
| `rate c by B, B = { x, x: 1, x, y: 2, ... }` | The same with one value per ordered pair: a contact matrix. |
| `enter B at x` | Inside `compose`: newcomers from another model's inflow enter `B` in compartment `x`. |
| `rate`, `parameter`, `observe` | Inside `compose`: extra constants and observables of the composed model. |
| `semantics: [ODE]` | Backends the model is valid for: `ODE`, `Gillespie`, `TauLeap`, `ABM`. Default `[ODE]`. |

Expressions support `+ - * / ^`, parentheses, the built-in symbols `N` (total
population) and `t` (time), and the functions `exp`, `log`, `sqrt`, `sin`, `cos`,
`abs`, `min`, `max`. Comments are `-- to end of line` and `/* block */`. A
trailing `;` after a statement is optional. The keywords (`model`, `compose`,
`compartment`, `parameter`, `rate`, `flow`, `semantics`, `models`, `sync`, `enter`,
`by`, `at`, `prior`, `observe`, `incidence`) cannot be used as names.

### Static checks

| Code | Severity | Condition |
|------|----------|-----------|
| C1 | error | A name is declared more than once. |
| C2 | error | A flow endpoint is not a compartment. |
| C3 | error | An expression uses an undefined symbol, an unknown function, or the wrong number of arguments. |
| C4 | error | A flow has the same compartment at both ends, or no endpoint at all. |
| C5 | warning | A `rate` or `parameter` has no unit. |
| C6 | error | An initial value is negative, or a declared value is not finite. |
| C7 | warning | A compartment, rate or parameter is never used. |
| C8 | error | A declaration reuses a reserved name (`N`, `t`, or a function name). |
| C11 | error | A prior is ill-defined (bounds in the wrong order, non-positive `loguniform` bound or standard deviation), or the parameter's value lies outside it. |
| C12 | error | `incidence` is used in a flow rate, or no flow has the endpoints it names. |

`--strict` turns warnings into errors.

### Command reference

| Command | Options |
|---------|---------|
| `epidsl check FILE` | `--strict` |
| `epidsl studio [FILE]` | `--port N` (default 8642, or a free port), `--no-browser` |
| `epidsl calibrate FILE` | `--data CSV` (required), `--method ode\|gillespie\|tau-leap` (default `ode`), `--particles N` (default 200), `--populations N` (default 5), `--seed N`, `--workers N`, `--model NAME`, `--out PATH`, `--strict` |
| `epidsl run FILE` | `--t-end T` (required), `--points N` (default 201), `--method ode\|gillespie\|tau-leap` (default `ode`), `--seed N`, `--runs K`, `--workers N`, `--quantiles Q1,Q2,...` (default `0.025,0.975`), `--epsilon E` (tau-leap only, default 0.03), `--model NAME`, `--out PATH`, `--strict` |

All commands exit with status 0 on success and 1 on any error.

## Develop

```bash
uv run pytest
```

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

To run the tests against the oldest dependencies that `pyproject.toml` allows:

```bash
uv run --isolated --resolution lowest-direct pytest
```

To build the package and try it in a clean environment:

```bash
uv build
```

```bash
uv run --isolated --no-project --with dist/*.whl epidsl check examples/sir.epidsl
```

Changes are recorded in [CHANGELOG.md](CHANGELOG.md).

## Roadmap

1. **Done:** grammar, typed IR, checks C1–C8, ODE backend, CLI.
2. **Done:** `compose` blocks with `sync` rules, rates that depend on another model
   (`by`) and entry states for inflows (`enter`).
3. **Done:** Gillespie and tau-leap stochastic backends, ensembles of runs.
4. **Done:** calibration by ABC-SMC, observables and incidence.
5. **Done:** release preparation (package builds and installs cleanly; not published).
6. **Done:** the studio, a visual workbench in the browser.
7. **Done:** contact matrices.
8. **Done:** running simulations over several processors; rates that depend on time
   in the stochastic methods.
9. Network-based agent backend, language server, first PyPI release.

Design rationale, and how each method differs from the paper it follows, are
recorded in [docs/DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md). References are in
[docs/references.bib](docs/references.bib).

## License

MIT
