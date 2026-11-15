# EpiDSL

A domain-specific language for epidemiological modelling. A model is written once as
compartments and flows, checked statically, and compiled to a typed intermediate
representation that simulation backends consume.

**Status:** early development. Deterministic (ODE) and stochastic (Gillespie,
tau-leap) simulation, model composition and calibration work end to end (see
[Roadmap](#roadmap)).

## Install

Requires Python 3.11 or later and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

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

Both methods require whole-number initial values and rates that do not use `t`. A
flow never fires from an empty compartment, and a rate that evaluates to a negative
number stops the run with an error.

Tau-leap chooses its own step sizes, and switches to exact steps while few
individuals are involved. Its error shrinks with its accuracy parameter `epsilon`
(default 0.03, set with `--epsilon`): on a 100 000
person SIR epidemic the mean trajectory is within about 0.6% of the population
of the ODE solution at 0.03, and 0.2% at 0.01.

## Calibrate parameters from data

Calibration estimates the parameters you are unsure of from observed counts. Give
each such parameter a prior, the range of values you consider plausible:

```
parameter beta = 0.3 prior uniform(0.05, 1) [1/day]
```

Put the observations in a CSV file whose first column is `t` and whose other
columns are named after compartments. Any subset of compartments will do:

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

The method is ABC-SMC (approximate Bayesian computation, sequential Monte Carlo):
it repeatedly simulates the model with candidate values and keeps those whose
simulation is closest to the data, tightening the tolerance at each of
`--populations` rounds. It works with any `--method`. The distance is the root mean
squared error, each series being divided by its largest observed value.

More populations or particles give a sharper estimate and take longer. A low
effective sample size relative to `--particles` means the estimate rests on few
particles: increase `--particles`.

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
as susceptibility by age. It does not express a contact matrix, in which the rate
also depends on the age of the people met.

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
See [examples/sir_age_structured.epidsl](examples/sir_age_structured.epidsl) for
both features together.

A composition is rejected when: a model is unknown or has errors; two models share
a compartment or constant name; the models' total populations differ or are zero;
they have no semantics in common; a joined or generated name is ambiguous; or a
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
| `compose Name { ... }` | Declares the product of several models. |
| `models: [A, B]` | Inside `compose`: the models to compose, two or more. |
| `sync a -> b, c -> d : expr` | Inside `compose`: a transition that moves two or more models at once, one move per model. |
| `rate c by B = { x: 1, y: 2 }` | Inside `compose`: gives constant `c` one value per compartment of model `B`. Also with `parameter`. |
| `enter B at x` | Inside `compose`: newcomers from another model's inflow enter `B` in compartment `x`. |
| `semantics: [ODE]` | Backends the model is valid for: `ODE`, `Gillespie`, `TauLeap`, `ABM`. Default `[ODE]`. |

Expressions support `+ - * / ^`, parentheses, the built-in symbols `N` (total
population) and `t` (time), and the functions `exp`, `log`, `sqrt`, `sin`, `cos`,
`abs`, `min`, `max`. Comments are `-- to end of line` and `/* block */`. A
trailing `;` after a statement is optional. The keywords (`model`, `compose`,
`compartment`, `parameter`, `rate`, `flow`, `semantics`, `models`, `sync`, `enter`,
`by`, `at`, `prior`) cannot be used as names.

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

`--strict` turns warnings into errors.

### Command reference

| Command | Options |
|---------|---------|
| `epidsl check FILE` | `--strict` |
| `epidsl calibrate FILE` | `--data CSV` (required), `--method ode\|gillespie\|tau-leap` (default `ode`), `--particles N` (default 200), `--populations N` (default 5), `--seed N`, `--model NAME`, `--out PATH`, `--strict` |
| `epidsl run FILE` | `--t-end T` (required), `--points N` (default 201), `--method ode\|gillespie\|tau-leap` (default `ode`), `--seed N`, `--runs K`, `--quantiles Q1,Q2,...` (default `0.025,0.975`), `--epsilon E` (tau-leap only, default 0.03), `--model NAME`, `--out PATH`, `--strict` |

All commands exit with status 0 on success and 1 on any error.

## Develop

```bash
uv run pytest
```

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

## Roadmap

1. **Done:** grammar, typed IR, checks C1–C8, ODE backend, CLI.
2. **Done:** `compose` blocks with `sync` rules, rates that depend on another model
   (`by`) and entry states for inflows (`enter`).
3. **Done:** Gillespie and tau-leap stochastic backends, ensembles of runs.
4. **Done:** calibration by ABC-SMC.
5. Network-based agent backend, language server, PyPI release.

Design rationale is recorded in [docs/DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md).

## License

MIT
