# EpiDSL

A domain-specific language for epidemiological modelling. A model is written once as
compartments and flows, checked statically, and compiled to a typed intermediate
representation that simulation backends consume.

**Status:** early development. The ODE backend works end to end; model composition
and stochastic backends are not implemented yet (see [Roadmap](#roadmap)).

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

## Language reference

| Statement | Meaning |
|-----------|---------|
| `model Name { ... }` | Declares a model. A file may hold several. |
| `compartment A = 10, B` | Declares compartments with initial values (default 0). |
| `parameter p = 0.3 [1/day]` | A named constant intended for calibration. The unit is optional. |
| `rate g = 0.1 [1/day]` | A fixed named constant. The unit is optional. |
| `flow A -> B : expr` | Moves population from `A` to `B` at rate `expr`. |
| `flow -> A : expr` | Inflow (for example births). |
| `flow A -> : expr` | Outflow (for example deaths). |
| `semantics: [ODE]` | Backends the model is valid for: `ODE`, `Gillespie`, `TauLeap`, `ABM`. Default `[ODE]`. |

Expressions support `+ - * / ^`, parentheses, the built-in symbols `N` (total
population) and `t` (time), and the functions `exp`, `log`, `sqrt`, `sin`, `cos`,
`abs`, `min`, `max`. Comments are `-- to end of line` and `/* block */`. A
trailing `;` after a statement is optional.

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

`--strict` turns warnings into errors.

### Command reference

| Command | Options |
|---------|---------|
| `epidsl check FILE` | `--strict` |
| `epidsl run FILE` | `--t-end T` (required), `--points N` (default 201), `--model NAME`, `--out PATH`, `--strict` |

Both commands exit with status 0 on success and 1 on any error.

## Develop

```bash
uv run pytest
```

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

## Roadmap

1. **Done:** grammar, typed IR, checks C1–C8, ODE backend, CLI.
2. `compose` blocks with synchronisation rules (stochastic automata network composition).
3. Stochastic backends: Gillespie SSA, then tau-leap.
4. Network-based agent backend, calibration, language server, PyPI release.

Design rationale is recorded in [docs/DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md).

## License

MIT
