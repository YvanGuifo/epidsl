# Changelog

All notable changes to EpiDSL are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-20

### Added

- **Language.** Models with compartments, `parameter` and `rate` constants with
  optional units, flows (including inflows and outflows), a `semantics` list, line
  and block comments.
- **Static checks** C1–C8, C11 and C12, with source positions; `--strict` turns
  warnings into errors.
- **Language server.** `epidsl server` starts an LSP server over stdio.  Provides
  diagnostics in real time (all static checks on every save), completion for keywords,
  math functions, and declared symbols, and goto-definition for compartments,
  constants and observables.  Any editor with LSP support can use it.
- **C9 — open flows without a counterpart.** Warning when a model has inflows but no
  outflows, or outflows but no inflows (unbounded growth or decay).  Demographic models
  with both are not warned.
- **C10 — backend not declared in `semantics`.** Error when `epidsl run` or
  `epidsl calibrate` requests a backend (`ode`, `gillespie`, `tau-leap`) that the
  model's `semantics` list does not include.  Replaces a `ValueError` from deep in
  the simulation engine with a structured diagnostic at CLI time.
- **Composition.** `compose` blocks: product of models, `sync` rules, constants
  stratified with `by`, contact matrices with `by X, X`, entry states with `enter`,
  and constants and observables declared in the block.
- **Backends.** Deterministic (ODE, LSODA), exact stochastic (Gillespie) and
  approximate stochastic (tau-leap with adaptive steps).
- **Ensembles.** `--runs` writes the mean and quantiles of many stochastic runs.
- **Several processors.** `--workers` spreads an ensemble or a calibration over
  processes without changing its result.
- **Observables.** `observe` statements and `incidence(A -> B)`.
- **Metapopulation migration.** A `migrate` statement in a `compose` block generates
  cross-patch flows for listed compartments according to a matrix constant declared with
  `rate X by Model, Model = {...}`.  Stochastic backends support migration natively.
  Example: `examples/metapop_sir.epidsl`.
- **Inline matrix constants.** A `rate` or `parameter` stratified with `by Model` or
  `by Model, Model` may now be declared inline in a `compose` block without a prior
  declaration in a component model.
- **Observation noise.** An observable may declare a noise distribution with
  `observe X = expr ~ poisson()`, `~ normal(sd)`, or `~ negbinomial(phi)`.
  The ABC-SMC calibration uses the negative log-likelihood of the declared
  distribution instead of RMSE, giving better posteriors for count data.
  Stochastic backends (Gillespie, tau-leap) draw from the noise distribution at
  each output time; the ODE backend always returns the deterministic mean.
  Check C13 validates noise parameters.
- **Calibration.** Priors on parameters and `epidsl calibrate`, by ABC-SMC.
- **Studio.** `epidsl studio` opens a visual workbench in the browser: the model
  as a grid of compartments and flows linked to the source, live diagnostics,
  simulation and calibration.
- **Command line.** `epidsl check`, `epidsl run`, `epidsl calibrate`, `epidsl studio`.

- **Fast Gillespie inversion for time-variable rates (D27).** Gillespie simulation with
  time-dependent rates is now 14× faster: 0.78 s instead of 11 s for
  ``examples/sir_seasonal.epidsl`` (120 days, 30 000 events, seed 42).  Newton-Raphson
  with a Simpson-rule integral (3–12 rate evaluations per jump) replaces
  ``quad + brentq`` (~160 evaluations per jump).  A slow fallback preserves
  correctness for non-monotone or near-zero rates.

### Known limits

- A contact matrix (a rate that depends on the strata of both parties) cannot be
  expressed.
- No reporting delay on observables.
