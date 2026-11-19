# Changelog

All notable changes to EpiDSL are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

First version, not yet published.

### Added

- **Language.** Models with compartments, `parameter` and `rate` constants with
  optional units, flows (including inflows and outflows), a `semantics` list, line
  and block comments.
- **Static checks** C1–C8, C11 and C12, with source positions; `--strict` turns
  warnings into errors.
- **Composition.** `compose` blocks: product of models, `sync` rules, constants
  stratified with `by`, contact matrices with `by X, X`, entry states with `enter`,
  and constants and observables declared in the block.
- **Backends.** Deterministic (ODE, LSODA), exact stochastic (Gillespie) and
  approximate stochastic (tau-leap with adaptive steps).
- **Ensembles.** `--runs` writes the mean and quantiles of many stochastic runs.
- **Several processors.** `--workers` spreads an ensemble or a calibration over
  processes without changing its result.
- **Observables.** `observe` statements and `incidence(A -> B)`.
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

### Known limits

- Exact stochastic simulation is much slower when a rate depends on time.
- A contact matrix (a rate that depends on the strata of both parties) cannot be
  expressed.
- No reporting delay on observables.
- Exact stochastic simulation with time-dependent rates uses numerical integration (slower).
