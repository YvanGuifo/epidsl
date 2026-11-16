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
  stratified with `by`, entry states with `enter`, and constants and observables
  declared in the block.
- **Backends.** Deterministic (ODE, LSODA), exact stochastic (Gillespie) and
  approximate stochastic (tau-leap with adaptive steps).
- **Ensembles.** `--runs` writes the mean and quantiles of many stochastic runs.
- **Observables.** `observe` statements and `incidence(A -> B)`.
- **Calibration.** Priors on parameters and `epidsl calibrate`, by ABC-SMC.
- **Studio.** `epidsl studio` opens a visual workbench in the browser: the model
  as a grid of compartments and flows linked to the source, live diagnostics,
  simulation and calibration.
- **Command line.** `epidsl check`, `epidsl run`, `epidsl calibrate`, `epidsl studio`.

### Known limits

- Rates that depend on time are refused by the stochastic backends.
- A contact matrix (a rate that depends on the strata of both parties) cannot be
  expressed.
- Observables are deterministic functions of the simulation: no measurement noise
  and no reporting delay.
- Simulations run one after another.
