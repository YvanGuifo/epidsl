# Changelog

All notable changes to EpiDSL are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Fonction `neighbor_attr(state, attr_name, value)` pour le mélange hétérogène (D41).**
  Renvoie le nombre de voisins dans le compartiment `state` ET dont l'attribut `attr_name`
  vaut `value`.  Combinée avec `attr()` (D39) et `contacts()` (D38), elle permet d'exprimer
  une matrice de mélange bilinéaire complète dans une seule expression de taux :
  `flow S -> I : beta_low * neighbor_attr(I, risk, Low) + beta_high * neighbor_attr(I, risk, High)`.
  Réservée à la sémantique Network.  Nouveaux contrôles statiques : C-NA1 (arité incorrecte),
  C-NA2 (hors sémantique Network), C-NA3 (état inconnu), C-NA4 (attribut non déclaré).
  Exemple : `examples/sir_neighbor_attr.epidsl`.

- **Règles de transition agent-agent `interact A + B -> A' + B' : expr` (D40).**
  Un flux de paire déclenche deux transitions simultanées : l'agent focal passe de A à A'
  et un voisin choisi aléatoirement parmi ses voisins en état B passe à B'.  Le taux pour
  l'agent focal est `expr × count(B-voisins)` : `expr` est le taux par paire, la
  multiplication par le compte de voisins est implicite.  Utile pour la dynamique d'opinion
  (voter model), la co-infection, la récupération couplée à la transmission.  Réservé à la
  sémantique Network.  Nouveaux contrôles statiques : C-I1 (état non déclaré), C-I2 (absent
  de sémantique Network).  Exemple : `examples/opinion_dynamics.epidsl`.

- **Agents hétérogènes via attributs discrets `attribute … ~ categorical(…)` (D39).**
  Chaque agent reçoit un attribut tiré d'une distribution catégorielle à la
  construction du modèle (ex. `attribute risk ~ categorical(Low=0.7, High=0.3)`).
  La fonction `attr(name, value)` dans une expression de taux renvoie 1.0 si
  l'agent focal possède la valeur indiquée, 0.0 sinon.  Combinée avec
  `contacts(X)`, elle permet de modéliser des taux d'infection différenciés par
  groupe de risque.  Nouveaux contrôles statiques : C-A1 (`attr()` hors sémantique
  Network), C-A2 (`attr()` arité incorrecte), C-A3 (attribut non déclaré).
  Exemple : `examples/sir_attributes.epidsl`.

- **Contact network backend `NetworkBackend` (D38).** New `semantics: [Network]`
  with a `network <generator>(params)` clause runs a Gillespie-on-network
  simulation: each agent is a graph node; `contacts(X)` in a rate expression
  evaluates to the number of neighbours currently in state X.  Four generators are
  supported: `erdos_renyi(p)`, `watts_strogatz(k, rewire)`,
  `barabasi_albert(m)`, `complete()`.  New static checks C-N1 (Network semantics
  without network clause) and C-N2 (`contacts()` wrong arity).  Performance:
  ~14 s for N=1000 agents, Watts-Strogatz k=6, t_end=50 (pure-Python loop;
  intended for small populations).  Example: `examples/sir_network.epidsl`.

- **Conditional expression `ifelse(cond, a, b)` (D37).** Returns `a` if
  `cond > 0`, `b` otherwise.  Enables explicit mode-switching in rate
  expressions without introducing time-step artefacts, e.g.
  `ifelse(t - t_lock, beta_lo, beta_hi)` for a hard lockdown at day 30.
  For state-dependent conditions (e.g. `ifelse(I - 500, beta_lo, beta_hi)`),
  the ODE solver does not inject breakpoints — the transition is handled by
  LSODA's adaptive step size, which may be slow near the threshold;
  for such cases `hill()` or `clamp()` are preferable if a smooth switch is
  acceptable.  Example: `examples/sir_ifelse.epidsl`.

- **Hill saturation function `hill(x, k, n)` (D36).** Returns `xⁿ / (kⁿ + xⁿ)`:
  0 at `x=0`, 0.5 at `x=k` (the half-saturation constant), and approaching 1 as
  `x → ∞`.  For `n=1`: standard Michaelis-Menten kinetics.  Useful for modelling
  hospital overcrowding, immune response saturation, and dose-response
  relationships in rate expressions.  No ODE breakpoints are injected (continuous
  function).  Example: `examples/sir_hill.epidsl`.

- **Smooth cubic interpolation `smoothstep(t, t0, t1)` (D35).** Returns 0 for
  `t <= t0`, 1 for `t >= t1`, and `3s²–2s³` (where `s = (t–t0)/(t1–t0)`) in
  between.  Unlike `ramp`, the derivative is 0 at both endpoints (C¹ continuous),
  so there is no abrupt change of slope when the intervention starts or ends.  No
  ODE breakpoints are injected (LSODA handles smooth kinks natively).  Example:
  `examples/sir_smooth.epidsl`.

- **Value-bounding function `clamp(x, lo, hi)` (D34).** Returns `lo` if
  `x < lo`, `hi` if `x > hi`, and `x` otherwise.  Useful for vaccination
  models where `clamp(vacc_rate * t, 0, 1)` guarantees coverage stays in
  `[0, 1]` even when the driving expression extrapolates outside the physical
  range.  No ODE breakpoints are injected (the function is continuous).
  Example: `examples/sir_clamp.epidsl`.

- **Rectangular pulse function for temporary interventions (D33).** `pulse(t, t0, t1)`
  returns 1 for `t0 <= t < t1` (closed left, open right) and 0 otherwise, enabling
  finite-duration interventions such as temporary lockdowns, school closures, or
  short vaccination campaigns in rate expressions.  The ODE backend injects both
  `t0` and `t1` as breakpoints in `t_eval` so LSODA crosses both discontinuities
  cleanly, then trims the output back to the requested grid.  Example:
  `examples/sir_pulse.epidsl`.

- **Linear ramp function for progressive interventions (D32).** `ramp(t, t0, t1)`
  returns 0 for `t <= t0`, 1 for `t >= t1`, and `(t - t0) / (t1 - t0)` in
  between, enabling gradual interventions (progressive lockdowns, staged
  vaccination campaigns) in rate expressions.  The ODE backend injects both
  `t0` and `t1` as breakpoints in `t_eval` so LSODA crosses both kinks cleanly,
  then trims the output back to the requested grid.  Example:
  `examples/sir_ramp.epidsl`.

- **Heaviside step function for interventions (D31).** `step(t, t0)` returns 0
  for `t < t0` and 1 for `t >= t0`, enabling time-triggered interventions such
  as lockdowns and vaccination campaigns directly in rate expressions (e.g.
  `beta * (1 - phi * step(t, 30)) * S * I`).  The ODE backend automatically
  injects `t0` as an extra breakpoint in `t_eval` so LSODA crosses the
  discontinuity cleanly, then trims the output back to the requested grid.
  Example: `examples/sir_lockdown.epidsl`.

- **Contact matrices for stratified transmission (D30).** `parameter beta by
  Model, Model = { i, j: value, ... }` declares a contact-rate matrix in a
  `compose` block.  A flow that uses such a parameter is automatically expanded
  into the sum `Σ_j beta_{i,j} * S_i * I_j` over all strata of the second
  model (the state met).  The parameter may be declared inline in the `compose`
  block without a prior declaration in any component model.  All four dynamics
  backends support the expansion unchanged.  Missing entries in a contact
  matrix are an error (unlike sparse migration matrices where omitted entries
  default to 0).  Example: `examples/sir_age.epidsl`.

- **Reporting delay on observables (D29).** An observable may declare a
  reporting delay with `observe X = expr delay(d)`, where `d` is a number of
  days (or whatever the model's time unit is).  The backend shifts the output
  time series right by `d` days, zero-padding the leading window.  Negative
  delays are rejected by check C14.  The clause may be combined with a noise
  distribution: `observe X = expr ~ poisson() delay(7)`.

- **Sparse migration matrices (D28).** Entries omitted from a `by M, M`
  matrix used in a `migrate` statement now default to 0 (no edge), so only
  non-zero travel rates need to be listed.  The dense syntax remains valid.
  Contact-rate and other non-migration matrices still require all entries.

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

- *(No remaining known limits from v0.1.0 — resolved in Unreleased.)*
