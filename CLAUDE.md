# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- **Run tests:** `uv run pytest`
- **Run a single test:** `uv run pytest tests/test_filename.py -k test_name`
- **Check code (lint & format check):** `uv run ruff check . && uv run ruff format --check . && uv run mypy`
- **Format code:** `uv run ruff format .`
- **CLI usage:** `uv run epidsl check FILE`, `uv run epidsl run FILE --t-end T`, `uv run epidsl calibrate FILE --data CSV`, `uv run epidsl studio [FILE]`

## Architecture & Code Structure

EpiDSL is a domain-specific language for compositional epidemiological modelling, built in Python using `uv`, `lark`, `numpy`, and `scipy`.

- `src/epidsl/` contains the core package source files:
  - `grammar.lark`: Lark grammar defining the DSL syntax (`model`, `compose`, compartments, flows, parameters, rates, observations, sync, by, enter).
  - `parser.py`: Parses DSL files into AST / intermediate representation.
  - `ir.py`: Typed intermediate representation and data structures for models and composed models.
  - `checks.py`: Static semantic checks (C1–C12 error/warning validation).
  - `calibration.py`: ABC-SMC calibration engine supporting multi-processing (`--workers`).
  - `cli.py`: Command-line interface (`check`, `run`, `calibrate`, `studio`).
- Simulation backends and Studio web workbench live within the package modules.
- `tests/`: Comprehensive test suite (`pytest`).
- `examples/`: Reference models and CSV datasets.
