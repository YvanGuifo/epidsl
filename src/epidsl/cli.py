"""Command-line interface: ``epidsl check`` and ``epidsl run``."""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from epidsl.backends import Backend, BackendResult
from epidsl.backends.gillespie import GillespieBackend
from epidsl.backends.ode import ODEBackend
from epidsl.backends.tau_leap import TauLeapBackend
from epidsl.checks import Severity, check
from epidsl.ensemble import EnsembleResult, run_ensemble
from epidsl.errors import EpiError
from epidsl.ir import EpiModel, SimulationConfig
from epidsl.parser import parse

_METHODS = ("ode", "gillespie", "tau-leap")


def _backend(method: str, epsilon: float | None) -> Backend:
    if epsilon is not None and method != "tau-leap":
        raise ValueError("--epsilon applies only to --method tau-leap")
    if method == "gillespie":
        return GillespieBackend()
    if method == "tau-leap":
        return TauLeapBackend() if epsilon is None else TauLeapBackend(epsilon=epsilon)
    return ODEBackend()


def _load(path: Path) -> tuple[EpiModel, ...] | None:
    try:
        return parse(path.read_text(encoding="utf-8"))
    except OSError as error:
        print(f"{path}: {error.strerror}", file=sys.stderr)
    except EpiError as error:
        print(f"{path}:{error}", file=sys.stderr)
    return None


def _report(path: Path, model: EpiModel, strict: bool) -> bool:
    """Print the model's diagnostics; return True if it is usable."""
    diagnostics = check(model)
    for diagnostic in diagnostics:
        print(f"{path}:{diagnostic}", file=sys.stderr)
    return not any(strict or d.severity is Severity.ERROR for d in diagnostics)


def _check(args: argparse.Namespace) -> int:
    models = _load(args.file)
    if models is None:
        return 1
    ok = True
    for model in models:
        if _report(args.file, model, args.strict):
            print(f"{model.name}: ok")
        else:
            ok = False
    return 0 if ok else 1


def _write_csv(result: BackendResult, stream: TextIO) -> None:
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["t", *result.y])
    for i, t in enumerate(result.t):
        writer.writerow([repr(float(t)), *(repr(float(series[i])) for series in result.y.values())])


def _write_summary(ensemble: EnsembleResult, quantiles: Sequence[float], stream: TextIO) -> None:
    """One row per time; per compartment, the mean then each quantile over the runs."""
    columns: list[tuple[str, list[float]]] = []
    mean = ensemble.mean()
    bands = [(q, ensemble.quantile(q)) for q in quantiles]
    for name in ensemble.names:
        columns.append((f"{name}_mean", mean[name].tolist()))
        columns.extend((f"{name}_q{q:g}", band[name].tolist()) for q, band in bands)

    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["t", *(header for header, _ in columns)])
    for i, t in enumerate(ensemble.t):
        writer.writerow([repr(float(t)), *(repr(values[i]) for _, values in columns)])


def _run(args: argparse.Namespace) -> int:
    models = _load(args.file)
    if models is None:
        return 1
    if args.model is not None:
        selected = [m for m in models if m.name == args.model]
        if not selected:
            print(f"{args.file}: no model named '{args.model}'", file=sys.stderr)
            return 1
        model = selected[0]
    elif len(models) == 1:
        model = models[0]
    else:
        names = ", ".join(m.name for m in models)
        print(f"{args.file}: several models ({names}); choose one with --model", file=sys.stderr)
        return 1

    if not _report(args.file, model, args.strict):
        return 1
    try:
        backend = _backend(args.method, args.epsilon)
        config = SimulationConfig(t_end=args.t_end, n_points=args.points, seed=args.seed)
        if args.runs is None:
            result = backend.run(model, config)

            def write(stream: TextIO) -> None:
                _write_csv(result, stream)
        elif args.method == "ode":
            raise ValueError("--runs needs a stochastic method (gillespie or tau-leap)")
        else:
            ensemble = run_ensemble(backend, model, config, args.runs)

            def write(stream: TextIO) -> None:
                _write_summary(ensemble, args.quantiles, stream)
    except (ValueError, RuntimeError) as error:
        print(f"{args.file}: {error}", file=sys.stderr)
        return 1

    if args.out is None:
        write(sys.stdout)
    else:
        with args.out.open("w", newline="", encoding="utf-8") as stream:
            write(stream)
    return 0


def _points(text: str) -> int:
    value = int(text)
    if value < 2:
        raise argparse.ArgumentTypeError("must be at least 2")
    return value


def _runs(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _quantiles(text: str) -> list[float]:
    try:
        values = [float(part) for part in text.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError("expected numbers separated by commas") from None
    if not all(0 <= value <= 1 for value in values):
        raise argparse.ArgumentTypeError("each quantile must be between 0 and 1")
    return values


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="epidsl", description="EpiDSL compiler and simulator")
    commands = parser.add_subparsers(dest="command", required=True)

    check_cmd = commands.add_parser("check", help="parse a file and run the static checks")
    check_cmd.add_argument("file", type=Path)
    check_cmd.add_argument("--strict", action="store_true", help="treat warnings as errors")
    check_cmd.set_defaults(handler=_check)

    run_cmd = commands.add_parser("run", help="simulate a model and write trajectories as CSV")
    run_cmd.add_argument("file", type=Path)
    run_cmd.add_argument("--t-end", type=float, required=True, help="end time of the simulation")
    run_cmd.add_argument("--points", type=_points, default=201, help="number of output points")
    run_cmd.add_argument(
        "--method",
        choices=list(_METHODS),
        default="ode",
        help="simulation method (default: ode)",
    )
    run_cmd.add_argument("--seed", type=int, help="random seed for stochastic methods")
    run_cmd.add_argument(
        "--runs",
        type=_runs,
        help="run this many independent simulations and write their mean and quantiles",
    )
    run_cmd.add_argument(
        "--quantiles",
        type=_quantiles,
        default=[0.025, 0.975],
        help="quantiles written with --runs, comma-separated (default: 0.025,0.975)",
    )
    run_cmd.add_argument(
        "--epsilon",
        type=float,
        help="accuracy of tau-leap, between 0 and 1; smaller is more accurate (default: 0.03)",
    )
    run_cmd.add_argument("--model", help="model to run when the file declares several")
    run_cmd.add_argument("--out", type=Path, help="CSV file to write (default: standard output)")
    run_cmd.add_argument("--strict", action="store_true", help="treat warnings as errors")
    run_cmd.set_defaults(handler=_run)

    args = parser.parse_args(argv)
    handler = args.handler
    return int(handler(args))


if __name__ == "__main__":
    sys.exit(main())
