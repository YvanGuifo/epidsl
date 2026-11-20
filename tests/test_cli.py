import csv
from pathlib import Path

import pytest

from epidsl.cli import main

EXAMPLES = Path(__file__).parent.parent / "examples"
SIR = str(EXAMPLES / "sir.epidsl")


def test_check_clean_model(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", SIR]) == 0
    assert "SIR: ok" in capsys.readouterr().out


def test_check_reports_errors_on_stderr(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.epidsl"
    bad.write_text("model M {\n  compartment S\n  flow S -> : nope\n}")
    assert main(["check", str(bad)]) == 1
    assert f"{bad}:3:15: error C3" in capsys.readouterr().err


def test_check_strict_promotes_warnings(tmp_path: Path) -> None:
    src = tmp_path / "w.epidsl"
    src.write_text("model M { compartment S = 1  rate g = 1  flow S -> : g * S }")
    assert main(["check", str(src)]) == 0
    assert main(["check", "--strict", str(src)]) == 1


def test_syntax_error_exit_code(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = tmp_path / "s.epidsl"
    src.write_text("model {")
    assert main(["check", str(src)]) == 1
    assert "syntax error" in capsys.readouterr().err


def test_run_writes_csv(tmp_path: Path) -> None:
    out = tmp_path / "sir.csv"
    assert main(["run", SIR, "--t-end", "100", "--points", "11", "--out", str(out)]) == 0

    rows = list(csv.reader(out.open()))
    assert rows[0] == ["t", "S", "I", "R"]
    assert len(rows) == 12
    assert [float(x) for x in rows[1]] == [0.0, 990.0, 10.0, 0.0]
    assert float(rows[-1][0]) == 100.0


def test_run_prints_csv_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", SIR, "--t-end", "1", "--points", "2"]) == 0
    assert capsys.readouterr().out.splitlines()[0] == "t,S,I,R"


def test_run_selects_model_by_name(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = tmp_path / "two.epidsl"
    src.write_text(
        "model A { compartment X = 1  flow X -> : X }\nmodel B { compartment Y = 2  flow Y -> : Y }"
    )
    assert main(["run", str(src), "--t-end", "1"]) == 1
    assert "--model" in capsys.readouterr().err
    assert main(["run", str(src), "--t-end", "1", "--model", "B", "--points", "2"]) == 0
    assert capsys.readouterr().out.splitlines()[0] == "t,Y"


def test_composed_model_is_checked_and_run(capsys: pytest.CaptureFixture[str]) -> None:
    source = str(EXAMPLES / "sir_age.epidsl")
    assert main(["check", source]) == 0
    assert capsys.readouterr().out.splitlines() == ["SIR: ok", "Age: ok", "AgeSIR: ok"]

    assert main(["run", source, "--model", "AgeSIR", "--t-end", "10", "--points", "2"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "t,S_young,S_old,I_young,I_old,R_young,R_old"
    assert lines[1] == "0.0,297.0,693.0,3.0,7.0,0.0,0.0"


def test_composition_error_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = tmp_path / "c.epidsl"
    src.write_text(
        "model A { compartment a = 1, b  flow a -> b : a }\ncompose X { models: [A, B] }"
    )
    assert main(["check", str(src)]) == 1
    assert f"{src}:2:25: composition error: unknown model 'B'" in capsys.readouterr().err


def test_run_c10_undeclared_semantics(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = tmp_path / "ode_only.epidsl"
    src.write_text(
        "model M {\n"
        "  semantics: [ODE]\n"
        "  compartment S = 990, I = 10, R = 0\n"
        "  rate beta = 0.3 [1/day]  rate gamma = 0.1 [1/day]\n"
        "  flow S -> I : beta * S * I / N  flow I -> R : gamma * I\n"
        "}\n"
    )
    assert main(["run", str(src), "--t-end", "10", "--method", "gillespie"]) == 1
    err = capsys.readouterr().err
    assert "C10" in err
    assert "Gillespie" in err


def test_run_declared_semantics_succeeds(tmp_path: Path) -> None:
    src = tmp_path / "full.epidsl"
    src.write_text(
        "model M {\n"
        "  semantics: [ODE, Gillespie, TauLeap]\n"
        "  compartment S = 990, I = 10, R = 0\n"
        "  rate beta = 0.3 [1/day]  rate gamma = 0.1 [1/day]\n"
        "  flow S -> I : beta * S * I / N  flow I -> R : gamma * I\n"
        "}\n"
    )
    assert main(["run", str(src), "--t-end", "1", "--points", "2", "--method", "gillespie", "--seed", "1"]) == 0


@pytest.mark.parametrize("example", sorted(EXAMPLES.glob("*.epidsl")), ids=lambda p: p.name)
def test_every_example_is_clean(example: Path) -> None:
    assert main(["check", "--strict", str(example)]) == 0


def test_run_gillespie_is_reproducible_with_seed(capsys: pytest.CaptureFixture[str]) -> None:
    args = ["run", SIR, "--t-end", "50", "--points", "6", "--method", "gillespie", "--seed", "7"]
    assert main(args) == 0
    first = capsys.readouterr().out
    assert main(args) == 0
    assert capsys.readouterr().out == first

    rows = [line.split(",") for line in first.splitlines()]
    assert rows[0] == ["t", "S", "I", "R"]
    assert all(sum(float(x) for x in row[1:]) == 1000.0 for row in rows[1:])


def test_run_gillespie_refuses_undeclared_model(capsys: pytest.CaptureFixture[str]) -> None:
    source = str(EXAMPLES / "sir_demography.epidsl")
    assert main(["run", source, "--t-end", "1", "--method", "gillespie"]) == 1
    assert "does not declare Gillespie semantics" in capsys.readouterr().err


def test_run_tau_leap(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = tmp_path / "big.epidsl"
    src.write_text(
        "model Big { semantics: [TauLeap] compartment S = 99000, I = 1000, R = 0 "
        "rate b = 0.3 [1/day] rate g = 0.1 [1/day] "
        "flow S -> I : b * S * I / N  flow I -> R : g * I }"
    )
    args = ["run", str(src), "--t-end", "50", "--points", "3", "--method", "tau-leap"]
    assert main([*args, "--seed", "7"]) == 0
    rows = [line.split(",") for line in capsys.readouterr().out.splitlines()]
    assert rows[0] == ["t", "S", "I", "R"]
    assert all(sum(float(x) for x in row[1:]) == 100000.0 for row in rows[1:])
    assert float(rows[-1][3]) > 0


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert capsys.readouterr().out.startswith("epidsl 0.")
