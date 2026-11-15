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
