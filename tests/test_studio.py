"""The studio: model description for the browser, and the local HTTP server."""

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from epidsl.cli import main
from epidsl.expr import format_expr
from epidsl.parser import parse
from epidsl.studio import api
from epidsl.studio.server import make_server

EXAMPLES = Path(__file__).parent.parent / "examples"
SIR = (EXAMPLES / "sir.epidsl").read_text()
AGE = (EXAMPLES / "sir_age_structured.epidsl").read_text()
HOSPITAL = (EXAMPLES / "sir_hospital.epidsl").read_text()
NETWORK = (EXAMPLES / "sir_network.epidsl").read_text()


def expr_text(text: str) -> str:
    (m,) = parse(f"model _ {{ flow -> X : {text} }}")
    return format_expr(m.flows[0].expr)


# --- printing expressions --------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "beta * S * I / N",
        "(1 - severe) * beta * S",
        "a - (b - c)",
        "a - b - c",
        "a / (b * c)",
        "a / b / c",
        "-a * b",
        "-(a + b)",
        "a ^ b ^ c",
        "(a ^ b) ^ c",
        "2 ^ -x",
        "exp(-t) + max(0, x)",
        "0.5 * 1e-06 * x",
        "rho * incidence(S -> I)",
        "incidence(-> S) - incidence(S ->)",
    ],
)
def test_format_expr_round_trips(text: str) -> None:
    printed = expr_text(text)
    assert printed == text
    assert expr_text(printed) == printed


def test_format_expr_drops_needless_parentheses() -> None:
    assert expr_text("((a * b)) + (c)") == "a * b + c"
    assert expr_text("(a + b) + c") == "a + b + c"


# --- describing a model ----------------------------------------------------------------


def described(source: str, name: str) -> dict[str, Any]:
    analysis = api.analyze(source)
    assert analysis["ok"]
    return next(m for m in analysis["models"] if m["name"] == name)


def test_plain_model_is_a_single_row() -> None:
    m = described(SIR, "SIR")

    assert m["grid"] == {"columns": ["S", "I", "R"], "rows": [""], "parts": ["SIR"]}
    assert [(c["name"], c["initial"], c["col"], c["row"]) for c in m["cells"]] == [
        ("S", 990.0, 0, 0),
        ("I", 10.0, 1, 0),
        ("R", 0.0, 2, 0),
    ]
    infection = m["flows"][0]
    assert (infection["source"], infection["target"]) == ("S", "I")
    assert infection["rate"] == "beta * S * I / N"
    assert (infection["kind"], infection["part"], infection["line"]) == ("local", 0, 11)
    assert m["semantics"] == ["ODE", "Gillespie", "TauLeap"]
    assert m["usable"] and m["diagnostics"] == []
    beta = m["constants"][0]
    assert (beta["name"], beta["kind"], beta["value"], beta["unit"]) == (
        "beta",
        "parameter",
        0.3,
        "1/day",
    )
    assert beta["prior"] == {"dist": "uniform", "a": 0.05, "b": 1.0}


def test_composed_model_is_a_grid_with_provenance() -> None:
    m = described(AGE, "AgeStructured")

    assert m["grid"] == {
        "columns": ["S", "I", "R"],
        "rows": ["young", "old"],
        "parts": ["Epidemic", "Age"],
    }
    cells = {c["name"]: (c["col"], c["row"]) for c in m["cells"]}
    assert cells == {
        "S_young": (0, 0),
        "S_old": (0, 1),
        "I_young": (1, 0),
        "I_old": (1, 1),
        "R_young": (2, 0),
        "R_old": (2, 1),
    }
    flows = {(f["source"], f["target"]): f for f in m["flows"]}
    young = flows[("S_young", "I_young")]
    assert young["rate"] == "beta_young * S_young * (I_young + I_old) / N"
    assert (young["kind"], young["part"]) == ("local", 0)
    ageing = flows[("S_young", "S_old")]
    assert (ageing["kind"], ageing["part"]) == ("local", 1)
    # Both infection arrows come from the one line that declares the flow.
    assert young["line"] == flows[("S_old", "I_old")]["line"]
    births = flows[(None, "S_young")]
    assert (births["kind"], births["part"]) == ("inflow", None)
    assert flows[("S_old", None)]["kind"] == "outflow"
    assert [o["name"] for o in m["observables"]] == ["cases_young", "cases_old"]
    assert m["observables"][0]["expr"] == "rho_young * incidence(S_young -> I_young)"


def test_synchronised_flow_is_marked() -> None:
    m = described(HOSPITAL, "Hospital")
    sync = next(f for f in m["flows"] if (f["source"], f["target"]) == ("S_home", "I_hospital"))
    assert (sync["kind"], sync["part"]) == ("sync", None)


def test_three_models_put_the_first_in_columns() -> None:
    source = (
        "model A { compartment a = 8, b  flow a -> b : a }\n"
        "model B { compartment c = 8, d  flow c -> d : c }\n"
        "model C { compartment e = 8, f  flow e -> f : e }\n"
        "compose M { models: [A, B, C] }"
    )
    m = described(source, "M")
    assert m["grid"]["columns"] == ["a", "b"]
    assert m["grid"]["rows"] == ["c_e", "c_f", "d_e", "d_f"]
    assert {c["name"]: (c["col"], c["row"]) for c in m["cells"]}["b_d_e"] == (1, 2)


def test_diagnostics_and_errors_are_reported() -> None:
    m = described(
        "model M {\n compartment S = 9, I = 1\n rate g = 1\n flow S -> I : nope * g * S\n}", "M"
    )
    assert [(d["code"], d["severity"], d["line"]) for d in m["diagnostics"]] == [
        ("C5", "warning", 3),
        ("C3", "error", 4),
    ]
    assert not m["usable"]

    broken = api.analyze("model M {\n  compartment S\n  flow S => I : 1\n}")
    assert not broken["ok"] and broken["models"] == []
    assert broken["error"]["kind"] == "syntax error" and broken["error"]["line"] == 3

    composition = api.analyze("model A { compartment a = 1 }\ncompose X { models: [A, B] }")
    assert composition["error"]["kind"] == "composition error"
    assert "unknown model 'B'" in composition["error"]["message"]


# --- running and calibrating -----------------------------------------------------------


def test_run_returns_series() -> None:
    result = api.run({"source": SIR, "model": "SIR", "t_end": 100, "points": 11})
    assert result["ok"]
    assert result["t"][0] == 0.0 and result["t"][-1] == 100.0 and len(result["t"]) == 11
    assert [s["name"] for s in result["series"]] == ["S", "I", "R"]
    s = result["series"][0]
    assert s["kind"] == "compartment" and s["mean"][0] == 990.0 and "low" not in s
    assert 58.8 < s["mean"][-1] < 60.6


def test_run_ensemble_returns_bands_and_observables() -> None:
    source = (EXAMPLES / "sir_reporting.epidsl").read_text()
    payload = {"source": source, "model": "Reported", "t_end": 28, "points": 5}
    result = api.run({**payload, "method": "gillespie", "runs": 20, "seed": 1})

    assert result["ok"]
    cases = next(s for s in result["series"] if s["name"] == "cases")
    assert cases["kind"] == "observable"
    bands = zip(cases["low"], cases["mean"], cases["high"], strict=True)
    assert all(low <= mean <= high for low, mean, high in bands)
    assert cases["high"][-1] > cases["low"][-1]
    assert api.run({**payload, "method": "gillespie", "runs": 20, "seed": 1}) == result


@pytest.mark.parametrize(
    ("change", "fragment"),
    [
        ({"model": "Nope"}, "no model named 'Nope'"),
        ({"source": "model {"}, "syntax error"),
        ({"method": "magic"}, "unknown method"),
        ({"t_end": -1}, "t_end"),
        ({"points": 1}, "points"),
        ({"points": 10**6}, "points"),
        ({"runs": 5}, "stochastic method"),
        ({"method": "gillespie", "runs": 10**6}, "runs"),
        ({"method": "gillespie", "epsilon": 0.1}, "epsilon"),
        (
            {
                "source": "model M { semantics: [Gillespie] compartment X = 1 flow X -> : X }",
                "model": "M",
            },
            "does not declare ODE",
        ),
    ],
)
def test_run_refusals(change: dict[str, Any], fragment: str) -> None:
    result = api.run({"source": SIR, "model": "SIR", "t_end": 10, "points": 5, **change})
    assert not result["ok"] and fragment in result["error"]


def test_calibrate_returns_posterior_fit_and_data() -> None:
    data = (EXAMPLES / "sir_observed.csv").read_text()
    payload = {"source": SIR, "model": "SIR", "data": data, "particles": 40, "populations": 3}
    result = api.calibrate({**payload, "seed": 1})

    assert result["ok"]
    (beta,) = result["posterior"]
    assert beta["name"] == "beta" and beta["low"] < 0.3 < beta["high"]
    assert beta["low"] < beta["median"] < beta["high"]
    assert beta["prior"] == {"dist": "uniform", "a": 0.05, "b": 1.0}
    assert result["simulations"] >= 120 and 0 < result["effective_sample_size"] <= 40
    assert result["data"]["t"] == [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    assert result["data"]["series"]["I"][1] == 69.0
    fit = result["fit"]
    assert fit["t"] == result["data"]["t"]
    assert max(next(s for s in fit["series"] if s["name"] == "I")["mean"]) > 200


def test_calibration_fit_is_comparable_with_incidence_data() -> None:
    # Weekly case counts: the fitted curve must be weekly counts too, not counts over
    # some finer grid.
    source = (EXAMPLES / "sir_reporting.epidsl").read_text()
    data = (EXAMPLES / "sir_weekly_cases.csv").read_text()
    result = api.calibrate(
        {
            "source": source,
            "model": "Reported",
            "data": data,
            "particles": 60,
            "populations": 4,
            "seed": 1,
        }
    )

    assert result["ok"]
    observed = result["data"]["series"]["cases"]
    fit = result["fit"]
    assert fit["t"] == [0.0, *result["data"]["t"]]
    fitted = next(s for s in fit["series"] if s["name"] == "cases")["mean"][1:]
    assert max(fitted) == pytest.approx(max(observed), rel=0.3)
    assert fitted.index(max(fitted)) == observed.index(max(observed))


def test_calibrate_refusals() -> None:
    base = {"source": SIR, "model": "SIR", "particles": 10, "populations": 2}
    assert "first column" in api.calibrate({**base, "data": "x,I\n0,1\n"})["error"]
    assert "'Z' is not a compartment" in api.calibrate({**base, "data": "t,Z\n0,1\n1,2\n"})["error"]
    assert "particles" in api.calibrate({**base, "data": "t,I\n0,1\n", "particles": 10**6})["error"]


# --- HTTP server -----------------------------------------------------------------------


class Client:
    def __init__(self, port: int) -> None:
        self.base = f"http://127.0.0.1:{port}"

    def request(
        self,
        path: str,
        body: Any = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, str, str]:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.base + path, data=data)
        if data is not None:
            request.add_header("Content-Type", "application/json")
        for key, value in (headers or {}).items():
            request.add_header(key, value)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return (
                    response.status,
                    response.headers.get_content_type(),
                    response.read().decode(),
                )
        except urllib.error.HTTPError as error:
            return error.code, error.headers.get_content_type(), error.read().decode()

    def json(self, path: str, body: Any = None) -> Any:
        status, kind, text = self.request(path, body)
        assert (status, kind) == (200, "application/json"), text
        return json.loads(text)


@pytest.fixture
def studio(tmp_path: Path) -> Iterator[tuple[Client, Path]]:
    source = tmp_path / "model.epidsl"
    source.write_text(SIR)
    server = make_server(source, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield Client(server.server_address[1]), source
    finally:
        server.shutdown()
        server.server_close()


def test_server_serves_the_page_and_its_assets(studio: tuple[Client, Path]) -> None:
    client, _ = studio
    status, kind, page = client.request("/")
    assert (status, kind) == (200, "text/html") and "EpiDSL Studio" in page
    assert client.request("/app.js")[:2] == (200, "text/javascript")
    assert client.request("/style.css")[:2] == (200, "text/css")
    assert client.request("/nope")[0] == 404
    assert client.request("/../pyproject.toml")[0] == 404
    # The page works offline: nothing is loaded from another host.
    assert "http://" not in page and "https://" not in page


def test_server_api_round_trip(studio: tuple[Client, Path]) -> None:
    client, source = studio
    loaded = client.json("/api/source")
    assert loaded == {"path": str(source), "name": "model.epidsl", "source": SIR}

    analysis = client.json("/api/analyze", {"source": SIR})
    assert analysis["ok"] and analysis["models"][0]["name"] == "SIR"

    result = client.json("/api/run", {"source": SIR, "model": "SIR", "t_end": 10, "points": 3})
    assert result["ok"] and len(result["t"]) == 3

    changed = SIR.replace("0.3", "0.4")
    assert client.json("/api/save", {"source": changed}) == {"ok": True}
    assert source.read_text() == changed


def test_server_rejects_what_it_should(studio: tuple[Client, Path]) -> None:
    client, source = studio
    body = {"source": SIR}

    # Requests made by another website, or for another host name, are refused.
    assert client.request("/api/analyze", body, {"Origin": "http://evil.example"})[0] == 403
    assert client.request("/api/analyze", body, {"Host": "evil.example"})[0] == 403
    assert client.request("/api/source", None, {"Host": "evil.example"})[0] == 403
    assert client.request("/api/analyze", body, {"Content-Type": "text/plain"})[0] == 415
    assert client.request("/api/nope", body)[0] == 404
    assert client.request("/api/analyze", [1, 2])[0] == 400
    assert client.request("/api/save", {"source": 3})[0] == 400
    assert source.read_text() == SIR

    request = urllib.request.Request(client.base + "/api/analyze", data=b"{not json")
    request.add_header("Content-Type", "application/json")
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(request, timeout=30)
    assert excinfo.value.code == 400


def test_server_without_a_file_cannot_save() -> None:
    server = make_server(None, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = Client(server.server_address[1])
        loaded = client.json("/api/source")
        assert loaded["path"] is None and "model SIR" in loaded["source"]
        assert api.analyze(loaded["source"])["ok"]
        assert client.request("/api/save", {"source": "x"})[0] == 409
    finally:
        server.shutdown()
        server.server_close()


def test_server_only_listens_locally(tmp_path: Path) -> None:
    server = make_server(None, port=0)
    try:
        assert server.server_address[0] == "127.0.0.1"
    finally:
        server.server_close()


def test_server_falls_back_to_a_free_port() -> None:
    first = make_server(None)
    try:
        second = make_server(None)
        try:
            assert first.server_address[1] != second.server_address[1]
            assert second.url == f"http://127.0.0.1:{second.server_address[1]}/"
        finally:
            second.server_close()
        with pytest.raises(OSError):
            make_server(None, port=first.server_address[1])
    finally:
        first.server_close()


def test_cli_studio_reports_a_missing_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "nope.epidsl"
    assert main(["studio", str(missing), "--no-browser"]) == 1
    assert str(missing) in capsys.readouterr().err


# --- D44 — réseau : describe() expose réseau et attributs ----------------------


_SIR_WITH_ATTRS = """\
model SIR {
    semantics: [Network]
    compartment S = 90, I = 10, R = 0
    attribute risk ~ categorical(Low=0.7, High=0.3)
    rate beta_low  = 0.1 [1/contact/day]
    rate beta_high = 0.5 [1/contact/day]
    rate gamma = 0.1 [1/day]
    network watts_strogatz(k=6, rewire=0.1)
    flow S -> I : beta_low * neighbor_attr(I, risk, Low)
               + beta_high * neighbor_attr(I, risk, High)
    flow I -> R : gamma
}
"""


def test_describe_network_model_has_network_key() -> None:
    m = described(NETWORK, "SIR")
    assert m["network"] is not None
    assert m["network"]["generator"] == "watts_strogatz"
    assert m["network"]["params"] == {"k": 6.0, "rewire": 0.1}


def test_describe_network_model_network_has_line() -> None:
    m = described(NETWORK, "SIR")
    assert isinstance(m["network"]["line"], int) and m["network"]["line"] > 0


def test_describe_non_network_model_network_is_null() -> None:
    m = described(SIR, "SIR")
    assert m["network"] is None


def test_describe_non_network_model_attributes_is_empty() -> None:
    m = described(SIR, "SIR")
    assert m["attributes"] == []


def test_describe_network_model_has_attributes_key() -> None:
    m = described(_SIR_WITH_ATTRS, "SIR")
    assert len(m["attributes"]) == 1
    attr = m["attributes"][0]
    assert attr["name"] == "risk"


def test_describe_network_attribute_categories() -> None:
    m = described(_SIR_WITH_ATTRS, "SIR")
    attr = m["attributes"][0]
    cats = {c["label"]: c["proportion"] for c in attr["categories"]}
    assert cats == {"Low": pytest.approx(0.7), "High": pytest.approx(0.3)}


def test_describe_network_attribute_has_line() -> None:
    m = described(_SIR_WITH_ATTRS, "SIR")
    attr = m["attributes"][0]
    assert isinstance(attr["line"], int) and attr["line"] > 0


def test_describe_network_model_no_attributes_has_empty_list() -> None:
    m = described(NETWORK, "SIR")
    assert m["attributes"] == []
