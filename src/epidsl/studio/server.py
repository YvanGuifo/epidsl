"""The studio's HTTP server. It listens on this machine only.

The page and its scripts are served from the package; nothing is fetched from
another host. Requests are answered only when they come from the page itself: a
request whose ``Host`` is not this server, or whose ``Origin`` is another site, is
refused, so that a web page open elsewhere in the browser cannot drive the studio.
"""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any

from epidsl.studio import api

_ASSETS = {
    "/": ("index.html", "text/html"),
    "/app.js": ("app.js", "text/javascript"),
    "/style.css": ("style.css", "text/css"),
}
_MAX_BODY = 4 * 1024 * 1024


class StudioServer(ThreadingHTTPServer):
    """Serves one model file, or the built-in template when ``source_path`` is None."""

    daemon_threads = True

    def __init__(self, source_path: Path | None, port: int) -> None:
        super().__init__(("127.0.0.1", port), _Handler)
        self.source_path = source_path

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/"


class _Handler(BaseHTTPRequestHandler):
    server: StudioServer

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _send(self, status: HTTPStatus, kind: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", f"{kind}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send(status, "application/json", json.dumps(value).encode("utf-8"))

    def _refuse(self, status: HTTPStatus, message: str) -> None:
        self._json({"ok": False, "error": message}, status)

    def _trusted(self) -> bool:
        port = self.server.server_address[1]
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        origin = self.headers.get("Origin")
        if self.headers.get("Host") not in hosts:
            return False
        return origin is None or origin in {f"http://{host}" for host in hosts}

    def do_GET(self) -> None:
        if not self._trusted():
            return self._refuse(HTTPStatus.FORBIDDEN, "request refused")
        if self.path == "/api/source":
            path = self.server.source_path
            if path is None:
                loaded = {"path": None, "name": "untitled.epidsl", "source": api.TEMPLATE}
            else:
                loaded = {
                    "path": str(path),
                    "name": path.name,
                    "source": path.read_text(encoding="utf-8"),
                }
            return self._json(loaded)
        if self.path in _ASSETS:
            name, kind = _ASSETS[self.path]
            body = files("epidsl.studio").joinpath("static", name).read_bytes()
            return self._send(HTTPStatus.OK, kind, body)
        self._refuse(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:
        if not self._trusted():
            return self._refuse(HTTPStatus.FORBIDDEN, "request refused")
        if self.headers.get_content_type() != "application/json":
            return self._refuse(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "send JSON")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if not 0 <= length <= _MAX_BODY:
            return self._refuse(HTTPStatus.BAD_REQUEST, "bad request size")
        try:
            payload = json.loads(self.rfile.read(length))
        except ValueError:
            return self._refuse(HTTPStatus.BAD_REQUEST, "the request is not JSON")
        if not isinstance(payload, dict):
            return self._refuse(HTTPStatus.BAD_REQUEST, "the request must be a JSON object")

        if self.path == "/api/analyze":
            source = payload.get("source")
            if not isinstance(source, str):
                return self._refuse(HTTPStatus.BAD_REQUEST, "source must be text")
            return self._json(api.analyze(source))
        if self.path == "/api/run":
            return self._json(api.run(payload))
        if self.path == "/api/calibrate":
            return self._json(api.calibrate(payload))
        if self.path == "/api/save":
            source = payload.get("source")
            if not isinstance(source, str):
                return self._refuse(HTTPStatus.BAD_REQUEST, "source must be text")
            path = self.server.source_path
            if path is None:
                return self._refuse(HTTPStatus.CONFLICT, "the studio was started without a file")
            path.write_text(source, encoding="utf-8")
            return self._json({"ok": True})
        self._refuse(HTTPStatus.NOT_FOUND, "not found")


#: Tried first when no port is asked for.
DEFAULT_PORT = 8642


def make_server(source_path: Path | None, port: int | None = None) -> StudioServer:
    """Create the server. Call ``serve_forever`` to run it.

    With ``port`` None, the default port is used if it is free and any free port
    otherwise; 0 always picks a free port. An explicit port that is taken raises OSError.
    """
    if port is not None:
        return StudioServer(source_path, port)
    try:
        return StudioServer(source_path, DEFAULT_PORT)
    except OSError:
        return StudioServer(source_path, 0)
