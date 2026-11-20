"""Language Server Protocol server for EpiDSL.

Exposes three pure functions that are tested independently, then wires them
into a pygls LanguageServer for use over stdio.
"""

from __future__ import annotations

from lsprotocol import types
from pygls.lsp.server import LanguageServer

from epidsl.checks import Diagnostic, Severity, check
from epidsl.errors import EpiError
from epidsl.expr import BUILTIN_SYMBOLS, FUNCTIONS
from epidsl.parser import parse

_KEYWORDS = [
    "model",
    "compose",
    "compartment",
    "rate",
    "parameter",
    "flow",
    "observe",
    "semantics",
    "sync",
    "by",
    "enter",
    "migrate",
    "ode",
    "gillespie",
    "tau-leap",
]


# ---------------------------------------------------------------------------
# Pure functions — testable without a running server
# ---------------------------------------------------------------------------


def _to_lsp_diagnostic(diag: Diagnostic) -> types.Diagnostic:
    severity = (
        types.DiagnosticSeverity.Error
        if diag.severity is Severity.ERROR
        else types.DiagnosticSeverity.Warning
    )
    # IR positions are 1-based; LSP is 0-based
    line = max(diag.line - 1, 0)
    col = max(diag.column - 1, 0)
    return types.Diagnostic(
        range=types.Range(
            start=types.Position(line=line, character=col),
            end=types.Position(line=line, character=col + 1),
        ),
        message=diag.message,
        code=diag.code,
        severity=severity,
        source="epidsl",
    )


def _check_document(text: str) -> list[types.Diagnostic]:
    """Parse and check ``text``; return LSP diagnostics (0-based positions)."""
    try:
        models = parse(text)
    except EpiError as exc:
        line = max(exc.line - 1, 0)
        col = max(exc.column - 1, 0)
        return [
            types.Diagnostic(
                range=types.Range(
                    start=types.Position(line=line, character=col),
                    end=types.Position(line=line, character=col + 1),
                ),
                message=str(exc),
                code="parse",
                severity=types.DiagnosticSeverity.Error,
                source="epidsl",
            )
        ]
    result: list[types.Diagnostic] = []
    for model in models:
        result.extend(_to_lsp_diagnostic(d) for d in check(model))
    return result


def _get_completions(text: str, line: int, char: int) -> list[types.CompletionItem]:
    """Return completions regardless of cursor position (global scope)."""
    names: set[str] = set()
    try:
        models = parse(text)
        for model in models:
            names.update(c.name for c in model.compartments)
            names.update(k.name for k in model.constants)
    except EpiError:
        pass

    items: list[types.CompletionItem] = []
    for kw in _KEYWORDS:
        items.append(types.CompletionItem(label=kw, kind=types.CompletionItemKind.Keyword))
    for func in FUNCTIONS:
        items.append(types.CompletionItem(label=func, kind=types.CompletionItemKind.Function))
    for sym in BUILTIN_SYMBOLS:
        items.append(types.CompletionItem(label=sym, kind=types.CompletionItemKind.Variable))
    for name in sorted(names):
        items.append(types.CompletionItem(label=name, kind=types.CompletionItemKind.Variable))
    return items


def _get_definition(
    text: str, line: int, char: int, uri: str
) -> types.Location | None:
    """Return the declaration location of the symbol under the cursor, or None."""
    source_lines = text.splitlines()
    if line >= len(source_lines):
        return None
    src_line = source_lines[line]
    # Walk left/right from cursor to extract the identifier under it
    start = char
    while start > 0 and (src_line[start - 1].isalnum() or src_line[start - 1] == "_"):
        start -= 1
    end = char
    while end < len(src_line) and (src_line[end].isalnum() or src_line[end] == "_"):
        end += 1
    word = src_line[start:end]
    if not word:
        return None

    try:
        models = parse(text)
    except EpiError:
        return None

    for model in models:
        for comp in model.compartments:
            if comp.name == word:
                return _make_location(uri, comp.line, comp.column)
        for const in model.constants:
            if const.name == word:
                return _make_location(uri, const.line, const.column)
        for obs in model.observables:
            if obs.name == word:
                return _make_location(uri, obs.line, obs.column)
    return None


def _make_location(uri: str, line: int, column: int) -> types.Location:
    ln = max(line - 1, 0)
    col = max(column - 1, 0)
    return types.Location(
        uri=uri,
        range=types.Range(
            start=types.Position(line=ln, character=col),
            end=types.Position(line=ln, character=col + 1),
        ),
    )


# ---------------------------------------------------------------------------
# pygls server
# ---------------------------------------------------------------------------

_server = LanguageServer("epidsl", "0.1.0")


@_server.feature(types.TEXT_DOCUMENT_DID_OPEN)
def _did_open(ls: LanguageServer, params: types.DidOpenTextDocumentParams) -> None:
    doc = params.text_document
    ls.publish_diagnostics(doc.uri, _check_document(doc.text))


@_server.feature(types.TEXT_DOCUMENT_DID_CHANGE)
def _did_change(ls: LanguageServer, params: types.DidChangeTextDocumentParams) -> None:
    uri = params.text_document.uri
    doc = ls.workspace.get_text_document(uri)
    ls.publish_diagnostics(uri, _check_document(doc.source))


@_server.feature(
    types.TEXT_DOCUMENT_COMPLETION,
    types.CompletionOptions(trigger_characters=[":", " ", "."]),
)
def _completion(
    ls: LanguageServer, params: types.CompletionParams
) -> types.CompletionList:
    pos = params.position
    uri = params.text_document.uri
    text = ls.workspace.get_text_document(uri).source
    items = _get_completions(text, pos.line, pos.character)
    return types.CompletionList(is_incomplete=False, items=items)


@_server.feature(types.TEXT_DOCUMENT_DEFINITION)
def _definition(
    ls: LanguageServer, params: types.DefinitionParams
) -> types.Location | None:
    pos = params.position
    uri = params.text_document.uri
    text = ls.workspace.get_text_document(uri).source
    return _get_definition(text, pos.line, pos.character, uri)


def make_server() -> LanguageServer:
    """Return the configured EpiDSL language server."""
    return _server
