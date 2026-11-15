"""Errors raised while turning source text into models."""

from __future__ import annotations


class EpiError(Exception):
    """An error at a source position; ``str()`` gives ``line:column: kind: message``."""

    kind = "error"

    def __init__(self, message: str, line: int, column: int) -> None:
        super().__init__(f"{line}:{column}: {self.kind}: {message}")
        self.message = message
        self.line = line
        self.column = column


class EpiSyntaxError(EpiError):
    kind = "syntax error"


class CompositionError(EpiError):
    kind = "composition error"
