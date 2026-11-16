"""Spreading independent work over processes, without changing what it produces.

Simulations do not share state, so they can run side by side. Everything that decides
a result — the seeds, the order of the pieces — is fixed before the work is handed
out, so ``workers`` changes how long a calculation takes and nothing else.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from types import TracebackType
from typing import TypeVar

Job = TypeVar("Job")
Result = TypeVar("Result")


def resolve_workers(workers: int) -> int:
    """Turn ``workers`` into a number of processes; 0 means one per processor."""
    if workers < 0:
        raise ValueError("workers must be 0 (one per processor) or more")
    return workers or (os.cpu_count() or 1)


def split(items: Sequence[Job], parts: int) -> list[list[Job]]:
    """Cut ``items`` into at most ``parts`` consecutive pieces of nearly equal size."""
    if not items:
        return []
    parts = max(1, min(parts, len(items)))
    size, extra = divmod(len(items), parts)
    pieces: list[list[Job]] = []
    start = 0
    for i in range(parts):
        stop = start + size + (1 if i < extra else 0)
        pieces.append(list(items[start:stop]))
        start = stop
    return pieces


class Pool:
    """Worker processes that stay alive for a whole calculation.

    Starting them costs about a second, since each one loads NumPy and SciPy afresh,
    so a calculation opens one pool and keeps it for every batch it hands out. With a
    single worker nothing is forked at all.

    Use it as a context manager::

        with Pool(workers) as pool:
            measured = pool.map(_measure, jobs)
    """

    def __init__(self, workers: int) -> None:
        self.workers = resolve_workers(workers)
        self._running: ProcessPoolExecutor | None = None
        self._open = False

    def __enter__(self) -> Pool:
        self._open = True
        if self.workers > 1:
            self._running = ProcessPoolExecutor(max_workers=self.workers)
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._open = False
        if self._running is not None:
            self._running.shutdown(cancel_futures=True)
            self._running = None

    def map(self, function: Callable[[Job], list[Result]], jobs: Sequence[Job]) -> list[Result]:
        """Apply ``function`` to each job and join the results, in the order of ``jobs``.

        ``function`` must be importable by name, since a worker looks it up there.
        """
        if not self._open:
            raise RuntimeError("a Pool only runs work inside its 'with' block")
        if self._running is None or len(jobs) <= 1:
            return [item for job in jobs for item in function(job)]
        return [item for produced in self._running.map(function, jobs) for item in produced]
