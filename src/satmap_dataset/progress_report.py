"""Optional progress reporting for long pipeline stages (studio UI, CLI)."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Protocol


class ProgressReporter(Protocol):
    def log(self, message: str) -> None: ...

    def progress(self, current: int, total: int, label: str) -> None: ...


_reporter: ContextVar[ProgressReporter | None] = ContextVar("satmap_progress_reporter", default=None)


def get_progress_reporter() -> ProgressReporter | None:
    return _reporter.get()


def set_progress_reporter(reporter: ProgressReporter | None) -> None:
    _reporter.set(reporter)


def report_log(message: str) -> None:
    reporter = get_progress_reporter()
    if reporter is not None:
        reporter.log(message)


def report_progress(current: int, total: int, label: str) -> None:
    reporter = get_progress_reporter()
    if reporter is not None:
        reporter.progress(current, total, label)
