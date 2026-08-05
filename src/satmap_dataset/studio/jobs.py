"""Background job runner for long studio pipeline tasks."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from satmap_dataset.progress_report import set_progress_reporter
from satmap_dataset.studio.log_handler import JobLogHandler


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class JobState:
    status: JobStatus = JobStatus.PENDING
    message: str = ""
    exit_code: int | None = None
    artifact_path: str | None = None
    error: str | None = None
    result: Any = None
    progress_current: int = 0
    progress_total: int = 0
    progress_label: str = ""
    logs: list[str] = field(default_factory=list)


class _JobProgressReporter:
    def __init__(self, job: Job) -> None:
        self._job = job

    def log(self, message: str) -> None:
        self._job.append_log(message)

    def progress(self, current: int, total: int, label: str) -> None:
        self._job.set_progress(current, total, label)


@dataclass
class Job:
    name: str
    state: JobState = field(default_factory=JobState)
    _thread: threading.Thread | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _log_handler: JobLogHandler | None = field(default=None, repr=False)

    def append_log(self, line: str, *, max_lines: int = 300) -> None:
        migrate_job_state(self.state)
        with self._lock:
            self.state.logs.append(line)
            if len(self.state.logs) > max_lines:
                self.state.logs = self.state.logs[-max_lines:]

    def set_progress(self, current: int, total: int, label: str) -> None:
        migrate_job_state(self.state)
        with self._lock:
            self.state.progress_current = max(0, current)
            self.state.progress_total = max(total, 1)
            self.state.progress_label = label
            self.state.message = label

    def start(self, fn: Callable[[], tuple[int, Any]], *, on_message: Callable[[str], None] | None = None) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError(f"Job {self.name!r} is already running")

        def runner() -> None:
            self.state.status = JobStatus.RUNNING
            self.set_progress(0, 1, f"Starting {self.name}…")
            if on_message:
                on_message(f"Running {self.name}...")
            reporter = _JobProgressReporter(self)
            set_progress_reporter(reporter)
            handler = JobLogHandler(self)
            handler.setLevel(logging.DEBUG)
            root = logging.getLogger()
            root.addHandler(handler)
            self._log_handler = handler
            try:
                self.append_log(f"Job {self.name} started")
                code, artifact = fn()
                self.state.exit_code = code
                self.state.artifact_path = str(artifact)
                self.state.result = artifact
                if code == 0:
                    self.state.status = JobStatus.SUCCESS
                    self.state.message = f"{self.name} finished successfully."
                    self.set_progress(
                        self.state.progress_total,
                        self.state.progress_total,
                        f"{self.name} finished successfully.",
                    )
                else:
                    self.state.status = JobStatus.FAILED
                    self.state.message = f"{self.name} failed (exit code {code})."
                    self.set_progress(
                        self.state.progress_current,
                        self.state.progress_total,
                        f"{self.name} failed (exit code {code}).",
                    )
                self.append_log(self.state.message)
            except Exception as exc:
                self.state.status = JobStatus.FAILED
                self.state.error = str(exc)
                self.state.message = f"{self.name} error: {exc}"
                self.append_log(self.state.message)
            finally:
                root.removeHandler(handler)
                self._log_handler = None
                set_progress_reporter(None)

        self._thread = threading.Thread(target=runner, daemon=True)
        self._thread.start()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


def migrate_job_state(state: JobState) -> JobState:
    """Backfill fields added after a Job was stored in Streamlit session_state."""
    if not hasattr(state, "progress_current"):
        state.progress_current = 0
    if not hasattr(state, "progress_total"):
        state.progress_total = 0
    if not hasattr(state, "progress_label"):
        state.progress_label = ""
    if not hasattr(state, "logs"):
        state.logs = []
    return state


def migrate_job(job: Job) -> Job:
    migrate_job_state(job.state)
    return job


__all__ = ["Job", "JobState", "JobStatus", "migrate_job", "migrate_job_state"]
