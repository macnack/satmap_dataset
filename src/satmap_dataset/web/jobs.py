"""Background job runner for satmap-web pipeline tasks."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


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
    progress_label: str = ""
    logs: list[str] = field(default_factory=list)


@dataclass
class Job:
    id: str
    name: str
    location_id: str
    state: JobState = field(default_factory=JobState)
    _thread: threading.Thread | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def append_log(self, line: str, *, max_lines: int = 200) -> None:
        with self._lock:
            self.state.logs.append(line)
            if len(self.state.logs) > max_lines:
                self.state.logs = self.state.logs[-max_lines:]

    def set_message(self, message: str) -> None:
        with self._lock:
            self.state.message = message
            self.state.progress_label = message

    def start(self, fn: Callable[[], tuple[int, Any]]) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError(f"Job {self.id} is already running")

        def runner() -> None:
            self.state.status = JobStatus.RUNNING
            self.set_message(f"Starting {self.name}…")
            try:
                self.append_log(f"Job {self.name} started")
                code, artifact = fn()
                self.state.exit_code = code
                self.state.artifact_path = str(artifact)
                if code == 0:
                    self.state.status = JobStatus.SUCCESS
                    self.set_message(f"{self.name} finished successfully.")
                    self.append_log(f"Job {self.name} succeeded: {artifact}")
                else:
                    self.state.status = JobStatus.FAILED
                    self.state.error = f"exit_code={code}"
                    self.set_message(f"{self.name} failed (exit {code}).")
                    self.append_log(f"Job {self.name} failed with exit {code}")
            except Exception as exc:  # noqa: BLE001 — surface to UI
                self.state.status = JobStatus.FAILED
                self.state.error = str(exc)
                self.set_message(f"{self.name} crashed: {exc}")
                self.append_log(f"Job {self.name} error: {exc}")

        self._thread = threading.Thread(target=runner, name=f"web-job-{self.id}", daemon=True)
        self._thread.start()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "id": self.id,
                "name": self.name,
                "location_id": self.location_id,
                "status": self.state.status.value,
                "message": self.state.message,
                "exit_code": self.state.exit_code,
                "artifact_path": self.state.artifact_path,
                "error": self.state.error,
                "progress_label": self.state.progress_label,
                "logs": list(self.state.logs[-40:]),
            }


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, *, name: str, location_id: str) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], name=name, location_id=location_id)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def active_for_location(self, location_id: str) -> Job | None:
        with self._lock:
            for job in reversed(list(self._jobs.values())):
                if job.location_id != location_id:
                    continue
                if job.state.status in {JobStatus.PENDING, JobStatus.RUNNING}:
                    return job
            return None
