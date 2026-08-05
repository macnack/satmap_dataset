"""Capture logging output into studio job state."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from satmap_dataset.studio.jobs import Job


class JobLogHandler(logging.Handler):
    """Append formatted log records to a running studio Job."""

    def __init__(self, job: Job, *, max_lines: int = 300) -> None:
        super().__init__()
        self._job = job
        self._max_lines = max_lines
        self.setFormatter(
            logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
        except Exception:
            line = record.getMessage()
        self._job.append_log(line)
