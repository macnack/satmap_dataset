from __future__ import annotations

import logging
import re
import sys


class _ColorFormatter(logging.Formatter):
    RESET = "\033[0m"
    _HTTP_STATUS_PATTERN = re.compile(r"\bHTTP/\d(?:\.\d)?\s+(\d{3})\b")
    COLORS = {
        logging.DEBUG: "\033[36m",     # cyan
        logging.INFO: "\033[32m",      # green
        logging.WARNING: "\033[33m",   # yellow
        logging.ERROR: "\033[31m",     # red
        logging.CRITICAL: "\033[31;1m" # bold red
    }

    def __init__(self, fmt: str, use_color: bool) -> None:
        super().__init__(fmt=fmt)
        self.use_color = use_color

    def _httpx_status_override_color(self, record: logging.LogRecord) -> str | None:
        if record.name != "httpx" and not record.name.startswith("httpx."):
            return None
        match = self._HTTP_STATUS_PATTERN.search(record.getMessage())
        if not match:
            return None
        status = int(match.group(1))
        if status >= 500:
            return self.COLORS[logging.ERROR]
        if status >= 400:
            return self.COLORS[logging.WARNING]
        return None

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        if not self.use_color:
            return message
        color = self._httpx_status_override_color(record) or self.COLORS.get(record.levelno)
        if not color:
            return message
        return f"{color}{message}{self.RESET}"


def configure_logging(level: str = "INFO") -> logging.Logger:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Avoid duplicated handlers when CLI re-enters configure.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler()
    formatter = _ColorFormatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        use_color=sys.stderr.isatty(),
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)

    return logging.getLogger("satmap_dataset")
