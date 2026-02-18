from __future__ import annotations

import logging

from satmap_dataset.logging_utils import _ColorFormatter


def _record(name: str, level: int, message: str) -> logging.LogRecord:
    return logging.LogRecord(
        name=name,
        level=level,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )


def test_httpx_400_uses_warning_color() -> None:
    formatter = _ColorFormatter(fmt="%(message)s", use_color=True)
    output = formatter.format(
        _record(
            "httpx",
            logging.INFO,
            'HTTP Request: GET https://example.test "HTTP/1.1 400 Bad Request"',
        )
    )
    assert output.startswith("\033[33m")
    assert output.endswith("\033[0m")


def test_httpx_500_uses_error_color() -> None:
    formatter = _ColorFormatter(fmt="%(message)s", use_color=True)
    output = formatter.format(
        _record(
            "httpx",
            logging.INFO,
            'HTTP Request: GET https://example.test "HTTP/1.1 500 Internal Server Error"',
        )
    )
    assert output.startswith("\033[31m")
    assert output.endswith("\033[0m")


def test_httpx_200_stays_info_color() -> None:
    formatter = _ColorFormatter(fmt="%(message)s", use_color=True)
    output = formatter.format(
        _record(
            "httpx",
            logging.INFO,
            'HTTP Request: GET https://example.test "HTTP/1.1 200 OK"',
        )
    )
    assert output.startswith("\033[32m")
    assert output.endswith("\033[0m")

