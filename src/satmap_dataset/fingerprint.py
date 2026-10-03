"""Stable fingerprints for config/manifest reuse predicates."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def fingerprint_provider_options(options: Mapping[str, Any] | None) -> str:
    """Return a SHA-256 hex digest of canonical JSON for ``provider_options``.

    Keys are sorted recursively; values are JSON-normalized so equivalent
    option dicts compare equal across Python/process boundaries. Empty or
    missing options fingerprint as the empty object.
    """
    payload = _canonicalize(options or {})
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _canonicalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v) for v in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):
        return value
    return str(value)
