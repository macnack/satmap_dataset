"""Stable fingerprints of effective run inputs for skip / reuse gates.

Kept in ``io`` (neutral of CLI/pipeline) so index/download reuse fingerprinting
can share the same canonical JSON + digest helpers later without depending on
orchestration code.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from satmap_dataset.config import RunConfig

# Bump when the hashed field set or normalization rules change.
_RUN_CONFIG_HASH_VERSION = 1

# Output-affecting RunConfig fields (reuse predicates + validation expectations).
# Operational knobs (concurrency, retries, timeouts, sleep jitter, overwrite)
# and deprecated experimental_wfs_swap_bbox_axes are intentionally omitted.
_RUN_CONFIG_HASH_FIELDS: tuple[str, ...] = (
    "year_start",
    "year_end",
    "bbox",
    "srs",
    "strict_years",
    "min_years",
    "mode",
    "profile",
    "px_per_meter",
    "wms_fallback_missing_years",
    "force_wms_years",
    "disable_color_norm",
    "target_width",
    "target_height",
    "auto_size_from_bbox",
    "pixel_profile",
    "download_root",
    "render_root",
    "target_bbox",
    "target_srs",
    "resample_method",
    "tile_size",
    "compression",
    "overview_levels",
    "experimental_force_srgb_from_ycbcr",
    "experimental_per_year_color_norm",
    "provider",
    "provider_options",
)


def canonicalize(value: Any) -> Any:
    """Normalize a value for stable JSON serialization."""
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): canonicalize(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [canonicalize(item) for item in value]
    if isinstance(value, set):
        return sorted(canonicalize(item) for item in value)
    return value


def canonical_json_dumps(value: Any) -> str:
    """Serialize ``value`` with sorted keys and compact separators."""
    return json.dumps(
        canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def stable_hash(value: Any) -> str:
    """SHA-256 hex digest of the canonical JSON form of ``value``."""
    encoded = canonical_json_dumps(value).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_config_fingerprint_payload(config: RunConfig) -> dict[str, Any]:
    """Return the versioned payload hashed by :func:`run_config_hash`."""
    dumped = config.model_dump(mode="json")
    payload: dict[str, Any] = {"v": _RUN_CONFIG_HASH_VERSION}
    for field in _RUN_CONFIG_HASH_FIELDS:
        payload[field] = dumped.get(field)
    return payload


def run_config_hash(config: RunConfig) -> str:
    """Stable fingerprint of effective RunConfig inputs that affect outputs."""
    return stable_hash(run_config_fingerprint_payload(config))
