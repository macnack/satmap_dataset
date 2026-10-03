from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.fingerprint import fingerprint_provider_options


def test_empty_and_none_match() -> None:
    assert fingerprint_provider_options(None) == fingerprint_provider_options({})
    assert len(fingerprint_provider_options({})) == 64


def test_key_order_independent() -> None:
    a = fingerprint_provider_options({"b": 1, "a": 2})
    b = fingerprint_provider_options({"a": 2, "b": 1})
    assert a == b


def test_nested_key_order_independent() -> None:
    a = fingerprint_provider_options({"opts": {"z": True, "y": [2, 1]}})
    b = fingerprint_provider_options({"opts": {"y": [2, 1], "z": True}})
    assert a == b


def test_value_change_changes_fingerprint() -> None:
    base = fingerprint_provider_options({"max_cloud_cover_pct": 10.0})
    changed = fingerprint_provider_options({"max_cloud_cover_pct": 5.0})
    assert base != changed


def test_bool_not_coerced_to_int() -> None:
    assert fingerprint_provider_options({"flag": True}) != fingerprint_provider_options(
        {"flag": 1}
    )
