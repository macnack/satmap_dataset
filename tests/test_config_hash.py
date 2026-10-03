"""Stable RunConfig fingerprint used by run-all-location-json skip."""

from __future__ import annotations

import json
from pathlib import Path

from satmap_dataset.config import RunConfig, ValidateConfig
from satmap_dataset.io.config_hash import (
    canonicalize,
    run_config_fingerprint_payload,
    run_config_hash,
    stable_hash,
)
from satmap_dataset.models import LayerManifest, ValidationReport
from satmap_dataset.pipeline import validator


def _base_run_config(**overrides) -> RunConfig:
    payload = {
        "year_start": 2015,
        "year_end": 2016,
        "bbox": "100,200,300,400",
        "srs": "EPSG:2180",
        "mode": "hybrid",
        "profile": "train",
        "provider": "geoportal",
    }
    payload.update(overrides)
    return RunConfig.model_validate(payload)


def test_run_config_hash_stable_for_equivalent_provider_options_key_order() -> None:
    a = _base_run_config(provider_options={"b": 2, "a": {"y": 1, "x": 0}})
    b = _base_run_config(provider_options={"a": {"x": 0, "y": 1}, "b": 2})
    assert run_config_hash(a) == run_config_hash(b)


def test_run_config_hash_changes_when_year_range_changes() -> None:
    a = _base_run_config(year_end=2016)
    b = _base_run_config(year_end=2017)
    assert run_config_hash(a) != run_config_hash(b)


def test_run_config_hash_changes_when_provider_options_change() -> None:
    a = _base_run_config(provider_options={"product_type": "CDRNAC4"})
    b = _base_run_config(provider_options={"product_type": "OTHER"})
    assert run_config_hash(a) != run_config_hash(b)


def test_run_config_hash_ignores_operational_retry_knobs() -> None:
    a = _base_run_config(concurrency=4, retries=2, timeout=30.0)
    b = _base_run_config(concurrency=8, retries=9, timeout=120.0)
    assert run_config_hash(a) == run_config_hash(b)


def test_fingerprint_payload_includes_version_and_core_fields() -> None:
    config = _base_run_config()
    payload = run_config_fingerprint_payload(config)
    assert payload["v"] == 1
    assert payload["bbox"] == "100,200,300,400"
    assert payload["provider"] == "geoportal"
    assert "concurrency" not in payload


def test_canonicalize_sorts_mapping_keys() -> None:
    assert canonicalize({"b": 1, "a": 2}) == {"a": 2, "b": 1}
    assert stable_hash({"b": 1, "a": 2}) == stable_hash({"a": 2, "b": 1})


def test_validator_persists_config_hash(tmp_path: Path) -> None:
    manifest = LayerManifest(
        layer="geoportal_rgb",
        role="rgb",
        stage="render",
        provider="geoportal",
        mode="hybrid",
        profile="train",
        years_requested=[2015],
        years_included=[2015],
        assets=[],
        passed=True,
        pixel_profile="RGB_U8",
    )
    manifest_path = tmp_path / "dataset_manifest_render.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    output = tmp_path / "validation_report.json"
    config = ValidateConfig(
        dataset_manifest=manifest_path,
        requested_years=[2015],
        output_json=output,
        config_hash="abc123",
    )
    code, path = validator.run(config)
    assert code == 1  # no assets → fail, but hash still persisted
    report = ValidationReport.model_validate_json(path.read_text(encoding="utf-8"))
    assert report.config_hash == "abc123"


def test_validation_report_defaults_missing_config_hash() -> None:
    payload = {
        "requested_years": [2015],
        "years_included": [2015],
        "missing_years": [],
        "passed": True,
    }
    report = ValidationReport.model_validate(payload)
    assert report.config_hash is None
    # Round-trip without the field still loads.
    raw = json.dumps(payload)
    assert ValidationReport.model_validate_json(raw).config_hash is None
