from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.config import (
    DownloadConfig,
    IndexConfig,
    PROVIDER_GEOPORTAL,
    PROVIDER_LANTMATERIET,
    RunConfig,
)
from satmap_dataset.models import IndexManifest, DatasetManifest


def test_run_config_defaults_to_geoportal_provider() -> None:
    cfg = RunConfig(
        year_start=2015,
        year_end=2016,
        bbox="0,0,1,1",
    )
    assert cfg.provider == PROVIDER_GEOPORTAL
    assert cfg.provider_options == {}


def test_run_config_accepts_lantmateriet_provider() -> None:
    cfg = RunConfig(
        year_start=2010,
        year_end=2024,
        bbox="536194,6426213,538194,6428213",
        srs="EPSG:3006",
        target_srs="EPSG:3006",
        provider=PROVIDER_LANTMATERIET,
        provider_options={"year_policy": "nearest_before"},
    )
    assert cfg.provider == PROVIDER_LANTMATERIET
    assert cfg.provider_options["year_policy"] == "nearest_before"


def test_run_config_rejects_unknown_provider() -> None:
    with pytest.raises(ValidationError):
        RunConfig(
            year_start=2015,
            year_end=2016,
            bbox="0,0,1,1",
            provider="opendata",
        )


def test_index_config_provider_round_trips() -> None:
    cfg = IndexConfig(
        year_start=2015,
        year_end=2016,
        bbox="0,0,1,1",
        provider=PROVIDER_LANTMATERIET,
        provider_options={"stac_url": "https://example.org/stac/search"},
    )
    payload = cfg.model_dump_json()
    restored = IndexConfig.model_validate_json(payload)
    assert restored.provider == PROVIDER_LANTMATERIET
    assert restored.provider_options["stac_url"] == "https://example.org/stac/search"


def test_download_config_for_lantmateriet_does_not_require_bbox_in_hybrid_mode() -> None:
    # The geoportal-only "bbox required for hybrid/wms_tiled" check must not
    # fire for non-geoportal providers.
    cfg = DownloadConfig(provider=PROVIDER_LANTMATERIET, bbox=None)
    assert cfg.provider == PROVIDER_LANTMATERIET
    assert cfg.bbox is None


def test_index_manifest_serialization_includes_provider() -> None:
    manifest = IndexManifest(
        provider=PROVIDER_LANTMATERIET,
        year_start=2015,
        year_end=2016,
        bbox="0,0,1,1",
        srs="EPSG:3006",
        years_requested=[2015, 2016],
        year_statuses=[],
        years_available_wfs=[],
        years_included=[],
        passed=False,
    )
    restored = IndexManifest.model_validate_json(manifest.model_dump_json())
    assert restored.provider == PROVIDER_LANTMATERIET


def test_dataset_manifest_supports_stac_mode_and_arbitrary_source_keys() -> None:
    manifest = DatasetManifest(
        provider=PROVIDER_LANTMATERIET,
        stage="download",
        mode="stac",
        years_source_map={2015: "stac", 2016: "wms_fallback"},
    )
    restored = DatasetManifest.model_validate_json(manifest.model_dump_json())
    assert restored.provider == PROVIDER_LANTMATERIET
    assert restored.mode == "stac"
    assert restored.years_source_map == {2015: "stac", 2016: "wms_fallback"}


def test_existing_geoportal_index_manifest_json_still_loads() -> None:
    # Backwards compatibility: pre-provider manifests must still parse.
    legacy_payload = (
        '{"kind":"index_manifest","year_start":2015,"year_end":2016,'
        '"bbox":"0,0,1,1","srs":"EPSG:2180","years_requested":[2015,2016],'
        '"year_statuses":[],"years_available_wfs":[],"years_included":[],'
        '"passed":true}'
    )
    restored = IndexManifest.model_validate_json(legacy_payload)
    assert restored.provider == "geoportal"
    assert restored.run_parameters == {}
