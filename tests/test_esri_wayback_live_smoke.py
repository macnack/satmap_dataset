"""Live Esri Wayback smoke (opt-in). Read docs/DATA_LICENSING.md before enabling."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

pytestmark = pytest.mark.skipif(
    os.environ.get("SATMAP_LIVE_TESTS") != "1",
    reason="set SATMAP_LIVE_TESTS=1 to hit the live Esri Wayback service",
)

# ~300 m square in Warsaw Wola, EPSG:2180.
BBOX_2180 = "635300,487000,635600,487300"


def test_live_wayback_index_and_single_year_download(tmp_path: Path) -> None:
    from satmap_dataset.config import DownloadConfig, IndexConfig
    from satmap_dataset.pipeline import render
    from satmap_dataset.providers import get_provider

    provider = get_provider("esri_wayback")
    options = {"probe": "center", "probe_zoom": 17}
    code, index_path = provider.index(
        IndexConfig(
            year_start=2010,
            year_end=2026,
            bbox=BBOX_2180,
            srs="EPSG:2180",
            provider="esri_wayback",
            min_years=2,
            output_json=tmp_path / "index_manifest.json",
            year_availability_output_json=tmp_path / "year_availability_report.json",
            provider_options=options,
        )
    )
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert code == 0, payload["errors"]
    meta = payload["provider_metadata"]
    assert meta["releases_total"] >= 150
    assert meta["dedupe_mode_effective"] == "tilemap"
    assert 2 <= meta["distinct_versions"] < meta["releases_total"]
    assert all(v["capture_date_source"] == "metadata" for v in meta["versions"])

    latest = max(payload["years_included"])
    index = dict(payload)
    for key in ("tile_sources_by_year", "tile_bboxes_by_year", "tile_acquisition_by_year"):
        index[key] = {str(latest): payload[key][str(latest)]}
    index["years_included"] = [latest]
    one_year = tmp_path / "index_one_year.json"
    one_year.write_text(json.dumps(index), encoding="utf-8")
    code, dl_path = provider.download(
        DownloadConfig(
            index_manifest=one_year,
            download_root=tmp_path / "downloads",
            output_json=tmp_path / "dataset_manifest_download.json",
            provider="esri_wayback",
            mode="wms_tiled",
            bbox=BBOX_2180,
            srs="EPSG:2180",
            concurrency=2,
            provider_options={**options, "zoom": 16},
        )
    )
    download = json.loads(dl_path.read_text(encoding="utf-8"))
    assert code == 0, download["notes"]
    asset = Path(download["assets"][0])
    assert render._read_source_epsg(asset) == 3857
