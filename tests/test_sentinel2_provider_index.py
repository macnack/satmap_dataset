from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.config import IndexConfig
from satmap_dataset.providers import get_provider
from satmap_dataset.providers.sentinel2 import provider as provider_module


FIXTURES = ROOT / "tests" / "fixtures" / "sentinel2"


def _patch_search_with_fixture(monkeypatch, fixture_name: str) -> None:
    payload = json.loads((FIXTURES / fixture_name).read_text(encoding="utf-8"))

    async def fake_search(*_args, **_kwargs):
        from satmap_dataset.providers.lantmateriet.stac import parse_stac_features

        return parse_stac_features(payload)

    monkeypatch.setattr(provider_module.stac, "search_features", fake_search)


def test_get_provider_returns_sentinel2() -> None:
    p = get_provider("sentinel2")
    assert p.name == "sentinel2"


def test_index_picks_closest_to_feb15_per_year(monkeypatch, tmp_path: Path) -> None:
    _patch_search_with_fixture(monkeypatch, "stac_search_response_multi_year.json")

    config = IndexConfig(
        year_start=2022,
        year_end=2024,
        bbox="15.55,57.95,15.70,58.02",
        srs="EPSG:4326",
        strict_years=False,
        min_years=1,
        output_json=tmp_path / "index_manifest.json",
        year_availability_output_json=tmp_path / "year_availability_report.json",
        provider="sentinel2",
    )
    exit_code, manifest_path = get_provider("sentinel2").index(config)
    assert exit_code == 0
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["provider"] == "sentinel2"
    assert payload["passed"] is True
    assert sorted(payload["years_included"]) == [2022, 2023, 2024]
    # 2022 has both a clear scene (4.5%) and a cloudy one (60%); clear wins.
    sources_2022 = payload["tile_sources_by_year"]["2022"]
    assert any("2022_02_13" in url for url in sources_2022.values())
    # 2024 has Jan 18 (28 days from Feb 15) and Feb 17 (2 days). Feb 17 wins.
    sources_2024 = payload["tile_sources_by_year"]["2024"]
    assert any("2024_02_17" in url for url in sources_2024.values())


def test_index_drops_year_when_all_scenes_above_cloud_threshold(monkeypatch, tmp_path: Path) -> None:
    _patch_search_with_fixture(monkeypatch, "stac_search_response_multi_year.json")

    config = IndexConfig(
        year_start=2022,
        year_end=2022,
        bbox="15.55,57.95,15.70,58.02",
        srs="EPSG:4326",
        strict_years=False,
        min_years=1,
        output_json=tmp_path / "index_manifest.json",
        year_availability_output_json=tmp_path / "year_availability_report.json",
        provider="sentinel2",
        provider_options={"max_cloud_cover_pct": 1.0},
    )
    exit_code, manifest_path = get_provider("sentinel2").index(config)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Both 2022 candidates exceed 1% cloud_cover -> year excluded.
    assert payload["years_included"] == []
    assert exit_code == 1


def test_index_records_chosen_utm_zones(monkeypatch, tmp_path: Path) -> None:
    _patch_search_with_fixture(monkeypatch, "stac_search_response_multi_year.json")

    config = IndexConfig(
        year_start=2022,
        year_end=2024,
        bbox="15.55,57.95,15.70,58.02",
        srs="EPSG:4326",
        strict_years=False,
        min_years=1,
        output_json=tmp_path / "index_manifest.json",
        year_availability_output_json=tmp_path / "year_availability_report.json",
        provider="sentinel2",
    )
    get_provider("sentinel2").index(config)
    payload = json.loads(config.output_json.read_text(encoding="utf-8"))
    assert payload["provider_metadata"]["chosen_epsgs"] == [32633]
    scenes = payload["provider_metadata"]["scenes"]
    assert {s["year"] for s in scenes} == {2022, 2023, 2024}
    for s in scenes:
        assert s["epsg"] == 32633


def test_index_against_real_earth_search_fixture(monkeypatch, tmp_path: Path) -> None:
    """Smoke test on a real Earth Search /search response shape."""
    _patch_search_with_fixture(monkeypatch, "stac_search_response_winter.json")

    config = IndexConfig(
        year_start=2024,
        year_end=2024,
        bbox="15.61,57.97,15.65,58.00",
        srs="EPSG:4326",
        strict_years=False,
        min_years=1,
        output_json=tmp_path / "index_manifest.json",
        year_availability_output_json=tmp_path / "year_availability_report.json",
        provider="sentinel2",
    )
    exit_code, manifest_path = get_provider("sentinel2").index(config)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["provider"] == "sentinel2"
    assert payload["passed"] is True
    assert payload["years_included"] == [2024]
    sources = payload["tile_sources_by_year"]["2024"]
    # Should have picked one of the two Jan items closest to Feb 15;
    # all are well within the 20% cloud threshold.
    assert len(sources) == 1
    href = next(iter(sources.values()))
    assert href.startswith("https://sentinel-cogs.s3.us-west-2.amazonaws.com/")
    assert href.endswith(".tif")
