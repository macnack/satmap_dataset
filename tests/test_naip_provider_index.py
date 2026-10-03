from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.config import IndexConfig
from satmap_dataset.providers import get_provider
from satmap_dataset.providers.naip import provider as provider_module


FIXTURES = ROOT / "tests" / "fixtures" / "naip"


def _patch_search_with_fixture(monkeypatch, fixture_name: str) -> None:
    payload = json.loads((FIXTURES / fixture_name).read_text(encoding="utf-8"))

    async def fake_search(*_args, **_kwargs):
        from satmap_dataset.providers.lantmateriet.stac import parse_stac_features

        return parse_stac_features(payload)

    monkeypatch.setattr(provider_module.stac, "search_features", fake_search)


def test_get_provider_returns_naip() -> None:
    p = get_provider("naip")
    assert p.name == "naip"


def test_index_picks_closest_to_june15_per_year(monkeypatch, tmp_path: Path) -> None:
    _patch_search_with_fixture(monkeypatch, "stac_search_response_multi_year.json")

    config = IndexConfig(
        year_start=2015,
        year_end=2023,
        bbox="-76.6657,39.2648,-76.6478,39.2724",
        srs="EPSG:4326",
        strict_years=False,
        min_years=1,
        output_json=tmp_path / "index_manifest.json",
        year_availability_output_json=tmp_path / "year_availability_report.json",
        provider="naip",
    )
    exit_code, manifest_path = get_provider("naip").index(config)
    assert exit_code == 0
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["provider"] == "naip"
    assert payload["passed"] is True
    assert sorted(payload["years_included"]) == [2015, 2017, 2018, 2021, 2023]
    # 2023 has May 25 (closer to June 15) and Sept 10 — May wins.
    sources_2023 = payload["tile_sources_by_year"]["2023"]
    assert any("20230525" in url for url in sources_2023.values())
    # 2018 has Nov 21 (far) and June 20 (near) — June wins.
    sources_2018 = payload["tile_sources_by_year"]["2018"]
    assert any("_early.tif" in url for url in sources_2018.values())


def test_index_records_chosen_epsg_and_gsd(monkeypatch, tmp_path: Path) -> None:
    _patch_search_with_fixture(monkeypatch, "stac_search_response_multi_year.json")

    config = IndexConfig(
        year_start=2021,
        year_end=2023,
        bbox="-76.6657,39.2648,-76.6478,39.2724",
        srs="EPSG:4326",
        strict_years=False,
        min_years=1,
        output_json=tmp_path / "index_manifest.json",
        year_availability_output_json=tmp_path / "year_availability_report.json",
        provider="naip",
    )
    get_provider("naip").index(config)
    payload = json.loads(config.output_json.read_text(encoding="utf-8"))
    assert payload["provider_metadata"]["stac_host"] == "planetary_computer"
    assert payload["provider_metadata"]["chosen_epsgs"] == [26918]
    scenes = {s["year"]: s for s in payload["provider_metadata"]["scenes"]}
    assert scenes[2023]["gsd"] == 0.3
    assert scenes[2021]["gsd"] == 0.6


def test_index_warns_on_earth_search_host(monkeypatch, tmp_path: Path) -> None:
    _patch_search_with_fixture(monkeypatch, "stac_search_response_multi_year.json")

    config = IndexConfig(
        year_start=2023,
        year_end=2023,
        bbox="-76.6657,39.2648,-76.6478,39.2724",
        srs="EPSG:4326",
        strict_years=False,
        min_years=1,
        output_json=tmp_path / "index_manifest.json",
        year_availability_output_json=tmp_path / "year_availability_report.json",
        provider="naip",
        provider_options={"stac_host": "earth_search"},
    )
    exit_code, manifest_path = get_provider("naip").index(config)
    assert exit_code == 0
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["provider_metadata"]["stac_host"] == "earth_search"
    assert any("requester-pays" in w for w in payload["warnings"])


def test_index_missing_years_fail_min_years(monkeypatch, tmp_path: Path) -> None:
    _patch_search_with_fixture(monkeypatch, "stac_search_response_multi_year.json")

    config = IndexConfig(
        year_start=2019,
        year_end=2020,
        bbox="-76.6657,39.2648,-76.6478,39.2724",
        srs="EPSG:4326",
        strict_years=False,
        min_years=1,
        output_json=tmp_path / "index_manifest.json",
        year_availability_output_json=tmp_path / "year_availability_report.json",
        provider="naip",
    )
    exit_code, manifest_path = get_provider("naip").index(config)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["years_included"] == []
    assert exit_code == 1
