from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.config import IndexConfig
from satmap_dataset.providers.lantmateriet import provider as provider_module
from satmap_dataset.providers.lantmateriet.provider import LantmaterietProvider


FIXTURES = ROOT / "tests" / "fixtures" / "lantmateriet"


def _patch_search_with_fixture(monkeypatch, fixture_name: str) -> None:
    payload = json.loads((FIXTURES / fixture_name).read_text(encoding="utf-8"))

    async def fake_search(*_args, **_kwargs):
        from satmap_dataset.providers.lantmateriet.stac import parse_stac_features

        return parse_stac_features(payload)

    monkeypatch.setattr(provider_module.stac, "search_features", fake_search)


def test_lantmateriet_index_writes_manifest_from_multi_year_fixture(monkeypatch, tmp_path: Path) -> None:
    _patch_search_with_fixture(monkeypatch, "stac_search_response_multi_year.json")

    config = IndexConfig(
        year_start=2018,
        year_end=2024,
        bbox="536194.91,6426213.28,538194.91,6428213.28",
        srs="EPSG:3006",
        strict_years=False,
        min_years=1,
        output_json=tmp_path / "index_manifest.json",
        year_availability_output_json=tmp_path / "year_availability_report.json",
        provider="lantmateriet",
        provider_options={"year_policy": "exact_only"},
    )
    exit_code, output_path = LantmaterietProvider().index(config)
    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["provider"] == "lantmateriet"
    assert payload["passed"] is True
    assert sorted(payload["years_included"]) == [2018, 2020, 2024]
    assert payload["tile_sources_by_year"]["2024"]
    assert payload["years_excluded_with_reason"]["2019"].startswith("exact_only")


def test_lantmateriet_index_applies_nearest_before_policy_for_missing_years(monkeypatch, tmp_path: Path) -> None:
    _patch_search_with_fixture(monkeypatch, "stac_search_response_multi_year.json")

    config = IndexConfig(
        year_start=2019,
        year_end=2021,
        bbox="536194.91,6426213.28,538194.91,6428213.28",
        srs="EPSG:3006",
        strict_years=False,
        min_years=1,
        output_json=tmp_path / "index_manifest.json",
        year_availability_output_json=tmp_path / "year_availability_report.json",
        provider="lantmateriet",
        provider_options={"year_policy": "nearest_before"},
    )
    exit_code, output_path = LantmaterietProvider().index(config)
    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    # 2020 is exact, 2019 falls back to 2018, 2021 falls back to 2020.
    assert sorted(payload["years_included"]) == [2019, 2020, 2021]
    sources_2019 = payload["tile_sources_by_year"]["2019"]
    sources_2021 = payload["tile_sources_by_year"]["2021"]
    assert any("2018" in url for url in sources_2019.values())
    assert any("2020" in url for url in sources_2021.values())
