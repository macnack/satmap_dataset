from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.config import IndexConfig
from satmap_dataset.providers.lroc_nac import provider as lroc_provider
from satmap_dataset.providers.lroc_nac.ode import OdeProduct


def _products() -> list[OdeProduct]:
    return [
        OdeProduct("M101013931LC", "2009-09-15T12:00:00", 2009, 42.5, 1.2, 0.5,
                   (30.60, 20.05, 30.72, 20.30), "https://pds.example/a.IMG", 51200000),
        OdeProduct("M198273648LC", "2012-04-03T08:15:00", 2012, 44.1, None, 0.52,
                   (30.61, 20.06, 30.73, 20.31), "https://pds.example/b.IMG", 49000000),
    ]


def test_index_builds_multitemporal_manifest(tmp_path, monkeypatch) -> None:
    async def fake_search(options, **kwargs):
        return _products()

    monkeypatch.setattr(lroc_provider.ode, "search_products", fake_search)

    out = tmp_path / "index.json"
    avail = tmp_path / "avail.json"
    cfg = IndexConfig(
        year_start=2009, year_end=2026,
        bbox="30.6,20.0,30.9,20.35", srs="IAU_2015:30100",
        provider="lroc_nac", min_years=2,
        output_json=out, year_availability_output_json=avail,
    )

    code, path = lroc_provider.LrocNacProvider().index(cfg)

    assert code == 0
    assert path == out
    manifest = json.loads(out.read_text())
    assert manifest["provider"] == "lroc_nac"
    assert manifest["years_included"] == [2009, 2012]
    assert manifest["tile_sources_by_year"]["2009"]["M101013931LC"] == "https://pds.example/a.IMG"
    assert avail.exists()


def test_index_fails_when_below_min_years(tmp_path, monkeypatch) -> None:
    async def fake_search(options, **kwargs):
        return _products()[:1]  # only 2009

    monkeypatch.setattr(lroc_provider.ode, "search_products", fake_search)

    cfg = IndexConfig(
        year_start=2009, year_end=2026,
        bbox="30.6,20.0,30.9,20.35", srs="IAU_2015:30100",
        provider="lroc_nac", min_years=2,
        output_json=tmp_path / "i.json",
        year_availability_output_json=tmp_path / "a.json",
    )
    code, _ = lroc_provider.LrocNacProvider().index(cfg)
    assert code == 1
