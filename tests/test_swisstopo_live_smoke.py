from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

pytestmark = pytest.mark.skipif(
    os.environ.get("SATMAP_LIVE_TESTS") != "1",
    reason="set SATMAP_LIVE_TESTS=1 to hit live swisstopo WMTS/WMS",
)


def test_live_swisstopo_index_bern_historical_depth(tmp_path: Path) -> None:
    from satmap_dataset.config import IndexConfig
    from satmap_dataset.providers.swisstopo import SwisstopoProvider

    cfg = IndexConfig(
        year_start=1926,
        year_end=2025,
        bbox="2600000,1199000,2602000,1201000",
        srs="EPSG:2056",
        provider="swisstopo",
        min_years=50,
        output_json=tmp_path / "index.json",
        year_availability_output_json=tmp_path / "avail.json",
    )
    code, path = SwisstopoProvider().index(cfg)
    assert code == 0, path.read_text(encoding="utf-8")
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    # Live WMTS currently advertises ~99 YYYY values from 1926–2025.
    assert len(payload["years_included"]) >= 90
    assert payload["years_included"][0] <= 1930
    assert payload["years_included"][-1] >= 2020
