from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

pytestmark = pytest.mark.skipif(
    os.environ.get("SATMAP_LIVE_TESTS") != "1",
    reason="set SATMAP_LIVE_TESTS=1 to hit the live Planetary Computer STAC API",
)


def test_live_naip_index_baltimore(tmp_path: Path) -> None:
    from satmap_dataset.config import IndexConfig
    from satmap_dataset.providers.naip import NaipProvider

    cfg = IndexConfig(
        year_start=2018,
        year_end=2023,
        bbox="-76.6657,39.2648,-76.6478,39.2724",
        srs="EPSG:4326",
        provider="naip",
        min_years=2,
        output_json=tmp_path / "index.json",
        year_availability_output_json=tmp_path / "avail.json",
    )
    code, path = NaipProvider().index(cfg)
    assert code == 0, path.read_text(encoding="utf-8")
