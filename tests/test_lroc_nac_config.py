from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.config import IndexConfig


def test_lroc_nac_accepts_lunar_crs() -> None:
    cfg = IndexConfig(
        year_start=2009, year_end=2026,
        bbox="30.6,20.0,30.9,20.35", srs="IAU_2015:30100",
        provider="lroc_nac",
    )
    assert cfg.provider == "lroc_nac"


def test_lroc_nac_rejects_earth_crs() -> None:
    with pytest.raises(ValueError, match="lunar"):
        IndexConfig(
            year_start=2009, year_end=2026,
            bbox="30.6,20.0,30.9,20.35", srs="EPSG:2180",
            provider="lroc_nac",
        )
