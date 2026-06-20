from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.models import TileAcquisitionMetadata, YearGsdSummary
from satmap_dataset.pipeline import index_builder


def test_gsd_key_canonicalizes():
    assert index_builder._gsd_key(0.05) == "0.05"
    assert index_builder._gsd_key(0.050) == "0.05"
    assert index_builder._gsd_key(0.25) == "0.25"


def test_summarize_mixed_year():
    tiles = {
        2024: {
            "a": TileAcquisitionMetadata(gsd=0.05),
            "b": TileAcquisitionMetadata(gsd=0.05),
            "c": TileAcquisitionMetadata(gsd=0.25),
        }
    }
    summary = index_builder._summarize_gsd_by_year(tiles)
    assert summary[2024].histogram == {"0.05": 2, "0.25": 1}
    assert summary[2024].finest == 0.05
    assert summary[2024].coarsest == 0.25


def test_summarize_all_none():
    tiles = {2014: {"a": TileAcquisitionMetadata(gsd=None)}}
    summary = index_builder._summarize_gsd_by_year(tiles)
    assert summary[2014].histogram == {}
    assert summary[2014].finest is None
    assert summary[2014].coarsest is None
