from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.providers.lroc_nac.crs import normalize_bbox_to_ode


def test_geographic_bbox_passthrough_reorders_to_ode() -> None:
    # bbox is xmin,ymin,xmax,ymax = westlon,minlat,eastlon,maxlat
    west, east, minlat, maxlat = normalize_bbox_to_ode(
        "30.60,20.00,30.90,20.35", "IAU_2015:30100"
    )
    assert (west, east, minlat, maxlat) == (30.60, 20.00, 30.90, 20.35)


def test_rejects_non_lunar_crs() -> None:
    with pytest.raises(ValueError, match="lunar"):
        normalize_bbox_to_ode("30.6,20.0,30.9,20.35", "EPSG:2180")
