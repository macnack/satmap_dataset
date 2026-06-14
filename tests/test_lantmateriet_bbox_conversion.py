from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.providers.lantmateriet.crs import (
    transform_bbox_wgs84_to,
    transform_point,
)


def _proj_available() -> bool:
    import importlib.util
    import shutil

    if importlib.util.find_spec("pyproj") is not None:
        return True
    return shutil.which("proj") is not None


pytestmark = pytest.mark.skipif(not _proj_available(), reason="pyproj or proj CLI not available")


def test_bbox_wgs84_to_sweref99tm_kisa_is_centered() -> None:
    minx, miny, maxx, maxy = transform_bbox_wgs84_to(
        "EPSG:3006",
        center_lat=57.985,
        center_lon=15.629,
        size_meters=2000.0,
    )
    # Side length must equal the requested size.
    assert maxx - minx == pytest.approx(2000.0, abs=1.0)
    assert maxy - miny == pytest.approx(2000.0, abs=1.0)
    # Center must round-trip back close to the input lat/lon.
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    lon, lat = transform_point("EPSG:3006", "EPSG:4326", cx, cy)
    assert lat == pytest.approx(57.985, abs=1e-3)
    assert lon == pytest.approx(15.629, abs=1e-3)


def test_bbox_falls_inside_sweden_extent() -> None:
    bbox = transform_bbox_wgs84_to(
        "EPSG:3006",
        center_lat=57.985,
        center_lon=15.629,
        size_meters=2000.0,
    )
    minx, miny, maxx, maxy = bbox
    # SWEREF 99 TM coordinates for southern Sweden are around x=400-700k, y=6.3-7.0M.
    assert 200_000 < minx < 900_000
    assert 6_000_000 < miny < 7_500_000
    assert maxx > minx
    assert maxy > miny
