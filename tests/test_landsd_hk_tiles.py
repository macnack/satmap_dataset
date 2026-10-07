from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.providers.landsd_hk import tiles
from satmap_dataset.providers.landsd_hk.provider import _resolve_zoom


def test_lonlat_tile_roundtrip_hk_airport() -> None:
    # HKairport approx — same indexing LandsD WGS84 XYZ uses.
    x, y = tiles.lonlat_to_mercator(114.0430, 22.4159)
    assert tiles.mercator_to_tile(x, y, 19) == (428231, 228632)
    lon, lat = tiles.mercator_to_lonlat(x, y)
    assert lon == pytest.approx(114.0430) and lat == pytest.approx(22.4159)


def test_tile_bounds_and_resolution() -> None:
    assert tiles.resolution(0) == pytest.approx(156543.03392804097)
    b = tiles.tile_bounds(1, 0, 0)
    assert (b.min_x, b.max_y) == pytest.approx((-tiles.ORIGIN_SHIFT, tiles.ORIGIN_SHIFT))
    assert b.max_x == pytest.approx(0.0) and b.min_y == pytest.approx(0.0)


def test_tile_range_excludes_exact_edge() -> None:
    t = tiles.tile_bounds(10, 500, 300)
    rng = tiles.tile_range_for_bbox(t, 10)
    assert (rng.x_min, rng.x_max, rng.y_min, rng.y_max) == (500, 500, 300, 300)
    assert rng.count == 1


def test_zoom_for_gsd_at_hk_latitude() -> None:
    lat = 22.4
    assert tiles.ground_gsd(19, lat) == pytest.approx(
        tiles.resolution(19) * math.cos(math.radians(lat)), rel=1e-6
    )
    assert tiles.zoom_for_gsd(0.3, lat, min_zoom=15, max_zoom=19) == 19
    assert tiles.zoom_for_gsd(0.6, lat, min_zoom=15, max_zoom=19) == 18


def test_resolve_zoom_precedence() -> None:
    x0, y0 = tiles.lonlat_to_mercator(114.04, 22.41)
    x1, y1 = tiles.lonlat_to_mercator(114.05, 22.42)
    aoi = tiles.MercatorBBox(
        min_x=min(x0, x1), min_y=min(y0, y1), max_x=max(x0, x1), max_y=max(y0, y1)
    )
    assert _resolve_zoom({"zoom": 17}, px_per_meter=None, aoi=aoi)[0] == 17
    assert _resolve_zoom({"gsd_m": 0.3}, px_per_meter=None, aoi=aoi)[0] == 19
    z, notes = _resolve_zoom({}, px_per_meter=3.333, aoi=aoi)
    assert z == 19 and notes


def test_aoi_to_mercator_passthrough_3857() -> None:
    bbox = (12694800.0, 2561600.0, 12695200.0, 2562000.0)
    merc = tiles.aoi_to_mercator_bbox(bbox, "EPSG:3857")
    assert merc.as_list() == list(bbox)


def _synthetic_tiles(rng: tiles.TileRange) -> dict[tuple[int, int], np.ndarray | None]:
    out: dict[tuple[int, int], np.ndarray | None] = {}
    for x, y in rng.tiles():
        arr = np.zeros((256, 256, 3), dtype=np.uint8)
        arr[..., 0] = (x - rng.x_min) * 60 + 10
        arr[..., 1] = (y - rng.y_min) * 60 + 10
        arr[..., 2] = 200
        out[(x, y)] = arr
    return out


def test_stitch_crops_and_georeferences(tmp_path: Path) -> None:
    z = 16
    rng = tiles.TileRange(z=z, x_min=1000, y_min=2000, x_max=1002, y_max=2001)
    res = tiles.resolution(z)
    grid_tl = tiles.tile_bounds(z, 1000, 2000)
    crop = tiles.MercatorBBox(
        min_x=grid_tl.min_x + 100.3 * res,
        min_y=grid_tl.max_y - (256 + 200.2) * res,
        max_x=grid_tl.min_x + (512 + 50.6) * res,
        max_y=grid_tl.max_y - 10.0 * res,
    )
    result = tiles.stitch(_synthetic_tiles(rng), rng, crop)
    assert result.array.shape == (
        int(math.ceil(256 + 200.2)) - 10,
        int(math.ceil(512 + 50.6)) - 100,
        3,
    )
    assert result.bbox.min_x == pytest.approx(grid_tl.min_x + 100 * res)
    assert result.bbox.max_y == pytest.approx(grid_tl.max_y - 10 * res)
    out = tmp_path / "stitch.tif"
    tiles.write_geotiff_3857(out, result)
    assert out.exists() and out.stat().st_size > 0
