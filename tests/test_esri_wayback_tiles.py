from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.pipeline import render
from satmap_dataset.providers.esri_wayback import tiles
from satmap_dataset.providers.esri_wayback.client import metadata_layer_id_for_zoom
from satmap_dataset.providers.esri_wayback.provider import resolve_download_zoom


def test_lonlat_tile_matches_live_service_indexing() -> None:
    # Warsaw center at z17 — the same x/y the live tilemap/tile endpoints use.
    x, y = tiles.lonlat_to_mercator(21.0122, 52.2297)
    assert tiles.mercator_to_tile(x, y, 17) == (73186, 43158)
    lon, lat = tiles.mercator_to_lonlat(x, y)
    assert lon == pytest.approx(21.0122) and lat == pytest.approx(52.2297)


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


def test_zoom_for_gsd_and_ground_gsd() -> None:
    assert tiles.ground_gsd(17, 52.0) == pytest.approx(1.1943 * math.cos(math.radians(52.0)), rel=1e-3)
    assert tiles.zoom_for_gsd(1.0, 52.0, min_zoom=10, max_zoom=18) == 17
    assert tiles.zoom_for_gsd(0.5, 52.0, min_zoom=10, max_zoom=18) == 18
    assert tiles.zoom_for_gsd(0.01, 52.0, min_zoom=10, max_zoom=18) == 18


def test_resolve_download_zoom_precedence_and_cap() -> None:
    aoi = tiles.aoi_to_mercator_bbox((634515.269, 486072.873, 636515.269, 488072.873), "EPSG:2180")
    assert resolve_download_zoom({}, px_per_meter=1.0, aoi=aoi)[0] == 17
    assert resolve_download_zoom({"gsd_m": 0.4}, px_per_meter=1.0, aoi=aoi)[0] == 18
    assert resolve_download_zoom({"zoom": 15}, px_per_meter=1.0, aoi=aoi)[0] == 15
    z, notes = resolve_download_zoom({"max_tiles": 100}, px_per_meter=4.0, aoi=aoi)
    assert z < 18 and notes
    with pytest.raises(ValueError, match="max_tiles"):
        resolve_download_zoom({"zoom": 19, "max_tiles": 100}, px_per_meter=1.0, aoi=aoi)


def test_aoi_to_mercator_envelope_contains_projected_corners() -> None:
    from pyproj import Transformer

    bbox = (634515.269, 486072.873, 636515.269, 488072.873)
    merc = tiles.aoi_to_mercator_bbox(bbox, "EPSG:2180")
    tr = Transformer.from_crs("EPSG:2180", "EPSG:3857", always_xy=True)
    for x, y in [(bbox[0], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3]), (bbox[2], bbox[1])]:
        mx, my = tr.transform(x, y)
        assert merc.min_x - 1e-3 <= mx <= merc.max_x + 1e-3
        assert merc.min_y - 1e-3 <= my <= merc.max_y + 1e-3


def test_probe_tiles_dedupes() -> None:
    small = tiles.tile_bounds(17, 100, 100)
    assert tiles.probe_tiles(small, 17, "grid") == [(17, 100, 100)]
    assert len(tiles.probe_tiles(tiles.pad_bbox(small, 200.0), 17, "grid")) == 5
    with pytest.raises(ValueError):
        tiles.probe_tiles(small, 17, "everything")


def test_metadata_layer_for_zoom() -> None:
    assert metadata_layer_id_for_zoom(17) == 6
    assert metadata_layer_id_for_zoom(18) == 5
    assert metadata_layer_id_for_zoom(3) == 13
    assert metadata_layer_id_for_zoom(25) == 0


def _synthetic_tiles(rng: tiles.TileRange) -> dict[tuple[int, int], np.ndarray | None]:
    out: dict[tuple[int, int], np.ndarray | None] = {}
    for i, (x, y) in enumerate(rng.tiles()):
        arr = np.zeros((256, 256, 3), dtype=np.uint8)
        arr[..., 0] = (x - rng.x_min) * 60 + 10
        arr[..., 1] = (y - rng.y_min) * 60 + 10
        arr[..., 2] = 200
        out[(x, y)] = arr
    return out


def test_stitch_crops_and_georeferences() -> None:
    z = 16
    rng = tiles.TileRange(z=z, x_min=1000, y_min=2000, x_max=1002, y_max=2001)
    res = tiles.resolution(z)
    grid_tl = tiles.tile_bounds(z, 1000, 2000)
    # Crop starts 100.3 px into the first tile and ends 50.6 px into the last column.
    crop = tiles.MercatorBBox(
        min_x=grid_tl.min_x + 100.3 * res,
        min_y=grid_tl.max_y - (256 + 200.2) * res,
        max_x=grid_tl.min_x + (512 + 50.6) * res,
        max_y=grid_tl.max_y - 10.0 * res,
    )
    result = tiles.stitch(_synthetic_tiles(rng), rng, crop)
    # Whole pixels covering the crop: floor(min) .. ceil(max).
    assert result.array.shape == (int(math.ceil(256 + 200.2)) - 10, int(math.ceil(512 + 50.6)) - 100, 3)
    assert result.bbox.min_x == pytest.approx(grid_tl.min_x + 100 * res)
    assert result.bbox.max_y == pytest.approx(grid_tl.max_y - 10 * res)
    assert result.bbox.min_x <= crop.min_x and result.bbox.max_x >= crop.max_x
    assert result.bbox.min_y <= crop.min_y and result.bbox.max_y >= crop.max_y
    # Pixel (0,0) is tile (1000,2000); the right-most column falls in tile x=1002.
    assert tuple(result.array[0, 0]) == (10, 10, 200)
    assert tuple(result.array[0, -1]) == (130, 10, 200)
    assert tuple(result.array[-1, 0]) == (10, 70, 200)


def test_stitch_fills_missing_tiles_black() -> None:
    rng = tiles.TileRange(z=12, x_min=5, y_min=5, x_max=6, y_max=5)
    tl = _synthetic_tiles(rng)
    tl[(6, 5)] = None
    full = tiles.MercatorBBox(
        tiles.tile_bounds(12, 5, 5).min_x, tiles.tile_bounds(12, 5, 5).min_y,
        tiles.tile_bounds(12, 6, 5).max_x, tiles.tile_bounds(12, 6, 5).max_y,
    )
    result = tiles.stitch(tl, rng, full)
    assert result.array.shape == (256, 512, 3)
    assert result.array[:, 256:].max() == 0


def test_write_geotiff_3857_round_trips_through_render_readers(tmp_path: Path) -> None:
    rng = tiles.TileRange(z=15, x_min=18000, y_min=10900, x_max=18001, y_max=10901)
    crop = tiles.pad_bbox(tiles.tile_bounds(15, 18000, 10900), -500.0)
    result = tiles.stitch(_synthetic_tiles(rng), rng, crop)
    out = tmp_path / "2023" / "wayback_1_z15.tif"
    tiles.write_geotiff_3857(out, result)
    assert out.exists() and not (tmp_path / "2023" / "wayback_1_z15.tif.part").exists()
    assert render._read_source_epsg(out) == 3857
    georef = render._read_georef(out)
    assert georef.origin_x == pytest.approx(result.bbox.min_x)
    assert georef.origin_y == pytest.approx(result.bbox.max_y)
    assert georef.pixel_size_x == pytest.approx(tiles.resolution(15))
    assert (georef.width, georef.height) == (result.array.shape[1], result.array.shape[0])


def test_write_geotiff_failure_leaves_no_partial(tmp_path: Path, monkeypatch) -> None:
    import tifffile

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(tifffile, "imwrite", boom)
    result = tiles.StitchResult(np.zeros((4, 4, 3), dtype=np.uint8), tiles.MercatorBBox(0, 0, 4, 4), 1.0)
    out = tmp_path / "x.tif"
    with pytest.raises(OSError):
        tiles.write_geotiff_3857(out, result)
    assert not out.exists()
