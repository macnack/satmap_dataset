"""Integration test for the reprojected-tile mosaic path in `_render_year`.

Reprojected tiles are expanded by gdalwarp to the full AOI extent and padded
with black where the tile has no data. Before the alpha fix, every reprojected
tile claimed the whole AOI and `canvas.insert` pasted its mostly-black canvas,
each overwriting the previous one -> the year rendered (near) black even though
geographic coverage was 100%.

These tests pin the fix: padding must be masked by the gdalwarp alpha band so a
tile only contributes where it actually has data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.pipeline import render

pyvips = pytest.importorskip("pyvips")


def _write_half_tile(path: Path, data_side: str, *, size: int = 100, value: int = 200) -> None:
    """Write a size×size RGBA tile that emulates a reprojected tile spanning the
    full AOI: one half carries real data (alpha=255), the other is black padding
    (alpha=0)."""
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    half = size // 2
    if data_side == "left":
        arr[:, :half, :3] = value
        arr[:, :half, 3] = 255
    else:
        arr[:, half:, :3] = value
        arr[:, half:, 3] = 255
    img = pyvips.Image.new_from_memory(arr.tobytes(), size, size, 4, "uchar")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.tiffsave(str(path))


def test_render_year_masks_reprojected_padding(tmp_path: Path, monkeypatch) -> None:
    size = 100
    target_bbox = render.BBox(min_x=0.0, min_y=0.0, max_x=float(size), max_y=float(size))

    repro = tmp_path / "_reprojected"
    tile_a = repro / "tileA__epsg2180_aaa.tif"
    tile_b = repro / "tileB__epsg2180_bbb.tif"
    _write_half_tile(tile_a, "left", size=size)
    _write_half_tile(tile_b, "right", size=size)

    # Every reprojected tile reports the full-AOI extent (the bug trigger).
    geo = render.GeoRef(
        origin_x=0.0,
        origin_y=float(size),
        pixel_size_x=1.0,
        pixel_size_y=1.0,
        width=size,
        height=size,
    )
    monkeypatch.setattr(render, "_read_georef", lambda _p: geo)

    out = tmp_path / "year_2014.tiff"
    render._render_year(
        year=2014,
        assets=[tile_a, tile_b],
        out_path=out,
        target_bbox=target_bbox,
        target_width=size,
        target_height=size,
        target_srs="EPSG:2180",
        resample_method="bilinear",
        tile_size=64,
        compression="deflate",
        force_srgb_from_ycbcr=False,
        per_year_color_norm=False,
        source_axis_mode="normal",
    )

    img = pyvips.Image.new_from_file(str(out), access="random")
    arr = np.ndarray(
        buffer=img.write_to_memory(),
        dtype=np.uint8,
        shape=(img.height, img.width, img.bands),
    )
    half = size // 2
    left_mean = float(arr[:, :half, :].mean())
    right_mean = float(arr[:, half:, :].mean())

    # tileA owns the left half, tileB the right half; neither tile's black
    # padding may overwrite the other's data.
    assert left_mean > 150, f"left half lost to padding (mean={left_mean})"
    assert right_mean > 150, f"right half lost to padding (mean={right_mean})"
