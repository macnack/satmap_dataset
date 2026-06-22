"""World-window ingest for mixed-GSD areas (satmap extension; not in sat_roma).

The ported `core.ingest_area` keys cells on a single tile origin and requires a
covering tile per cell. When a location is captured at multiple GSDs across
years (e.g. Geoportal godło sheets: 0.25 m most years, 0.05 m others), the finer
grid is offset from the coarser one by a sub-metre, known translation, so no
single origin is covered by both families and the years split into separate
cells that collide on one `cell_key`.

This module instead resolves, per geographic *spot*, the common geographic
window = the intersection of every year's covering-tile footprint, snapped
inward to the **coarsest** grid present (so the window is integer-pixel-aligned
to every GSD — the coarse grid step is an exact multiple of the finer one). Each
year is then losslessly cropped to that identical window at its native GSD, and
finer years are downsampled to the coarse GSD by exact integer decimation
(box average) so all years land co-registered AND equal-dimension. The native
full-resolution tiles remain available under the export `<year>/` folders.

Reuses the ported `core` primitives; `core.py` stays verbatim.
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import pyvips

from satmap_dataset.raw_tiles.core import (
    GeoTransform,
    cell_key,
    derive_cell_grid,
    min_coverage_for_epsg,
    provider_for_epsg,
    read_tile_info,
    valid_pixel_fraction,
    world_window_to_pixel,
    write_prj_wkt,
    write_tfw,
    _YEAR_DIR_RE,
)

_SNAP_EPS = 1e-6  # fractional-pixel tolerance when snapping to the coarse grid


def _tile_contains(t, x: float, y: float) -> bool:
    return t.gt.ulx <= x <= t.lrx and t.lry <= y <= t.gt.uly


def common_window(chosen: dict) -> tuple:
    """Common geographic window for a spot's per-year covering tiles.

    `chosen` maps year -> TileInfo (one covering tile per year). Returns
    ``(ulx, uly, w_m, h_m, res)`` where ``res`` is the coarsest GSD present and
    ``(ulx, uly)`` sits on that coarse grid, so the window is integer-pixel
    aligned to every tile. Returns None if the snapped window is empty.
    """
    gref = max(chosen.values(), key=lambda t: t.gt.xres)  # coarsest grid defines the snap
    ox, oy, res = gref.gt.ulx, gref.gt.uly, gref.gt.xres
    left = max(t.gt.ulx for t in chosen.values())
    right = min(t.lrx for t in chosen.values())
    top = min(t.gt.uly for t in chosen.values())
    bot = max(t.lry for t in chosen.values())
    ulx = ox + math.ceil((left - ox) / res - _SNAP_EPS) * res
    rgt = ox + math.floor((right - ox) / res + _SNAP_EPS) * res
    uly = oy - math.ceil((oy - top) / res - _SNAP_EPS) * res
    bot_s = oy - math.floor((oy - bot) / res + _SNAP_EPS) * res
    w_m, h_m = rgt - ulx, uly - bot_s
    if w_m <= 0 or h_m <= 0:
        return None
    return (ulx, uly, w_m, h_m, res)


def _equalize_to_grid(img: "pyvips.Image", gsd: float, res: float, tw: int, th: int) -> tuple:
    """Resample a native-GSD crop to the coarse target grid at exactly (tw, th).

    Integer ratio (res/gsd) -> exact box-average decimation (no interpolation;
    Geoportal 0.05->0.25 = 5x). Non-integer ratio -> anti-aliased Lanczos
    downscale to the coarsest GSD (Lantmäteriet 0.16->0.25 = 1.5625x). Equal GSD
    -> identity. Returns ``(out_img, method)`` with method in
    {"native", "decimate", "resize"}.
    """
    factor = res / gsd
    fi = round(factor)
    if abs(factor - 1.0) < 1e-9:
        out, method = img, "native"
    elif fi >= 2 and abs(factor - fi) < 1e-6:
        out, method = img.shrink(fi, fi), "decimate"  # exact integer box average
    else:
        out = img.resize(tw / img.width, kernel="lanczos3", vscale=th / img.height)
        method = "resize"
    # Guard against ±1 px rounding so every year lands on identical dims.
    if out.width != tw or out.height != th:
        out = out.crop(0, 0, min(out.width, tw), min(out.height, th))
        if out.width != tw or out.height != th:
            out = out.embed(0, 0, tw, th, extend="copy")
    return out, method


def ingest_area_world_window(src_area: Path, out_root: Path, registry: dict, *,
                             cell_size_m: "float | None" = None,
                             min_coverage: "float | None" = None) -> dict:
    """Mixed-GSD-aware ingest. Writes co-registered, equal-dimension season
    stacks ``<out_root>/<provider>/<area>/<cellkey>/year_YYYY.tif`` (+ tfw/prj),
    every year resampled (finer years only, by exact integer decimation) to the
    coarsest GSD of its spot. Returns a manifest dict shaped like
    `core.ingest_area`'s, with extra per-season `native_gsd`/`window_gsd`/`dims`.
    """
    tiles = [read_tile_info(p) for p in sorted(src_area.glob("*/*.tif"))
             if _YEAR_DIR_RE.match(p.parent.name)]
    if not tiles:
        raise SystemExit(f"No <year>/*.tif under {src_area}")
    epsg = next(t.epsg for t in tiles if t.epsg is not None)
    provider = provider_for_epsg(epsg, registry)
    mc = min_coverage if min_coverage is not None else min_coverage_for_epsg(epsg, registry)
    out_area = out_root / provider / src_area.name
    cells = derive_cell_grid(tiles, cell_size_m)
    by_year: dict = defaultdict(list)
    for t in tiles:
        by_year[t.year].append(t)

    # Collapse float-jitter / mixed-GSD cells that share a cell_key into one spot.
    spots: dict = {}
    for c in cells:
        spots.setdefault(cell_key(c), c)

    manifest: dict = {"provider": provider, "area": src_area.name, "epsg": epsg,
                      "cell_size_m": [round(cells[0].w_m, 3), round(cells[0].h_m, 3)] if cells else None,
                      "locations": {}}
    for key, cell in spots.items():
        cx, cy = cell.ulx + cell.w_m / 2.0, cell.uly - cell.h_m / 2.0
        chosen: dict = {}
        for year, yts in sorted(by_year.items()):
            covering = [t for t in yts if _tile_contains(t, cx, cy)]
            if covering:
                chosen[year] = min(covering, key=lambda t: t.gsd)  # finest GSD
        if not chosen:
            continue
        win = common_window(chosen)
        if win is None:
            continue
        ulx, uly, w_m, h_m, res = win
        tw, th = round(w_m / res), round(h_m / res)  # shared equal-dim target
        loc_dir = out_area / key
        coarse_gt = GeoTransform(ulx, res, 0.0, uly, 0.0, -res)
        seasons = []
        for year, t in sorted(chosen.items()):
            x, y, w, h = world_window_to_pixel(t.gt, ulx, uly, w_m, h_m)
            # Random access (not sequential): tiffsave(pyramid=True) makes multiple
            # passes over the region to build overviews; real source tiles are tiled.
            img = pyvips.Image.new_from_file(str(t.path)).crop(x, y, w, h)
            out_img, method = _equalize_to_grid(img, t.gt.xres, res, tw, th)
            cov = valid_pixel_fraction(out_img)
            if cov < mc:
                seasons.append({"year": year, "dropped": True, "coverage": round(cov, 4),
                                "native_gsd": t.gsd})
                continue
            loc_dir.mkdir(parents=True, exist_ok=True)
            out_tif = loc_dir / f"year_{year}.tif"
            out_img.tiffsave(str(out_tif), compression="lzw", tile=True, pyramid=True, bigtiff=True)
            write_tfw(coarse_gt, loc_dir / f"year_{year}.tfw")
            write_prj_wkt(t.wkt, loc_dir / f"year_{year}.prj")
            seasons.append({"year": year, "window_gsd": round(res, 6), "native_gsd": t.gsd,
                            "coverage": round(cov, 4), "source": t.path.name,
                            "dims": [out_img.width, out_img.height],
                            "downsampled": method != "native", "resample": method})
        if any(not s.get("dropped") for s in seasons):
            manifest["locations"][key] = {
                "alias": None,
                "bbox": [ulx, uly - h_m, ulx + w_m, uly],
                "window_gsd": round(res, 6),
                "seasons": seasons,
            }
    return manifest
