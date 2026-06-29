"""Ported, self-contained copy of sat_roma `romatch/datasets/raw_tiles.py`.

SOURCE OF TRUTH: sat_roma romatch/datasets/raw_tiles.py — keep this file in sync;
drift is caught by tests/test_raw_tiles_core.py (shared vectors).

Pure georeferencing + indexing logic with no training dependencies. Reads each
tile's geotransform and CRS via the `gdalinfo` CLI and writes the `.tfw`/`.prj`
sidecars the raw-tile datasets expect. Everything geometric is read from
metadata; nothing is parsed from filenames except, as a fallback, the
acquisition year.
"""
from __future__ import annotations

import json
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pyvips
import yaml

_EPSG_RE = re.compile(r'ID\["EPSG",(\d+)\]')


@dataclass(frozen=True)
class GeoTransform:
    """GDAL geotransform (pixel-corner origin). Order matches gdalinfo -json."""
    ulx: float
    xres: float
    xskew: float
    uly: float
    yskew: float
    yres: float


def gdalinfo_json(tif: Path) -> dict:
    out = subprocess.run(
        ["gdalinfo", "-json", str(tif)], capture_output=True, text=True, check=True
    ).stdout
    return json.loads(out)


def read_geotransform(tif: Path) -> GeoTransform:
    g = gdalinfo_json(tif)["geoTransform"]
    return GeoTransform(ulx=g[0], xres=g[1], xskew=g[2], uly=g[3], yskew=g[4], yres=g[5])


def read_crs_wkt(tif: Path) -> str:
    return gdalinfo_json(tif)["coordinateSystem"]["wkt"]


def _epsg_from_wkt(wkt: str) -> "int | None":
    """EPSG of the CRS itself, or None if the WKT carries no CRS authority.

    Takes the ``ID["EPSG",N]`` written at the TOP LEVEL of the CRS node (bracket
    depth 1), never a nested unit/method/parameter/datum code. Robust to WKTs
    where the projected CRS has no authority and the trailing id is a unit
    (e.g. some Geoportal tiles end in ``ID["EPSG",9001]``, the metre unit).
    """
    best = None
    for m in _EPSG_RE.finditer(wkt):
        depth = wkt.count("[", 0, m.start()) - wkt.count("]", 0, m.start())
        if depth == 1:
            best = int(m.group(1))
    return best


def geotransform_to_tfw_lines(gt: GeoTransform) -> list:
    a, d, b, e = gt.xres, gt.yskew, gt.xskew, gt.yres
    c = gt.ulx + 0.5 * a + 0.5 * b
    f = gt.uly + 0.5 * d + 0.5 * e
    return [a, d, b, e, c, f]


def write_tfw(gt: GeoTransform, path: Path) -> None:
    path.write_text("".join(f"{v:.12f}\n" for v in geotransform_to_tfw_lines(gt)))


def write_prj_wkt(wkt: str, path: Path) -> None:
    path.write_text(wkt)


_REGISTRY_PATH = Path(__file__).with_name("raw_tile_providers.yaml")
_YEAR_DIR_RE = re.compile(r"^\d{4}$")
_YEAR_TOKEN_RE = re.compile(r"(?:^|_)(?:year_)?((?:19|20)\d{2})(?:_|$)")


def load_provider_registry(path: "Path | None" = None) -> dict:
    """EPSG -> {"provider": str, "min_coverage": float | None}.

    A registry value may be a plain provider name or a mapping with an optional
    per-provider ``min_coverage`` override; both are normalised to a dict.
    """
    data = yaml.safe_load((path or _REGISTRY_PATH).read_text())
    out: dict = {}
    for k, v in data.items():
        if isinstance(v, dict):
            out[int(k)] = {"provider": str(v["provider"]),
                           "min_coverage": v.get("min_coverage")}
        else:
            out[int(k)] = {"provider": str(v), "min_coverage": None}
    return out


def _registry_entry(epsg: int, registry: dict) -> dict:
    try:
        entry = registry[epsg]
    except KeyError:
        raise KeyError(
            f"EPSG:{epsg} has no provider in {_REGISTRY_PATH.name}; add a line "
            f"'{epsg}: <provider>'."
        ) from None
    return entry if isinstance(entry, dict) else {"provider": str(entry), "min_coverage": None}


def provider_for_epsg(epsg: int, registry: dict) -> str:
    return _registry_entry(epsg, registry)["provider"]


def min_coverage_for_epsg(epsg: int, registry: dict, default: float = 0.95) -> float:
    """Per-provider coverage gate from the registry, else ``default``."""
    mc = _registry_entry(epsg, registry).get("min_coverage")
    return float(mc) if mc is not None else float(default)


def detect_year(tif: Path) -> int:
    if _YEAR_DIR_RE.match(tif.parent.name):
        return int(tif.parent.name)
    m = _YEAR_TOKEN_RE.search(tif.stem)
    if m:
        return int(m.group(1))
    dt = gdalinfo_json(tif).get("metadata", {}).get("", {}).get("TIFFTAG_DATETIME")
    if dt:
        return int(str(dt)[:4])
    raise ValueError(f"Cannot determine year for {tif}")


@dataclass(frozen=True)
class TileInfo:
    path: "Path | None"
    width: int
    height: int
    gt: GeoTransform
    epsg: "int | None"
    wkt: str
    year: int

    @property
    def gsd(self) -> float:
        return self.gt.xres

    @property
    def width_m(self) -> float:
        return self.width * self.gt.xres

    @property
    def height_m(self) -> float:
        return self.height * -self.gt.yres

    @property
    def lrx(self) -> float:
        return self.gt.ulx + self.width_m

    @property
    def lry(self) -> float:
        return self.gt.uly - self.height_m


def read_tile_info(tif: Path) -> TileInfo:
    hdr = pyvips.Image.new_from_file(str(tif))
    w, h = hdr.width, hdr.height
    del hdr
    wkt = read_crs_wkt(tif)
    return TileInfo(tif, w, h, read_geotransform(tif), _epsg_from_wkt(wkt), wkt, detect_year(tif))


@dataclass(frozen=True)
class Cell:
    ulx: float
    uly: float
    w_m: float
    h_m: float


def cell_key(cell: Cell) -> str:
    return f"e{round(cell.ulx)}_n{round(cell.uly - cell.h_m)}"


def derive_cell_grid(tiles: list, cell_size_m: "float | None" = None) -> list:
    if cell_size_m is not None:
        w = h = float(cell_size_m)
    else:
        base = min(tiles, key=lambda t: t.width_m * t.height_m)
        w, h = base.width_m, base.height_m
    seen: dict = {}
    for t in tiles:
        if abs(t.width_m - w) < 1.0 and abs(t.height_m - h) < 1.0:
            seen.setdefault((t.gt.ulx, t.gt.uly), Cell(t.gt.ulx, t.gt.uly, w, h))
    return list(seen.values())


def tile_covers_cell(tile: TileInfo, cell: Cell, tol_m: float = 0.01) -> bool:
    return (
        tile.gt.ulx <= cell.ulx + tol_m
        and tile.gt.uly >= cell.uly - tol_m
        and tile.lrx >= cell.ulx + cell.w_m - tol_m
        and tile.lry <= cell.uly - cell.h_m + tol_m
    )


def world_window_to_pixel(gt: GeoTransform, ulx: float, uly: float,
                          w_m: float, h_m: float) -> tuple:
    out = ((ulx - gt.ulx) / gt.xres, (uly - gt.uly) / gt.yres, w_m / gt.xres, h_m / -gt.yres)
    rounded = tuple(round(v) for v in out)
    if any(abs(v - r) > 0.51 for v, r in zip(out, rounded)):
        raise ValueError(f"World window not integer-aligned to source grid: {out}")
    return rounded


def valid_pixel_fraction(image: "pyvips.Image", nodata: int = 0) -> float:
    """Fraction of pixels where ANY band differs from nodata (default 0)."""
    valid = image[0] != nodata
    for i in range(1, image.bands):
        valid = valid | (image[i] != nodata)
    return float(valid.avg()) / 255.0


def resolve_season_tile(cell: Cell, year_tiles: list) -> "TileInfo | None":
    """Return the covering tile for a cell, preferring the finest GSD.

    Returns None if no tile in year_tiles covers the cell.
    """
    covering = [t for t in year_tiles if tile_covers_cell(t, cell)]
    return min(covering, key=lambda t: t.gsd) if covering else None


_COVERAGE_THUMB_PX = 2048  # coverage estimate resolution (shrink-on-load when overviews exist)


def ingest_area(src_area: Path, out_root: Path, registry: dict, *,
                cell_size_m: "float | None" = None, min_coverage: "float | None" = None) -> dict:
    """Scan one area dir and write co-located season stacks into out_root.

    Resolves <provider> from the first tile's EPSG, builds a geographic cell
    grid (defaulting to the smallest tile footprint), writes
    ``<out_root>/<provider>/<area>/<cellkey>/year_YYYY.tif`` (+ ``.tfw``/``.prj``),
    and returns a manifest dict. Native (cell-sized) tiles are symlinked 1:1;
    larger tiles are losslessly cropped via pyvips.tiffsave.

    Parameters
    ----------
    src_area:      source <provider>/<area> directory containing <year>/*.tif subdirs.
    out_root:      output root (e.g. sat_data_raw).
    registry:      EPSG->provider dict from load_provider_registry.
    cell_size_m:   override cell size in metres; defaults to the smallest tile footprint.
    min_coverage:  drop (cell, year) pairs whose valid-pixel fraction is below this.
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

    manifest: dict = {"provider": provider, "area": src_area.name, "epsg": epsg,
                      "cell_size_m": [round(cells[0].w_m, 3), round(cells[0].h_m, 3)] if cells else None,
                      "locations": {}}
    for cell in cells:
        key = cell_key(cell)
        loc_dir = out_area / key
        seasons = []
        for year, yts in sorted(by_year.items()):
            tile = resolve_season_tile(cell, yts)
            if tile is None:
                continue
            x, y, w, h = world_window_to_pixel(tile.gt, cell.ulx, cell.uly,
                                               cell.w_m, cell.h_m)
            native = (w == tile.width and h == tile.height)
            # Coverage gate: for native (full-tile) seasons estimate it from a
            # shrink-on-load thumbnail (cheap; avoids decoding gigapixel tiles just
            # to gate them); only sub-window crops decode the actual window.
            if native:
                cov = valid_pixel_fraction(
                    pyvips.Image.thumbnail(str(tile.path), _COVERAGE_THUMB_PX))
                crop = None
            else:
                crop = pyvips.Image.new_from_file(str(tile.path)).crop(x, y, w, h)
                cov = valid_pixel_fraction(crop)
            if cov < mc:
                seasons.append({"year": year, "dropped": True, "coverage": round(cov, 4)})
                continue
            loc_dir.mkdir(parents=True, exist_ok=True)
            out_tif = loc_dir / f"year_{year}.tif"
            if native:                                   # 1:1 -> symlink, no copy
                if out_tif.is_symlink() or out_tif.exists():
                    out_tif.unlink()
                out_tif.symlink_to(tile.path.resolve())
                cell_gt = tile.gt
            else:                                        # sub-window -> lossless crop
                crop.tiffsave(str(out_tif), compression="lzw", tile=True,
                              pyramid=True, bigtiff=True)
                cell_gt = GeoTransform(cell.ulx, tile.gt.xres, 0.0, cell.uly, 0.0, tile.gt.yres)
            write_tfw(cell_gt, loc_dir / f"year_{year}.tfw")
            write_prj_wkt(tile.wkt, loc_dir / f"year_{year}.prj")
            seasons.append({"year": year, "gsd": tile.gsd, "coverage": round(cov, 4),
                            "source": tile.path.name, "native": native})
        if any(not s.get("dropped") for s in seasons):
            manifest["locations"][key] = {
                "alias": None,
                "bbox": [cell.ulx, cell.uly - cell.h_m, cell.ulx + cell.w_m, cell.uly],
                "seasons": seasons,
            }
    return manifest
