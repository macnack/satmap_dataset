"""Web Mercator (EPSG:3857) XYZ tile math, zoom selection and GeoTIFF stitching.

Shared helpers used by the LandsD HK Imagery XYZ provider.
"""

import io
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from satmap_dataset.io.atomic import part_path_for, unlink_quiet

EARTH_RADIUS_M = 6378137.0
ORIGIN_SHIFT = math.pi * EARTH_RADIUS_M  # 20037508.342789244
TILE_SIZE = 256
MAX_LAT = 85.0511287798066
MERCATOR_EPSG = 3857


@dataclass(frozen=True)
class MercatorBBox:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def as_list(self) -> list[float]:
        return [self.min_x, self.min_y, self.max_x, self.max_y]

    @property
    def center(self) -> tuple[float, float]:
        return ((self.min_x + self.max_x) / 2.0, (self.min_y + self.max_y) / 2.0)


@dataclass(frozen=True)
class TileRange:
    z: int
    x_min: int
    y_min: int
    x_max: int  # inclusive
    y_max: int  # inclusive

    @property
    def cols(self) -> int:
        return self.x_max - self.x_min + 1

    @property
    def rows(self) -> int:
        return self.y_max - self.y_min + 1

    @property
    def count(self) -> int:
        return self.cols * self.rows

    def tiles(self) -> list[tuple[int, int]]:
        return [(x, y) for y in range(self.y_min, self.y_max + 1) for x in range(self.x_min, self.x_max + 1)]


def resolution(z: int) -> float:
    """Mercator metres per pixel at zoom ``z`` (true at the equator only)."""
    return (2.0 * ORIGIN_SHIFT) / (TILE_SIZE * (2**z))


def ground_gsd(z: int, lat_deg: float) -> float:
    """Approximate ground metres per pixel at zoom ``z`` and latitude ``lat_deg``."""
    return resolution(z) * math.cos(math.radians(lat_deg))


def lonlat_to_mercator(lon: float, lat: float) -> tuple[float, float]:
    lat = max(-MAX_LAT, min(MAX_LAT, lat))
    x = math.radians(lon) * EARTH_RADIUS_M
    y = math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0)) * EARTH_RADIUS_M
    return x, y


def mercator_to_lonlat(x: float, y: float) -> tuple[float, float]:
    lon = math.degrees(x / EARTH_RADIUS_M)
    lat = math.degrees(2.0 * math.atan(math.exp(y / EARTH_RADIUS_M)) - math.pi / 2.0)
    return lon, lat


def mercator_to_tile(x: float, y: float, z: int) -> tuple[int, int]:
    n = 2**z
    span = 2.0 * ORIGIN_SHIFT
    # Tolerance absorbs float error so a coordinate on a tile edge maps to the
    # tile that starts there.
    tx = int(math.floor((x + ORIGIN_SHIFT) / span * n + 1e-9))
    ty = int(math.floor((ORIGIN_SHIFT - y) / span * n + 1e-9))
    return max(0, min(n - 1, tx)), max(0, min(n - 1, ty))


def tile_bounds(z: int, x: int, y: int) -> MercatorBBox:
    size = TILE_SIZE * resolution(z)
    min_x = -ORIGIN_SHIFT + x * size
    max_y = ORIGIN_SHIFT - y * size
    return MercatorBBox(min_x=min_x, min_y=max_y - size, max_x=min_x + size, max_y=max_y)


def tile_range_for_bbox(bbox: MercatorBBox, z: int) -> TileRange:
    # Nudge the max edge inward so a bbox ending exactly on a tile edge does not
    # pull in an extra row/column of tiles.
    eps = resolution(z) * 1e-3
    x0, y0 = mercator_to_tile(bbox.min_x, bbox.max_y, z)
    x1, y1 = mercator_to_tile(bbox.max_x - eps, bbox.min_y + eps, z)
    return TileRange(z=z, x_min=x0, y_min=y0, x_max=x1, y_max=y1)


def aoi_to_mercator_bbox(bbox: tuple[float, float, float, float], srs: str) -> MercatorBBox:
    """Densified envelope of a projected AOI bbox in EPSG:3857."""
    if srs.upper() in {"EPSG:3857", "EPSG:900913", "EPSG:102100"}:
        return MercatorBBox(*bbox)
    from pyproj import Transformer

    transformer = Transformer.from_crs(srs, "EPSG:4326", always_xy=True)
    lon_min, lat_min, lon_max, lat_max = transformer.transform_bounds(*bbox, densify_pts=21)
    x0, y0 = lonlat_to_mercator(lon_min, lat_min)
    x1, y1 = lonlat_to_mercator(lon_max, lat_max)
    return MercatorBBox(min_x=x0, min_y=y0, max_x=x1, max_y=y1)


def pad_bbox(bbox: MercatorBBox, pad: float) -> MercatorBBox:
    return MercatorBBox(bbox.min_x - pad, bbox.min_y - pad, bbox.max_x + pad, bbox.max_y + pad)


def zoom_for_gsd(target_gsd_m: float, lat_deg: float, *, min_zoom: int, max_zoom: int) -> int:
    """Smallest zoom whose ground GSD is at least as fine as ``target_gsd_m``."""
    if target_gsd_m <= 0:
        raise ValueError("target GSD must be > 0")
    for z in range(min_zoom, max_zoom + 1):
        if ground_gsd(z, lat_deg) <= target_gsd_m * (1.0 + 1e-9):
            return z
    return max_zoom


def probe_tiles(bbox: MercatorBBox, z: int, mode: str) -> list[tuple[int, int, int]]:
    """Tiles used to detect local changes: AOI center, plus corners for ``grid``."""
    cx, cy = bbox.center
    points = [(cx, cy)]
    if mode == "grid":
        eps = resolution(z) * 1e-3
        x0, x1 = bbox.min_x, bbox.max_x - eps
        y0, y1 = bbox.min_y + eps, bbox.max_y
        points += [(x0, y1), (x1, y1), (x0, y0), (x1, y0)]
    elif mode != "center":
        raise ValueError("probe must be 'center' or 'grid'")
    seen: list[tuple[int, int, int]] = []
    for px, py in points:
        tx, ty = mercator_to_tile(px, py, z)
        if (z, tx, ty) not in seen:
            seen.append((z, tx, ty))
    return seen


def decode_tile(data: bytes) -> np.ndarray:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        rgb = image.convert("RGB")
        arr = np.asarray(rgb, dtype=np.uint8)
    if arr.shape[:2] != (TILE_SIZE, TILE_SIZE):
        raise ValueError(f"unexpected tile shape {arr.shape}")
    return arr


@dataclass(frozen=True)
class StitchResult:
    array: np.ndarray
    bbox: MercatorBBox
    pixel_size: float


def stitch(
    tiles: dict[tuple[int, int], np.ndarray | None],
    tile_range: TileRange,
    crop_bbox: MercatorBBox,
) -> StitchResult:
    """Mosaic tiles (missing ones black) and crop to whole pixels covering ``crop_bbox``."""
    canvas = np.zeros((tile_range.rows * TILE_SIZE, tile_range.cols * TILE_SIZE, 3), dtype=np.uint8)
    for (x, y), arr in tiles.items():
        if arr is None:
            continue
        r0 = (y - tile_range.y_min) * TILE_SIZE
        c0 = (x - tile_range.x_min) * TILE_SIZE
        canvas[r0 : r0 + TILE_SIZE, c0 : c0 + TILE_SIZE] = arr

    res = resolution(tile_range.z)
    grid = tile_bounds(tile_range.z, tile_range.x_min, tile_range.y_min)
    origin_x, origin_y = grid.min_x, grid.max_y
    c0 = max(0, int(math.floor((crop_bbox.min_x - origin_x) / res)))
    c1 = min(canvas.shape[1], int(math.ceil((crop_bbox.max_x - origin_x) / res)))
    r0 = max(0, int(math.floor((origin_y - crop_bbox.max_y) / res)))
    r1 = min(canvas.shape[0], int(math.ceil((origin_y - crop_bbox.min_y) / res)))
    if c1 <= c0 or r1 <= r0:
        raise ValueError("crop bbox does not intersect the tile range")
    cropped = np.ascontiguousarray(canvas[r0:r1, c0:c1])
    out_bbox = MercatorBBox(
        min_x=origin_x + c0 * res,
        min_y=origin_y - r1 * res,
        max_x=origin_x + c1 * res,
        max_y=origin_y - r0 * res,
    )
    return StitchResult(array=cropped, bbox=out_bbox, pixel_size=res)


def write_geotiff_3857(path: Path, result: StitchResult) -> None:
    """Write an RGB GeoTIFF tagged EPSG:3857 via a sibling ``.part`` file."""
    import tifffile

    from satmap_dataset.pipeline.downloader import _geo_key_directory_for_epsg

    scale = (float(result.pixel_size), float(result.pixel_size), 0.0)
    tie = (0.0, 0.0, 0.0, float(result.bbox.min_x), float(result.bbox.max_y), 0.0)
    geokey = _geo_key_directory_for_epsg(MERCATOR_EPSG)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = part_path_for(path)
    try:
        tifffile.imwrite(
            part,
            result.array,
            photometric="rgb",
            compression="deflate",
            tile=(TILE_SIZE, TILE_SIZE),
            metadata=None,
            extratags=[
                (33550, "d", 3, scale, False),
                (33922, "d", 6, tie, False),
                (34735, "H", len(geokey), geokey, False),
            ],
        )
        part.replace(path)
    except Exception:
        unlink_quiet(part)
        raise
