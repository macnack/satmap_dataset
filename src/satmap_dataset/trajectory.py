from __future__ import annotations

import csv
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrackPoint:
    lat: float
    lon: float


def load_track(path: Path | str) -> list[TrackPoint]:
    """Load a trajectory as WGS84 lat/lon points from a CSV, an IGC file, or a
    directory containing exactly one ``*.igc``."""
    path = Path(path)
    if path.is_dir():
        igc_files = sorted(path.glob("*.igc"))
        if len(igc_files) != 1:
            raise ValueError(
                f"expected exactly one .igc file in {path}, found {len(igc_files)}"
            )
        path = igc_files[0]
    suffix = path.suffix.lower()
    if suffix == ".igc":
        points = _load_igc(path)
    elif suffix == ".csv":
        points = _load_csv(path)
    else:
        raise ValueError(f"unsupported track format: {path.suffix!r} (use .csv or .igc)")
    if not points:
        raise ValueError(f"no track points parsed from {path}")
    return points


def _load_igc(path: Path) -> list[TrackPoint]:
    points: list[TrackPoint] = []
    for line in path.read_text(encoding="latin-1").splitlines():
        if not line.startswith("B") or len(line) < 24:
            continue
        try:
            # IGC DDMMmmmN: 5-digit field = whole_minutes*1000 + thousandths; /60000 = /60/1000
            lat = int(line[7:9]) + int(line[9:14]) / 60000.0
            if line[14] == "S":
                lat = -lat
            lon = int(line[15:18]) + int(line[18:23]) / 60000.0
            if line[23] == "W":
                lon = -lon
        except ValueError:
            continue
        points.append(TrackPoint(lat=lat, lon=lon))
    return points


def _load_csv(path: Path) -> list[TrackPoint]:
    points: list[TrackPoint] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError(f"empty CSV: {path}")
        lower = {name.lower(): name for name in reader.fieldnames}
        lat_key = lower.get("lat") or lower.get("latitude")
        lon_key = lower.get("lon") or lower.get("longitude")
        if lat_key is None or lon_key is None:
            raise ValueError(
                f"CSV must have lat/lon (or latitude/longitude) columns, got {reader.fieldnames}"
            )
        for row in reader:
            try:
                lat = float(row[lat_key])
                lon = float(row[lon_key])
            except (TypeError, ValueError):
                continue
            points.append(TrackPoint(lat=lat, lon=lon))
    return points


@dataclass(frozen=True)
class Cell:
    ix: int
    iy: int
    bbox_2180: tuple[float, float, float, float]
    bbox_wgs84: tuple[float, float, float, float]
    center_lat: float
    center_lon: float
    name: str


def _transform(src_crs: str, dst_crs: str, x: float, y: float) -> tuple[float, float]:
    from satmap_dataset.providers.lantmateriet.crs import transform_point

    try:
        return transform_point(src_crs, dst_crs, x, y)
    except Exception as exc:  # noqa: BLE001 - surface both backends
        raise RuntimeError(
            "Trajectory projection requires pyproj or the PROJ 'proj' CLI in PATH."
        ) from exc


def _densify(
    x0: float, y0: float, x1: float, y1: float, step: float
) -> Iterator[tuple[float, float]]:
    dist = math.hypot(x1 - x0, y1 - y0)
    n = max(1, int(dist // step) + 1)
    for i in range(n + 1):
        t = i / n
        yield x0 + (x1 - x0) * t, y0 + (y1 - y0) * t


def _cells_from_projected(
    projected: list[tuple[float, float]],
    *,
    cell_m: float,
    origin: tuple[float, float],
) -> list[tuple[int, int]]:
    if cell_m <= 0:
        raise ValueError("cell_m must be > 0")
    ox, oy = origin
    seen: set[tuple[int, int]] = set()
    if len(projected) == 1:
        segments = [(projected[0], projected[0])]
    else:
        segments = list(zip(projected, projected[1:]))
    for (x0, y0), (x1, y1) in segments:
        for x, y in _densify(x0, y0, x1, y1, cell_m / 2.0):
            seen.add((math.floor((x - ox) / cell_m), math.floor((y - oy) / cell_m)))
    return sorted(seen)


def select_cells(
    points: list[TrackPoint],
    *,
    cell_m: float = 1000.0,
    origin: tuple[float, float] = (0.0, 0.0),
    srs: str = "EPSG:2180",
    name_stem: str = "track",
) -> list[Cell]:
    """Select the fixed-grid cells a track crosses (no buffer)."""
    ox, oy = origin
    projected = [_transform("EPSG:4326", srs, p.lon, p.lat) for p in points]
    indices = _cells_from_projected(projected, cell_m=cell_m, origin=origin)
    cells: list[Cell] = []
    for ix, iy in indices:
        xmin = ix * cell_m + ox
        ymin = iy * cell_m + oy
        xmax = xmin + cell_m
        ymax = ymin + cell_m
        clon, clat = _transform(srs, "EPSG:4326", (xmin + xmax) / 2.0, (ymin + ymax) / 2.0)
        a_lon, a_lat = _transform(srs, "EPSG:4326", xmin, ymin)
        b_lon, b_lat = _transform(srs, "EPSG:4326", xmax, ymax)
        cells.append(
            Cell(
                ix=ix,
                iy=iy,
                bbox_2180=(xmin, ymin, xmax, ymax),
                bbox_wgs84=(min(a_lon, b_lon), min(a_lat, b_lat), max(a_lon, b_lon), max(a_lat, b_lat)),
                center_lat=clat,
                center_lon=clon,
                name=f"{name_stem}_x{ix}_y{iy}",
            )
        )
    return cells
