from __future__ import annotations

import csv
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
