#!/usr/bin/env python3
"""Print a square bbox around a WGS84 lat/lon center, projected into a target CRS.

Output sections:
  bbox:    minx,miny,maxx,maxy in the target CRS
  geojson: WGS84 polygon footprint
  command: a ready-to-paste `python -m satmap_dataset.cli run` invocation
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.providers.lantmateriet.crs import (  # noqa: E402
    transform_bbox_wgs84_to,
    transform_point,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--center-lat", type=float, required=True, help="Latitude in WGS84.")
    parser.add_argument("--center-lon", type=float, required=True, help="Longitude in WGS84.")
    parser.add_argument(
        "--size-meters", type=float, default=2000.0, help="Square AOI side length in meters."
    )
    parser.add_argument(
        "--target-srs",
        default="EPSG:3006",
        help="Target CRS for the bbox. Defaults to SWEREF 99 TM (Sweden).",
    )
    parser.add_argument(
        "--provider",
        default="lantmateriet",
        help="Provider used in the suggested CLI command.",
    )
    parser.add_argument(
        "--year-start",
        type=int,
        default=2010,
        help="Year range start in the suggested CLI command.",
    )
    parser.add_argument(
        "--year-end",
        type=int,
        default=2024,
        help="Year range end in the suggested CLI command.",
    )
    parser.add_argument(
        "--render-root",
        default="rendered",
        help="render-root path used in the suggested CLI command.",
    )
    return parser.parse_args()


def _format_bbox(bbox: tuple[float, float, float, float]) -> str:
    return ",".join(f"{value:.3f}" for value in bbox)


def _wgs84_polygon_for_target_bbox(
    bbox: tuple[float, float, float, float], target_srs: str
) -> dict:
    minx, miny, maxx, maxy = bbox
    corners_target = [
        (minx, miny),
        (maxx, miny),
        (maxx, maxy),
        (minx, maxy),
        (minx, miny),
    ]
    corners_wgs84 = [
        transform_point(target_srs, "EPSG:4326", x, y) for x, y in corners_target
    ]
    return {
        "type": "Polygon",
        "coordinates": [[[lon, lat] for lon, lat in corners_wgs84]],
    }


def main() -> int:
    args = _parse_args()
    bbox = transform_bbox_wgs84_to(
        args.target_srs,
        center_lat=args.center_lat,
        center_lon=args.center_lon,
        size_meters=args.size_meters,
    )
    bbox_str = _format_bbox(bbox)
    polygon = _wgs84_polygon_for_target_bbox(bbox, args.target_srs)

    cmd = [
        "python",
        "-m",
        "satmap_dataset.cli",
        "run",
        "--provider",
        args.provider,
        "--year-start",
        str(args.year_start),
        "--year-end",
        str(args.year_end),
        "--bbox",
        bbox_str,
        "--target-srs",
        args.target_srs,
        "--srs",
        args.target_srs,
        "--render-root",
        args.render_root,
        "--profile",
        "train",
    ]

    print(f"bbox: {bbox_str}")
    print(f"target_srs: {args.target_srs}")
    print("geojson:")
    print(json.dumps(polygon, indent=2))
    print("command:")
    print(" \\\n  ".join(shlex.quote(part) for part in cmd))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
