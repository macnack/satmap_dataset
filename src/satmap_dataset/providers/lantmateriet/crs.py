"""CRS helpers for the Lantmäteriet provider.

Prefers `pyproj` when installed, falls back to the `proj` CLI to keep the
runtime dependency surface small (matches the existing pattern in cli.py).
"""

from __future__ import annotations

import subprocess


def transform_point(
    src_crs: str,
    dst_crs: str,
    x: float,
    y: float,
    *,
    always_xy: bool = True,
) -> tuple[float, float]:
    try:
        from pyproj import Transformer

        transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=always_xy)
        out_x, out_y = transformer.transform(x, y)
        return float(out_x), float(out_y)
    except ImportError:
        return _transform_point_proj_cli(src_crs, dst_crs, x, y, always_xy=always_xy)


def transform_bbox_wgs84_to(
    dst_crs: str,
    *,
    center_lat: float,
    center_lon: float,
    size_meters: float,
) -> tuple[float, float, float, float]:
    """Project a WGS84 lat/lon center to `dst_crs`, then build a square bbox of
    side `size_meters` around it. Returns (minx, miny, maxx, maxy)."""
    if size_meters <= 0:
        raise ValueError("size_meters must be > 0")
    cx, cy = transform_point("EPSG:4326", dst_crs, center_lon, center_lat)
    half = size_meters / 2.0
    return (cx - half, cy - half, cx + half, cy + half)


_PROJECTED_DEFS = {
    "EPSG:3006": [
        # SWEREF 99 TM (Sweden)
        "+proj=tmerc",
        "+lat_0=0",
        "+lon_0=15",
        "+k=0.9996",
        "+x_0=500000",
        "+y_0=0",
        "+ellps=GRS80",
        "+units=m",
        "+no_defs",
    ],
    "EPSG:2180": [
        # ETRS89 / Poland CS92
        "+proj=tmerc",
        "+lat_0=0",
        "+lon_0=19",
        "+k=0.9993",
        "+x_0=500000",
        "+y_0=-5300000",
        "+ellps=GRS80",
        "+units=m",
        "+no_defs",
    ],
    "EPSG:3067": [
        # ETRS89 / TM35FIN (Finland) — same TM parameters as UTM 35N but
        # GRS80 datum.
        "+proj=tmerc",
        "+lat_0=0",
        "+lon_0=27",
        "+k=0.9996",
        "+x_0=500000",
        "+y_0=0",
        "+ellps=GRS80",
        "+units=m",
        "+no_defs",
    ],
}


def _utm_defs_for_epsg(code: int) -> list[str] | None:
    """Return PROJ defs for WGS84 UTM zones (EPSG:32601–32660 N, 32701–32760 S)."""
    if 32601 <= code <= 32660:
        zone = code - 32600
        return ["+proj=utm", f"+zone={zone}", "+ellps=WGS84", "+datum=WGS84", "+units=m", "+no_defs"]
    if 32701 <= code <= 32760:
        zone = code - 32700
        return [
            "+proj=utm",
            f"+zone={zone}",
            "+south",
            "+ellps=WGS84",
            "+datum=WGS84",
            "+units=m",
            "+no_defs",
        ]
    return None


def _projected_defs(epsg_label: str) -> list[str] | None:
    if epsg_label in _PROJECTED_DEFS:
        return _PROJECTED_DEFS[epsg_label]
    if epsg_label.startswith("EPSG:"):
        try:
            return _utm_defs_for_epsg(int(epsg_label.split(":", 1)[1]))
        except ValueError:
            return None
    return None


def _transform_point_proj_cli(
    src_crs: str,
    dst_crs: str,
    x: float,
    y: float,
    *,
    always_xy: bool,
) -> tuple[float, float]:
    src_upper = src_crs.upper()
    dst_upper = dst_crs.upper()
    src_defs = _projected_defs(src_upper)
    dst_defs = _projected_defs(dst_upper)
    if src_upper == "EPSG:4326" and dst_defs is not None:
        forward = True
        defs = dst_defs
    elif dst_upper == "EPSG:4326" and src_defs is not None:
        forward = False
        defs = src_defs
    else:
        raise RuntimeError(
            "pyproj is required for general CRS transforms; install pyproj or "
            "restrict to lat/lon <-> EPSG:3006 / EPSG:2180 / EPSG:326NN (UTM N) "
            "/ EPSG:327NN (UTM S)."
        )

    args = ["proj", *defs] if forward else ["proj", "-I", *defs]
    completed = subprocess.run(
        args,
        input=f"{x} {y}\n",
        text=True,
        capture_output=True,
        check=True,
    )
    parts = completed.stdout.strip().split()
    if len(parts) < 2:
        raise RuntimeError("PROJ CLI did not return two coordinates")
    if not forward:
        # `proj -I` may emit DMS-formatted angles like 21d44'19.277"E.
        return _parse_proj_angle(parts[0]), _parse_proj_angle(parts[1])
    return float(parts[0]), float(parts[1])


def _parse_proj_angle(value: str) -> float:
    text = value.strip()
    sign = 1.0
    if text and text[-1] in {"N", "E"}:
        text = text[:-1]
    elif text and text[-1] in {"S", "W"}:
        text = text[:-1]
        sign = -1.0
    if "d" not in text:
        return sign * float(text)
    degrees, _, rest = text.partition("d")
    minutes_part, _, seconds_part = rest.partition("'")
    seconds = seconds_part.replace('"', "").strip()
    deg = float(degrees) if degrees else 0.0
    minutes = float(minutes_part) if minutes_part else 0.0
    secs = float(seconds) if seconds else 0.0
    return sign * (deg + minutes / 60.0 + secs / 3600.0)
