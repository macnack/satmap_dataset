"""WGS84 map geometry and bbox helpers for the studio UI."""

from __future__ import annotations

import math
from typing import Any

from satmap_dataset.providers.lantmateriet.crs import transform_bbox_wgs84_to, transform_point


def utm_epsg_for_lon_lat(lon: float, lat: float) -> str:
    """Return the WGS84 UTM EPSG code for a lon/lat point."""
    zone = int((lon + 180.0) / 6.0) + 1
    if lat >= 0:
        return f"EPSG:{32600 + zone}"
    return f"EPSG:{32700 + zone}"


def square_side_meters(area_km2: float) -> float:
    if area_km2 <= 0:
        raise ValueError("area_km2 must be > 0")
    return math.sqrt(area_km2) * 1000.0


def format_bbox(bbox: tuple[float, float, float, float]) -> str:
    return ",".join(f"{value:.3f}" for value in bbox)


def bbox_from_center(
    center_lat: float,
    center_lon: float,
    area_km2: float,
    target_srs: str,
) -> tuple[str, tuple[float, float, float, float]]:
    """Build a square bbox string and tuple in ``target_srs``."""
    side_m = square_side_meters(area_km2)
    minx, miny, maxx, maxy = transform_bbox_wgs84_to(
        target_srs,
        center_lat=center_lat,
        center_lon=center_lon,
        size_meters=side_m,
    )
    bbox_tuple = (minx, miny, maxx, maxy)
    return format_bbox(bbox_tuple), bbox_tuple


def bbox_corners_wgs84(bbox: tuple[float, float, float, float], srs: str) -> list[tuple[float, float]]:
    """Return AOI rectangle corners as (lat, lon) for Folium."""
    minx, miny, maxx, maxy = bbox
    corners_xy = [
        (minx, miny),
        (minx, maxy),
        (maxx, maxy),
        (maxx, miny),
        (minx, miny),
    ]
    corners_latlon: list[tuple[float, float]] = []
    for x, y in corners_xy:
        lon, lat = transform_point(srs, "EPSG:4326", x, y)
        corners_latlon.append((lat, lon))
    return corners_latlon


def estimate_output_pixels(side_m: float, px_per_meter: float) -> tuple[int, int]:
    pixels = max(1, int(round(side_m * px_per_meter)))
    return pixels, pixels


def nominatim_search(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Search OpenStreetMap Nominatim for place names (online)."""
    import httpx

    response = httpx.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": query, "format": "json", "limit": limit},
        headers={"User-Agent": "satmap_dataset/0.1 studio"},
        timeout=10.0,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        return []
    return payload
