from __future__ import annotations

from typing import Any

from satmap_dataset.geoportal.http import RetryPolicy, request_with_retry

CATEGORY_QUERIES: dict[str, str] = {
    "buildings": 'way["building"]',
    "highways": 'way["highway"]',
    "landuse": 'way["landuse"]',
    "water": '(way["natural"="water"]; way["waterway"];)',
}

_DEFAULT_OVERPASS_URL = "https://overpass.kumi.systems/api/interpreter"


def _bbox_to_overpass(bbox_wgs84: str) -> str:
    """Convert lon_min,lat_min,lon_max,lat_max → Overpass S,W,N,E."""
    lon_min, lat_min, lon_max, lat_max = (float(x) for x in bbox_wgs84.split(","))
    return f"{lat_min},{lon_min},{lat_max},{lon_max}"


def _build_query(category: str, bbox_overpass: str, snapshot_date: str) -> str:
    clause = CATEGORY_QUERIES[category]
    date_tag = f'[date:"{snapshot_date}"]'
    return f'[out:json][timeout:55]{date_tag};({clause}({bbox_overpass}););out geom;'


def _ways_to_geojson(overpass_result: dict) -> dict[str, Any]:
    """Convert Overpass JSON (out geom) → GeoJSON FeatureCollection."""
    features = []
    for el in overpass_result.get("elements", []):
        if el.get("type") != "way":
            continue
        nodes = el.get("geometry", [])
        if len(nodes) < 2:
            continue
        coords = [[n["lon"], n["lat"]] for n in nodes]
        if coords[0] == coords[-1] and len(coords) >= 4:
            geometry: dict[str, Any] = {"type": "Polygon", "coordinates": [coords]}
        else:
            geometry = {"type": "LineString", "coordinates": coords}
        features.append({
            "type": "Feature",
            "geometry": geometry,
            "properties": el.get("tags", {}),
        })
    return {"type": "FeatureCollection", "features": features}


async def get_elements_geometry(
    bbox_wgs84: str,
    category: str,
    snapshot_date: str,
    *,
    overpass_url: str = _DEFAULT_OVERPASS_URL,
    timeout: float = 60.0,
    retry_policy: RetryPolicy | None = None,
) -> dict[str, Any]:
    """Query Overpass API at a historical snapshot and return GeoJSON FeatureCollection."""
    normalized = snapshot_date if snapshot_date.endswith("Z") else snapshot_date + "T00:00:00Z"
    bbox_overpass = _bbox_to_overpass(bbox_wgs84)
    query = _build_query(category, bbox_overpass, normalized)
    response = await request_with_retry(
        "POST",
        overpass_url,
        data={"data": query},
        timeout=timeout,
        retry_policy=retry_policy,
    )
    return _ways_to_geojson(response.json())
