from __future__ import annotations

from typing import Any

from pyproj import Transformer

from satmap_dataset.geoportal.http import RetryPolicy, request_with_retry

CATEGORY_FILTERS: dict[str, str] = {
    "buildings": "building=* and type:way",
    "highways": "highway=* and type:way",
    "landuse": "landuse=* and type:way",
    "water": "(natural=water or waterway=*) and type:way",
}

_DEFAULT_OHSOME_URL = "https://api.ohsome.org/v1"


def bbox_to_wgs84(bbox_str: str, source_srs: str = "EPSG:2180") -> str:
    """Reproject a projected bbox to WGS84 lon_min,lat_min,lon_max,lat_max.

    `source_srs` may be any CRS pyproj can resolve (EPSG:2180 Poland, EPSG:3006
    Sweden, EPSG:3067 Finland, ...); axis order is normalised via always_xy.
    All four corners are transformed and min/max taken so the result stays a
    valid enclosing box even when the source projection is rotated relative to
    lon/lat.
    """
    xmin, ymin, xmax, ymax = (float(x) for x in bbox_str.split(","))
    t = Transformer.from_crs(source_srs, "EPSG:4326", always_xy=True)
    corners = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    lons, lats = zip(*(t.transform(x, y) for x, y in corners))
    return f"{min(lons):.6f},{min(lats):.6f},{max(lons):.6f},{max(lats):.6f}"


def bbox_epsg2180_to_wgs84(bbox_str: str) -> str:
    """Return ohsome-format bbox: lon_min,lat_min,lon_max,lat_max in WGS84.

    Backwards-compatible wrapper around :func:`bbox_to_wgs84` for EPSG:2180.
    """
    return bbox_to_wgs84(bbox_str, "EPSG:2180")


async def get_elements_geometry(
    bbox_wgs84: str,
    filter_str: str,
    snapshot_date: str,
    *,
    ohsome_url: str = _DEFAULT_OHSOME_URL,
    timeout: float = 60.0,
    retry_policy: RetryPolicy | None = None,
) -> dict[str, Any]:
    """Query ohsome /elements/geometry and return a GeoJSON FeatureCollection."""
    normalized_time = (
        snapshot_date if snapshot_date.endswith("Z") else snapshot_date + "T00:00:00Z"
    )
    url = ohsome_url.rstrip("/") + "/elements/geometry"
    payload: dict[str, str] = {
        "bboxes": bbox_wgs84,
        "filter": filter_str,
        "time": normalized_time,
        "clipGeometry": "false",
    }
    response = await request_with_retry(
        "POST",
        url,
        data=payload,
        timeout=timeout,
        retry_policy=retry_policy,
    )
    return response.json()
