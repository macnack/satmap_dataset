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


def bbox_to_wgs84(bbox_str: str, srs: str) -> str:
    """Convert a projected bbox into ohsome/Overpass WGS84 lon/lat order.

    Returns lon_min,lat_min,lon_max,lat_max. `srs` may be any CRS pyproj can
    resolve (EPSG:2180 for Poland, EPSG:3006 for Sweden, EPSG:3067 for Finland,
    etc.); the source axis order is normalised via always_xy.
    """
    xmin, ymin, xmax, ymax = (float(x) for x in bbox_str.split(","))
    t = Transformer.from_crs(srs, "EPSG:4326", always_xy=True)
    lon_min, lat_min = t.transform(xmin, ymin)
    lon_max, lat_max = t.transform(xmax, ymax)
    return f"{lon_min:.6f},{lat_min:.6f},{lon_max:.6f},{lat_max:.6f}"


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
