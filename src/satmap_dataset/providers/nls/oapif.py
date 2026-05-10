"""OGC API Features client for NLS orthophoto metadata.

Used to find out which years actually have orthophoto coverage for a given
AOI, before any GetCoverage download. Each feature carries a `kuvausvuosi`
(year of photography) string; the union of those values is the per-AOI
year list. Without this check the WCS happily returns no-data tiles for
years it never flew that area.
"""

from __future__ import annotations

import json
from urllib.parse import urlencode


DEFAULT_OAPIF_URL = (
    "https://avoin-karttakuva.maanmittauslaitos.fi/ortokuvat-ja-korkeusmallit/features/v2"
)
DEFAULT_COLLECTION = "ortokuva_vari"
EPSG_3067_URI = "http://www.opengis.net/def/crs/EPSG/0/3067"


def build_items_url(
    base_url: str,
    *,
    collection: str,
    bbox: tuple[float, float, float, float],
    limit: int = 1000,
) -> str:
    xmin, ymin, xmax, ymax = bbox
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("bbox must satisfy xmin<xmax and ymin<ymax")
    params = [
        ("bbox", f"{xmin},{ymin},{xmax},{ymax}"),
        ("bbox-crs", EPSG_3067_URI),
        ("limit", str(int(limit))),
        ("f", "json"),
    ]
    return f"{base_url.rstrip('/')}/collections/{collection}/items?{urlencode(params)}"


def parse_aoi_years(features_geojson: bytes) -> set[int]:
    """Return the set of integer years from `kuvausvuosi` across all features."""
    try:
        data = json.loads(features_geojson)
    except (ValueError, json.JSONDecodeError):
        return set()
    years: set[int] = set()
    for feature in data.get("features", []):
        props = feature.get("properties", {}) or {}
        value = props.get("kuvausvuosi")
        if value is None:
            continue
        try:
            years.add(int(str(value)))
        except ValueError:
            continue
    return years
