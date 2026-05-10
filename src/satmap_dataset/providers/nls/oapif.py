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


class OapifParseError(ValueError):
    """Raised when an OAPIF response cannot be interpreted as GeoJSON features.

    The provider distinguishes this from a successful response with zero
    features so that a 200 HTML error page or truncated body falls back to
    the WCS-wide year list with a warning, instead of being mistaken for
    "AOI has no orthophoto coverage for any year".
    """


def parse_aoi_years(features_geojson: bytes) -> set[int]:
    """Return the set of integer years from `kuvausvuosi` across all features.

    Raises OapifParseError if the payload is not parseable JSON or doesn't
    look like a FeatureCollection. An empty `features` list is *not* an
    error and returns the empty set.
    """
    data = _parse_geojson(features_geojson)
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


def parse_next_link(features_geojson: bytes) -> str | None:
    """Extract the `rel='next'` link href from an OAPIF FeatureCollection page.

    Returns None when there are no more pages. Raises OapifParseError on
    malformed payloads (the same fallback path the provider already uses).
    """
    data = _parse_geojson(features_geojson)
    for link in data.get("links", []) or []:
        if not isinstance(link, dict):
            continue
        if (link.get("rel") or "").lower() != "next":
            continue
        # Prefer GeoJSON over HTML when both are advertised.
        link_type = (link.get("type") or "").lower()
        if link_type and "geo+json" not in link_type and "json" not in link_type:
            continue
        href = link.get("href")
        if isinstance(href, str) and href:
            return href
    return None


def _parse_geojson(features_geojson: bytes) -> dict:
    try:
        data = json.loads(features_geojson)
    except (ValueError, json.JSONDecodeError) as exc:
        raise OapifParseError(f"OAPIF response is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or "features" not in data:
        raise OapifParseError(
            "OAPIF response missing 'features' array (likely an HTML error page or unexpected schema)"
        )
    return data
