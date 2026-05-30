from __future__ import annotations

import math
from typing import Any

from satmap_dataset.geoportal.http import RetryPolicy, request_with_retry

DEFAULT_ENDPOINTS = {
    "nmt": "https://mapy.geoportal.gov.pl/wss/service/PZGIK/NMT/GRID1/WCS/DigitalTerrainModelFormatTIFF",
    "nmpt": "https://mapy.geoportal.gov.pl/wss/service/PZGIK/NMPT/GRID1/WCS/DigitalSurfaceModelFormatTIFF",
}
_PRODUCT_PREFIX = {"nmt": "DTM", "nmpt": "DSM"}
_DATUM_TOKEN = {"evrf2007": "PL-EVRF2007-NH", "kron86": "PL-KRON86-NH"}


def endpoint_url(product: str, options: dict[str, Any] | None = None) -> str:
    options = options or {}
    overrides = options.get("endpoints") or {}
    if product in overrides:
        return str(overrides[product])
    if product not in DEFAULT_ENDPOINTS:
        raise ValueError(f"Unknown product {product!r}; expected one of {sorted(DEFAULT_ENDPOINTS)}")
    return DEFAULT_ENDPOINTS[product]


def coverage_id(product: str, datum: str, options: dict[str, Any] | None = None) -> str:
    options = options or {}
    template = str(options.get("coverage_id_template", "{prefix}_{datum}_TIFF"))
    if product not in _PRODUCT_PREFIX:
        raise ValueError(f"Unknown product {product!r}; expected one of {sorted(_PRODUCT_PREFIX)}")
    if datum not in _DATUM_TOKEN:
        raise ValueError(f"Unknown datum {datum!r}; expected one of {sorted(_DATUM_TOKEN)}")
    return template.format(prefix=_PRODUCT_PREFIX[product], datum=_DATUM_TOKEN[datum])


def split_bbox(
    bbox: tuple[float, float, float, float],
    max_request_px: int,
    gsd_m: float = 1.0,
) -> list[tuple[float, float, float, float]]:
    """Split an AOI bbox into non-overlapping sub-bboxes, each at most
    ``max_request_px`` pixels per side at the given ground sample distance."""
    if max_request_px < 1:
        raise ValueError("max_request_px must be >= 1")
    if gsd_m <= 0:
        raise ValueError("gsd_m must be > 0")
    xmin, ymin, xmax, ymax = bbox
    span_m = max_request_px * gsd_m
    nx = max(1, math.ceil((xmax - xmin) / span_m))
    ny = max(1, math.ceil((ymax - ymin) / span_m))
    tiles: list[tuple[float, float, float, float]] = []
    for iy in range(ny):
        y0 = ymin + iy * span_m
        y1 = min(ymax, y0 + span_m)
        for ix in range(nx):
            x0 = xmin + ix * span_m
            x1 = min(xmax, x0 + span_m)
            tiles.append((x0, y0, x1, y1))
    return tiles


async def get_coverage(
    endpoint: str,
    coverage_id_value: str,
    sub_bbox: tuple[float, float, float, float],
    srs: str,
    *,
    options: dict[str, Any] | None = None,
    timeout: float = 120.0,
    retry_policy: RetryPolicy | None = None,
    client: Any | None = None,
) -> bytes:
    """Issue a WCS 2.0.1 GetCoverage and return the GeoTIFF bytes.

    Axis labels for the SUBSET parameters default to ``"x"``/``"y"``. WCS 2.0.1
    servers can be strict about matching the SUBSET axis names to the coverage's
    CRS axis names — for EPSG:2180 some servers expect ``"E"``/``"N"`` instead.
    If GetCoverage requests are rejected with an axis/subset error, override the
    labels via ``options={"axis_label_x": "E", "axis_label_y": "N"}`` (and, if
    needed, ``options["subsetting_crs"]`` / ``options["format"]`` /
    ``options["wcs_version"]``). The correct values for the GUGiK NMT/NMPT WCS
    must be confirmed against the live service.
    """
    options = options or {}
    axis_x = str(options.get("axis_label_x", "x"))
    axis_y = str(options.get("axis_label_y", "y"))
    fmt = str(options.get("format", "image/tiff"))
    epsg = srs.split(":")[-1]
    subsetting_crs = str(
        options.get("subsetting_crs", f"http://www.opengis.net/def/crs/EPSG/0/{epsg}")
    )
    xmin, ymin, xmax, ymax = sub_bbox
    params: dict[str, Any] = {
        "SERVICE": "WCS",
        "VERSION": str(options.get("wcs_version", "2.0.1")),
        "REQUEST": "GetCoverage",
        "COVERAGEID": coverage_id_value,
        "FORMAT": fmt,
        "SUBSETTINGCRS": subsetting_crs,
        "SUBSET": [f"{axis_x}({xmin},{xmax})", f"{axis_y}({ymin},{ymax})"],
    }
    response = await request_with_retry(
        "GET",
        endpoint,
        params=params,
        timeout=timeout,
        retry_policy=retry_policy,
        client=client,
    )
    return response.content
