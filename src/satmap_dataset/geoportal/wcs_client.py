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
