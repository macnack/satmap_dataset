from __future__ import annotations

import re
from typing import Any

from satmap_dataset.geoportal import wfs_client
from satmap_dataset.geoportal.http import RetryPolicy
from satmap_dataset.models import YearStatus

_BASE = "https://mapy.geoportal.gov.pl/wss/service/PZGIK"
_SERVICE = {
    ("nmt", "kron86"): "NumerycznyModelTerenuKRON86",
    ("nmt", "evrf2007"): "NumerycznyModelTerenuEVRF2007",
    ("nmpt", "kron86"): "NumerycznyModelPokryciaTerenuKRON86",
    ("nmpt", "evrf2007"): "NumerycznyModelPokryciaTerenuEVRF2007",
}
_TYPENAME_TOKEN = {"nmt": "NMT", "nmpt": "NMPT"}


def endpoint(product: str, datum: str, options: dict[str, Any] | None = None) -> str:
    options = options or {}
    overrides = options.get("skorowidz_endpoints") or {}
    key = f"{product}|{datum}"
    if key in overrides:
        return str(overrides[key])
    if (product, datum) not in _SERVICE:
        raise ValueError(
            f"Unknown (product, datum)=({product!r}, {datum!r}); "
            f"expected product in {sorted(_TYPENAME_TOKEN)} and datum in {{evrf2007, kron86}}."
        )
    return f"{_BASE}/{_SERVICE[(product, datum)]}/WFS/Skorowidze"


def typename_pattern(product: str) -> "re.Pattern[str]":
    if product not in _TYPENAME_TOKEN:
        raise ValueError(f"Unknown product {product!r}; expected one of {sorted(_TYPENAME_TOKEN)}")
    return re.compile(rf"Skorowidz{_TYPENAME_TOKEN[product]}(\d{{4}})", re.IGNORECASE)


async def year_typenames(
    product: str,
    datum: str,
    options: dict[str, Any] | None = None,
    *,
    timeout: float = 45.0,
    retry_policy: RetryPolicy | None = None,
) -> dict[int, str]:
    _root, mapping = await wfs_client.get_capabilities(
        base_url=endpoint(product, datum, options),
        timeout=timeout,
        retry_policy=retry_policy,
        typename_pattern=typename_pattern(product),
    )
    return mapping


async def tiles_for_year(
    product: str,
    datum: str,
    year: int,
    bbox: str,
    srs: str,
    *,
    year_to_typename: dict[int, str],
    options: dict[str, Any] | None = None,
    timeout: float = 45.0,
    retry_policy: RetryPolicy | None = None,
) -> tuple[YearStatus, dict[str, str], dict[str, list[float]], dict[str, dict[str, int | str | None]]]:
    return await wfs_client.get_year_tiles(
        year=year,
        bbox=bbox,
        srs=srs,
        base_url=endpoint(product, datum, options),
        timeout=timeout,
        retry_policy=retry_policy,
        year_to_typename=year_to_typename,
    )
