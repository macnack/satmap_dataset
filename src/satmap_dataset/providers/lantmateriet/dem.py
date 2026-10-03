"""Lantmäteriet Markhöjdmodell (Swedish national DTM) via STAC-höjd.

This is intentionally separate from orthophoto STAC (`stac-bild`). Elevation
tiles live under `https://api.lantmateriet.se/stac-hojd/v1` (collection
`dtm-cog` by default). Catalog/search is public; asset download on
`dl1.lantmateriet.se` requires a Geotorget subscription for
**Markhöjdmodell Nedladdning** (credentials may differ from Ortofoto
Nedladdning).

License: CC BY 4.0 — attribute © Lantmäteriet.
"""

from __future__ import annotations

import logging
import os
from typing import Any
from urllib.parse import urlparse

from satmap_dataset.providers.lantmateriet import stac
from satmap_dataset.providers.lantmateriet.provider import (
    _basic_auth_header,
    _option,
)

logger = logging.getLogger("satmap_dataset.lantmateriet.dem")

# National 1 m Markhöjdmodell COG collection (not ortofoto stac-bild).
DEFAULT_STAC_HOJD_URL = "https://api.lantmateriet.se/stac-hojd/v1/search"
DEFAULT_STAC_HOJD_COLLECTION = "dtm-cog"
DEFAULT_ATTRIBUTION = "© Lantmäteriet"
DEFAULT_LICENSE = "CC-BY-4.0"


def resolve_dem_search_options(options: dict[str, Any]) -> stac.StacSearchOptions:
    """Resolve STAC-höjd search options (never the ortofoto stac-bild URL)."""
    url = _option(
        options,
        "stac_hojd_url",
        "SATMAP_LANTMATERIET_STAC_HOJD_URL",
        DEFAULT_STAC_HOJD_URL,
    )
    collection_value = _option(
        options,
        "stac_hojd_collection",
        "SATMAP_LANTMATERIET_STAC_HOJD_COLLECTION",
        DEFAULT_STAC_HOJD_COLLECTION,
    )
    if isinstance(collection_value, (list, tuple)):
        collections: tuple[str, ...] = tuple(str(c) for c in collection_value if c)
    else:
        collections = (str(collection_value),) if collection_value else (DEFAULT_STAC_HOJD_COLLECTION,)

    # Prefer DEM-specific credentials when set; fall back to shared Geotorget env.
    api_key = _option(options, "api_key", "SATMAP_LANTMATERIET_API_KEY", None)
    dem_user = os.environ.get("SATMAP_LANTMATERIET_DEM_USERNAME")
    dem_password = os.environ.get("SATMAP_LANTMATERIET_DEM_PASSWORD")
    dem_options = dict(options)
    if dem_user and dem_password:
        dem_options = {**options, "username": dem_user, "password": dem_password}
    authorization = _basic_auth_header(dem_options)
    if authorization is None and api_key:
        authorization = f"Bearer {api_key}"

    page_limit = int(options.get("page_limit", 100))
    max_pages = int(options.get("max_pages", 50))
    return stac.StacSearchOptions(
        url=str(url),
        collections=collections,
        api_key=str(api_key) if api_key else None,
        authorization=authorization,
        page_limit=page_limit,
        max_pages=max_pages,
    )


def auth_headers(options: dict[str, Any]) -> dict[str, str]:
    """Authorization headers for dl1.lantmateriet.se DEM asset downloads."""
    headers = {"User-Agent": "satmap_dataset/0.1"}
    search = resolve_dem_search_options(options)
    if search.authorization:
        headers["Authorization"] = search.authorization
    elif search.api_key:
        headers["Authorization"] = f"Bearer {search.api_key}"
    return headers


def select_dem_asset(item: stac.StacItem) -> stac.StacAsset | None:
    """Pick the elevation GeoTIFF/COG asset (role=data preferred)."""
    return stac.select_asset(item)


def filename_for_item(item: stac.StacItem, asset: stac.StacAsset) -> str:
    name = urlparse(asset.href).path.rsplit("/", 1)[-1] if asset.href else ""
    return name or f"{item.item_id}.tif"


def items_with_raster_assets(items: list[stac.StacItem]) -> list[tuple[stac.StacItem, stac.StacAsset]]:
    selected: list[tuple[stac.StacItem, stac.StacAsset]] = []
    for item in items:
        asset = select_dem_asset(item)
        if asset is None:
            logger.warning("STAC-höjd item %s has no downloadable raster asset", item.item_id)
            continue
        media = (asset.media_type or "").lower()
        href = asset.href.lower()
        if "tiff" not in media and "tif" not in media and not href.endswith((".tif", ".tiff")):
            # Skip point-cloud / non-raster assets if a broad collection is used.
            logger.warning(
                "Skipping non-raster STAC-höjd asset item=%s key=%s type=%s",
                item.item_id,
                asset.key,
                asset.media_type,
            )
            continue
        selected.append((item, asset))
    return selected
