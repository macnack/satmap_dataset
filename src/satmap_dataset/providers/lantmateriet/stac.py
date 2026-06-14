"""Minimal STAC search client and item parsing for Lantmäteriet orthophotos.

Network code lives in `search_features` and uses the same retry policy as the
geoportal HTTP layer. Everything else is pure parsing so tests can run against
fixture JSON without hitting the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import Any, Iterable, Sequence

import httpx

from satmap_dataset.geoportal.http import RetryPolicy, request_with_retry

logger = logging.getLogger("satmap_dataset.lantmateriet.stac")

# Asset selection priority for downloadable orthophoto assets.
# Earlier entries beat later ones.
_ASSET_TYPE_PRIORITY: tuple[str, ...] = (
    "image/tiff; application=geotiff; profile=cloud-optimized",
    "image/tiff; application=geotiff",
    "image/tiff",
    "image/geotiff",
    "image/vnd.stac.geotiff",
)

_ASSET_ROLE_PRIORITY: tuple[str, ...] = ("data", "visual", "overview")


@dataclass(frozen=True)
class StacAsset:
    key: str
    href: str
    media_type: str | None
    roles: tuple[str, ...]


@dataclass(frozen=True)
class StacItem:
    item_id: str
    collection: str | None
    datetime_iso: str | None
    year: int | None
    bbox: tuple[float, float, float, float] | None
    proj_bbox: tuple[float, float, float, float] | None
    epsg: int | None
    gsd: float | None
    license: str | None
    attribution: str | None
    spectral_type: str | None
    cloud_cover: float | None
    assets: tuple[StacAsset, ...]


@dataclass
class StacSearchOptions:
    url: str
    collections: Sequence[str] = field(default_factory=tuple)
    api_key: str | None = None
    page_limit: int = 100
    max_pages: int = 50


def parse_iso_year(value: str | None) -> int | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        # Items sometimes carry a date-only string.
        try:
            parsed = datetime.strptime(value[:10], "%Y-%m-%d")
        except ValueError:
            return None
    return parsed.year


def _extract_epsg(properties: dict[str, Any]) -> int | None:
    candidates = (
        properties.get("proj:epsg"),
        properties.get("epsg"),
        properties.get("proj:code"),
    )
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, int):
            return candidate
        if isinstance(candidate, str):
            digits = "".join(ch for ch in candidate if ch.isdigit())
            if digits:
                try:
                    return int(digits)
                except ValueError:
                    continue
    return None


def _extract_year(properties: dict[str, Any]) -> int | None:
    for key in ("acquisition_year", "year"):
        value = properties.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    for key in ("datetime", "start_datetime", "acquisition_date", "created"):
        year = parse_iso_year(properties.get(key))
        if year is not None:
            return year
    return None


def _extract_gsd(properties: dict[str, Any]) -> float | None:
    raw = properties.get("gsd")
    if raw is None:
        raw = properties.get("upplosning")
    return _to_float(raw)


def _extract_spectral_type(properties: dict[str, Any]) -> str | None:
    value = properties.get("spektraltyp") or properties.get("spectral_type")
    if isinstance(value, str):
        return value.strip().lower() or None
    return None


def _normalize_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        coords = tuple(float(v) for v in value)
    except (TypeError, ValueError):
        return None
    return coords  # type: ignore[return-value]


def _coerce_assets(raw_assets: dict[str, Any]) -> tuple[StacAsset, ...]:
    items: list[StacAsset] = []
    for key, payload in raw_assets.items():
        if not isinstance(payload, dict):
            continue
        href = payload.get("href")
        if not isinstance(href, str) or not href:
            continue
        media_type = payload.get("type")
        roles_raw = payload.get("roles") or []
        roles = tuple(str(r) for r in roles_raw if isinstance(r, (str, bytes)))
        items.append(
            StacAsset(
                key=str(key),
                href=href,
                media_type=str(media_type) if isinstance(media_type, str) else None,
                roles=roles,
            )
        )
    return tuple(items)


def parse_stac_item(payload: dict[str, Any]) -> StacItem:
    """Parse a single STAC Item payload into a StacItem."""
    properties = payload.get("properties") or {}
    if not isinstance(properties, dict):
        properties = {}
    assets = payload.get("assets") or {}
    if not isinstance(assets, dict):
        assets = {}
    item_id = str(payload.get("id") or "")
    collection = payload.get("collection")
    return StacItem(
        item_id=item_id,
        collection=str(collection) if isinstance(collection, str) else None,
        datetime_iso=str(properties.get("datetime")) if properties.get("datetime") else None,
        year=_extract_year(properties),
        bbox=_normalize_bbox(payload.get("bbox")),
        proj_bbox=_normalize_bbox(properties.get("proj:bbox")),
        epsg=_extract_epsg(properties),
        gsd=_extract_gsd(properties),
        license=_to_str(properties.get("license")),
        attribution=_to_str(properties.get("attribution") or properties.get("license_attribution")),
        spectral_type=_extract_spectral_type(properties),
        cloud_cover=_to_float(properties.get("eo:cloud_cover")),
        assets=_coerce_assets(assets),
    )


def parse_stac_features(payload: dict[str, Any]) -> list[StacItem]:
    """Parse the FeatureCollection returned by /search."""
    features = payload.get("features") or []
    if not isinstance(features, list):
        return []
    parsed: list[StacItem] = []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        parsed.append(parse_stac_item(feature))
    return parsed


def select_asset(item: StacItem) -> StacAsset | None:
    """Pick the best downloadable raster asset from a STAC item.

    Priority: media-type (COG > GeoTIFF > tiff) → role (data > visual > overview)
    → href ending (`.tif` / `.tiff`). Returns None when nothing matches.
    """
    if not item.assets:
        return None

    def media_score(asset: StacAsset) -> int:
        if asset.media_type is None:
            return len(_ASSET_TYPE_PRIORITY)
        normalized = asset.media_type.strip().lower()
        for index, candidate in enumerate(_ASSET_TYPE_PRIORITY):
            if candidate.lower() in normalized:
                return index
        if normalized.endswith("tiff") or normalized.endswith("tif"):
            return len(_ASSET_TYPE_PRIORITY) - 1
        return len(_ASSET_TYPE_PRIORITY)

    def role_score(asset: StacAsset) -> int:
        roles = {role.lower() for role in asset.roles}
        for index, candidate in enumerate(_ASSET_ROLE_PRIORITY):
            if candidate in roles:
                return index
        return len(_ASSET_ROLE_PRIORITY)

    def href_score(asset: StacAsset) -> int:
        href = asset.href.lower()
        if href.endswith((".tif", ".tiff")):
            return 0
        return 1

    candidates = [a for a in item.assets if (media_score(a) < len(_ASSET_TYPE_PRIORITY)) or href_score(a) == 0]
    pool = candidates or list(item.assets)
    pool.sort(key=lambda asset: (media_score(asset), role_score(asset), href_score(asset), asset.key))
    return pool[0]


def group_items_by_year(items: Iterable[StacItem]) -> dict[int, list[StacItem]]:
    grouped: dict[int, list[StacItem]] = {}
    for item in items:
        if item.year is None:
            continue
        grouped.setdefault(item.year, []).append(item)
    return grouped


def pick_item_for_bbox(items: Sequence[StacItem], bbox: tuple[float, float, float, float]) -> StacItem | None:
    """Pick the item whose footprint best covers `bbox`.

    Ranking: largest intersection area first; ties broken by IoU (so a tighter
    footprint that fully contains the target beats a sprawling one with the
    same intersection area), then smallest GSD, then item_id.
    Returns None for empty input.
    """
    if not items:
        return None

    target_area = max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

    def overlap_area(b: tuple[float, float, float, float] | None) -> float:
        if b is None:
            return 0.0
        min_x = max(b[0], bbox[0])
        min_y = max(b[1], bbox[1])
        max_x = min(b[2], bbox[2])
        max_y = min(b[3], bbox[3])
        if min_x >= max_x or min_y >= max_y:
            return 0.0
        return (max_x - min_x) * (max_y - min_y)

    def iou(b: tuple[float, float, float, float] | None) -> float:
        if b is None:
            return 0.0
        item_area = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
        union = target_area + item_area - overlap_area(b)
        if union <= 0.0:
            return 0.0
        return overlap_area(b) / union

    ranked = sorted(
        items,
        key=lambda it: (
            -overlap_area(it.bbox),
            -iou(it.bbox),
            it.gsd if it.gsd is not None else float("inf"),
            it.item_id,
        ),
    )
    return ranked[0]


def _to_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _to_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


async def search_features(
    options: StacSearchOptions,
    *,
    bbox: tuple[float, float, float, float] | None,
    bbox_crs: str | None,
    datetime_range: str | None,
    timeout: float = 60.0,
    retry_policy: RetryPolicy | None = None,
    client: httpx.AsyncClient | None = None,
    extra_body: dict[str, Any] | None = None,
) -> list[StacItem]:
    """POST /search against a STAC API and return parsed items.

    Pages through links[rel="next"] up to options.max_pages. `extra_body`
    is shallow-merged into the POST body — useful for STAC `query`
    extensions (e.g. ``{"query": {"eo:cloud_cover": {"lt": 20}}}``).
    """
    body: dict[str, Any] = {"limit": options.page_limit}
    if options.collections:
        body["collections"] = list(options.collections)
    if bbox is not None:
        body["bbox"] = list(bbox)
        if bbox_crs:
            body["bbox-crs"] = bbox_crs
    if datetime_range:
        body["datetime"] = datetime_range
    if extra_body:
        for key, value in extra_body.items():
            body[key] = value

    headers = {"Content-Type": "application/json", "Accept": "application/geo+json"}
    if options.api_key:
        headers["Authorization"] = f"Bearer {options.api_key}"

    owns_client = client is None
    active_client = client or httpx.AsyncClient(timeout=timeout, headers={"User-Agent": "satmap_dataset/0.1"})
    try:
        url = options.url
        method = "POST"
        all_items: list[StacItem] = []
        for _ in range(options.max_pages):
            logger.info("STAC: %s %s body_keys=%s", method, url, sorted(body.keys()))
            policy = retry_policy or RetryPolicy()
            for attempt in range(1, policy.max_attempts + 1):
                try:
                    if method == "POST":
                        response = await active_client.post(url, json=body, headers=headers)
                    else:
                        response = await active_client.get(url, headers=headers)
                    if response.status_code in policy.retry_for_statuses and attempt < policy.max_attempts:
                        logger.warning("STAC retryable status=%s attempt=%s", response.status_code, attempt)
                        continue
                    response.raise_for_status()
                    break
                except httpx.HTTPError:
                    if attempt >= policy.max_attempts:
                        raise
            payload = response.json()
            items = parse_stac_features(payload)
            all_items.extend(items)
            next_url = _next_page_url(payload)
            if not next_url:
                break
            url = next_url
            method = "GET"
            body = {}
        return all_items
    finally:
        if owns_client:
            await active_client.aclose()


def _next_page_url(payload: dict[str, Any]) -> str | None:
    links = payload.get("links") or []
    if not isinstance(links, list):
        return None
    for link in links:
        if not isinstance(link, dict):
            continue
        if str(link.get("rel", "")).lower() != "next":
            continue
        href = link.get("href")
        if isinstance(href, str) and href:
            return href
    return None
