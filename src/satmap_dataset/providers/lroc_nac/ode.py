"""Minimal ODE (Orbital Data Explorer) REST client for LROC NAC products.

Network code lives in `search_products`; everything else is pure parsing so
tests run against fixture JSON without hitting the network. ODE returns
`ODEResults.Products.Product` as a list, or a bare dict when a single product
matches — the parser normalizes both.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
import logging
import random
from typing import Any, Iterable
from urllib.parse import urlencode

import httpx

from satmap_dataset.geoportal.http import RetryPolicy

logger = logging.getLogger("satmap_dataset.lroc_nac.ode")


@dataclass(frozen=True)
class OdeProduct:
    pdsid: str
    observation_time: str | None
    acquisition_year: int | None
    incidence_angle: float | None
    emission_angle: float | None
    map_resolution: float | None
    footprint_bbox: tuple[float, float, float, float] | None
    file_url: str | None
    file_bytes: int | None


@dataclass
class OdeSearchOptions:
    url: str = "https://oderest.rsl.wustl.edu/live2/"
    product_type: str = "CDRNAC4"
    loc: str = "f"
    results: str = "opmf"
    limit: int = 100
    max_pages: int = 20


def build_query_url(
    base: str,
    *,
    product_type: str,
    westlon: float,
    eastlon: float,
    minlat: float,
    maxlat: float,
    loc: str = "f",
    min_obtime: str | None = None,
    max_obtime: str | None = None,
    results: str = "opmf",
    limit: int = 100,
    offset: int = 0,
) -> str:
    params: list[tuple[str, str]] = [
        ("query", "product"),
        ("target", "moon"),
        ("ihid", "LRO"),
        ("iid", "LROC"),
        ("pt", product_type),
        ("westernlon", _fmt(westlon)),
        ("easternlon", _fmt(eastlon)),
        ("minlat", _fmt(minlat)),
        ("maxlat", _fmt(maxlat)),
        ("loc", loc),
        ("results", results),
        ("limit", str(limit)),
        ("offset", str(offset)),
        ("output", "JSON"),
    ]
    if min_obtime:
        params.append(("minobtime", min_obtime))
    if max_obtime:
        params.append(("maxobtime", max_obtime))
    return f"{base}?{urlencode(params)}"


def _fmt(value: float) -> str:
    # Avoid trailing ".0" noise but keep fractional precision (e.g. 30.6, 20.0).
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text or "0"


def _to_float(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _parse_year(value: str | None) -> int | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).year
    except ValueError:
        try:
            return datetime.strptime(value[:10], "%Y-%m-%d").year
        except ValueError:
            return None


def _select_file(record: dict[str, Any]) -> tuple[str | None, int | None]:
    files_block = record.get("Product_files") or {}
    raw_files = _as_list(files_block.get("Product_file")) if isinstance(files_block, dict) else []
    # Prefer Type == "Product" with an image extension; fall back to first Product.
    best: dict[str, Any] | None = None
    for entry in raw_files:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("Type", "")).lower() != "product":
            continue
        name = str(entry.get("FileName", "")).lower()
        if name.endswith((".img", ".tif", ".tiff", ".cub")):
            best = entry
            break
        if best is None:
            best = entry
    if best is None:
        return None, None
    url = best.get("URL")
    kbytes = _to_float(best.get("KBytes"))
    file_bytes = int(kbytes * 1024) if kbytes is not None else None
    return (str(url) if url else None), file_bytes


def _footprint_bbox(record: dict[str, Any]) -> tuple[float, float, float, float] | None:
    west = _to_float(record.get("Westernmost_longitude"))
    east = _to_float(record.get("Easternmost_longitude"))
    south = _to_float(record.get("Minimum_latitude"))
    north = _to_float(record.get("Maximum_latitude"))
    if None in (west, east, south, north):
        return None
    return (west, south, east, north)  # type: ignore[return-value]


def parse_product(record: dict[str, Any]) -> OdeProduct:
    obs_time = record.get("Observation_time") or record.get("UTC_start_time")
    obs_time = str(obs_time) if obs_time else None
    url, file_bytes = _select_file(record)
    return OdeProduct(
        pdsid=str(record.get("pdsid") or ""),
        observation_time=obs_time,
        acquisition_year=_parse_year(obs_time),
        incidence_angle=_to_float(record.get("Incidence_angle")),
        emission_angle=_to_float(record.get("Emission_angle")),
        map_resolution=_to_float(record.get("Map_resolution")),
        footprint_bbox=_footprint_bbox(record),
        file_url=url,
        file_bytes=file_bytes,
    )


def parse_products(payload: dict[str, Any]) -> list[OdeProduct]:
    results = payload.get("ODEResults") or {}
    products_block = results.get("Products") or {}
    if not isinstance(products_block, dict):
        return []
    records = _as_list(products_block.get("Product"))
    return [parse_product(r) for r in records if isinstance(r, dict)]


def group_products_by_year(products: Iterable[OdeProduct]) -> dict[int, list[OdeProduct]]:
    grouped: dict[int, list[OdeProduct]] = {}
    for product in products:
        if product.acquisition_year is None or product.file_url is None:
            continue
        grouped.setdefault(product.acquisition_year, []).append(product)
    return grouped


async def search_products(
    options: OdeSearchOptions,
    *,
    westlon: float,
    eastlon: float,
    minlat: float,
    maxlat: float,
    min_obtime: str | None,
    max_obtime: str | None,
    timeout: float = 60.0,
    retry_policy: RetryPolicy | None = None,
    client: httpx.AsyncClient | None = None,
) -> list[OdeProduct]:
    """Page ODE by offset until a short page or `max_pages`. Returns parsed products."""
    owns_client = client is None
    active = client or httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers={"User-Agent": "satmap_dataset/0.1"})
    policy = retry_policy or RetryPolicy()
    all_products: list[OdeProduct] = []
    try:
        for page in range(options.max_pages):
            url = build_query_url(
                options.url,
                product_type=options.product_type,
                westlon=westlon, eastlon=eastlon, minlat=minlat, maxlat=maxlat,
                loc=options.loc, min_obtime=min_obtime, max_obtime=max_obtime,
                results=options.results, limit=options.limit, offset=page * options.limit,
            )
            logger.info("ODE GET %s", url)
            response = None
            for attempt in range(1, policy.max_attempts + 1):
                await asyncio.sleep(random.uniform(0, policy.jitter_seconds))
                try:
                    response = await active.get(url)
                    if response.status_code in policy.retry_for_statuses and attempt < policy.max_attempts:
                        await asyncio.sleep(policy.backoff_seconds * attempt)
                        continue
                    response.raise_for_status()
                    break
                except httpx.HTTPError:
                    if attempt >= policy.max_attempts:
                        raise
            assert response is not None
            products = parse_products(response.json())
            all_products.extend(products)
            if len(products) < options.limit:
                break
        return all_products
    finally:
        if owns_client:
            await active.aclose()
