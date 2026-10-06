"""Polite async HTTP for Wayback: descriptive User-Agent, jittered sleeps, retries."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import re
from dataclasses import dataclass
from typing import Any

import httpx

from satmap_dataset.geoportal.http import RetryPolicy
from satmap_dataset.providers.esri_wayback.catalog import (
    DEFAULT_TILEMAP_URL_TEMPLATE,
    WaybackRelease,
)
from satmap_dataset.providers.esri_wayback.versions import TilemapResult

logger = logging.getLogger("satmap_dataset.esri_wayback")

DEFAULT_USER_AGENT = (
    "satmap_dataset/0.1 (+https://github.com/macnack/satmap_dataset; "
    "research dataset builder; provider=esri_wayback)"
)
_TERMINAL_STATUSES = frozenset({400, 401, 403, 404, 410})
_FINAL_RELEASE_RE = re.compile(r"/tile/(\d+)/", re.IGNORECASE)


class WaybackHTTPError(RuntimeError):
    pass


@dataclass
class PoliteClient:
    client: httpx.AsyncClient
    sleep_min: float = 0.2
    sleep_max: float = 0.6
    retry_policy: RetryPolicy = RetryPolicy(max_attempts=4)
    requests_made: int = 0

    async def get(self, url: str, *, params: dict[str, Any] | None = None) -> httpx.Response | None:
        """GET with pre-request jitter and exponential backoff.

        Returns ``None`` for terminal 4xx (e.g. a tile absent at that zoom);
        raises :class:`WaybackHTTPError` when retries are exhausted.
        """
        policy = self.retry_policy
        last_exc: Exception | None = None
        for attempt in range(1, policy.max_attempts + 1):
            if self.sleep_max > 0:
                await asyncio.sleep(random.uniform(self.sleep_min, self.sleep_max))
            self.requests_made += 1
            try:
                response = await self.client.get(url, params=params)
            except httpx.HTTPError as exc:
                last_exc = exc
                logger.warning("Wayback GET failed attempt=%s/%s url=%s: %s", attempt, policy.max_attempts, url, exc)
            else:
                if response.status_code < 400:
                    return response
                if response.status_code in _TERMINAL_STATUSES:
                    return None
                last_exc = WaybackHTTPError(f"HTTP {response.status_code} for {url}")
                logger.warning(
                    "Wayback GET status=%s attempt=%s/%s url=%s",
                    response.status_code,
                    attempt,
                    policy.max_attempts,
                    url,
                )
            if attempt < policy.max_attempts:
                backoff = policy.backoff_seconds * (2 ** (attempt - 1))
                await asyncio.sleep(backoff + random.uniform(0.0, policy.jitter_seconds))
        raise WaybackHTTPError(f"GET {url} failed after {policy.max_attempts} attempts: {last_exc}")

    async def get_json(self, url: str, *, params: dict[str, Any] | None = None) -> Any:
        response = await self.get(url, params=params)
        if response is None:
            raise WaybackHTTPError(f"GET {url} returned a terminal 4xx")
        try:
            return response.json()
        except ValueError as exc:
            raise WaybackHTTPError(f"GET {url} did not return JSON: {exc}") from exc

    async def tilemap(
        self,
        release: WaybackRelease,
        tile: tuple[int, int, int],
        *,
        template: str = DEFAULT_TILEMAP_URL_TEMPLATE,
    ) -> TilemapResult:
        z, x, y = tile
        payload = await self.get_json(template.format(release=release.release_num, z=z, y=y, x=x))
        if not isinstance(payload, dict):
            raise WaybackHTTPError("tilemap response is not a JSON object")
        return TilemapResult.from_json(payload)

    async def tile_digest(self, release: WaybackRelease, tile: tuple[int, int, int]) -> tuple[str, int | None] | None:
        """sha256 of the tile bytes plus the release number revealed by redirects."""
        z, x, y = tile
        response = await self.get(release.tile_url(z, x, y))
        if response is None:
            return None
        origin_match = _FINAL_RELEASE_RE.search(str(response.url))
        origin = int(origin_match.group(1)) if origin_match else None
        return hashlib.sha256(response.content).hexdigest(), origin

    async def metadata_query(
        self,
        release: WaybackRelease,
        *,
        layer_id: int,
        geometry: str,
        geometry_type: str,
    ) -> list[dict[str, Any]]:
        if not release.metadata_layer_url:
            raise WaybackHTTPError(f"release {release.release_num} has no metadata layer URL")
        payload = await self.get_json(
            f"{release.metadata_layer_url}/{layer_id}/query",
            params={
                "f": "json",
                "where": "1=1",
                "outFields": "SRC_DATE,SRC_RES,SRC_ACC,SRC_DESC,NICE_DESC,NICE_NAME",
                "returnGeometry": "false",
                "geometryType": geometry_type,
                "geometry": geometry,
                "inSR": "3857",
                "spatialRel": "esriSpatialRelIntersects",
            },
        )
        if not isinstance(payload, dict) or "error" in payload:
            raise WaybackHTTPError(f"metadata query error: {payload.get('error') if isinstance(payload, dict) else payload}")
        return list(payload.get("features") or [])


def metadata_layer_id_for_zoom(z: int) -> int:
    """Metadata sublayer for a zoom: 0 = 1.9 cm (z23) … 13 = 150 m (z10 and coarser)."""
    return max(0, min(13, 23 - int(z)))
