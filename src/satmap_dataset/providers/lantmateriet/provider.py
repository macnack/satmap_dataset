"""Lantmäteriet (Sweden) orthophoto provider.

The provider talks to a STAC API by default and uses the same async download,
retry, and jitter machinery as the geoportal flow. Configuration comes from
`provider_options` on the run/index/download configs, with environment-variable
fallbacks for the bits that are typically secret or site-specific:

- SATMAP_LANTMATERIET_STAC_URL
- SATMAP_LANTMATERIET_STAC_COLLECTION
- SATMAP_LANTMATERIET_WMS_URL
- SATMAP_LANTMATERIET_WMS_LAYER
- SATMAP_LANTMATERIET_API_KEY
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import aiofiles
import httpx

from satmap_dataset.config import DownloadConfig, IndexConfig
from satmap_dataset.geoportal.http import RetryPolicy
from satmap_dataset.models import (
    DatasetManifest,
    IndexManifest,
    TileAcquisitionMetadata,
    YearAvailabilityReport,
    YearStatus,
)
from satmap_dataset.pipeline.validator import evaluate_year_policy
from satmap_dataset.providers.base import Provider
from satmap_dataset.providers.lantmateriet import crs, stac, year_policy, wms

DEFAULT_STAC_URL = "https://api.lantmateriet.se/stac-bild/v1/search"
DEFAULT_STAC_COLLECTION = ""  # search across all collections by bbox+datetime
# WMS fallback targets *Ortofoto Visning, Årsvisa* — a paid product (~166k SEK/month
# as of 2026). It is OFF by default; only enable via wms_fallback_missing_years.
DEFAULT_WMS_URL = ""
DEFAULT_WMS_LAYER = ""
DEFAULT_TARGET_SRS = "EPSG:3006"
# Lantmäteriet labels pre-2019 RGB items as "rgb" and post-2019 (4-band RGB+NIR
# COGs) as "rgbi". Both are valid color sources — the render stage drops the
# NIR band via _harmonize_rgb_u8, so default to accepting either.
DEFAULT_SPECTRAL_TYPES = ("rgb", "rgbi")

logger = logging.getLogger("satmap_dataset.lantmateriet")


def _option(options: dict[str, Any], key: str, env_var: str | None, default: Any) -> Any:
    if key in options and options[key] not in (None, ""):
        return options[key]
    if env_var:
        env_value = os.environ.get(env_var)
        if env_value:
            return env_value
    return default


def _resolve_search_options(options: dict[str, Any]) -> stac.StacSearchOptions:
    url = _option(options, "stac_url", "SATMAP_LANTMATERIET_STAC_URL", DEFAULT_STAC_URL)
    collection_value = _option(
        options,
        "stac_collection",
        "SATMAP_LANTMATERIET_STAC_COLLECTION",
        DEFAULT_STAC_COLLECTION,
    )
    if isinstance(collection_value, (list, tuple)):
        collections: tuple[str, ...] = tuple(str(c) for c in collection_value if c)
    else:
        collections = (str(collection_value),) if collection_value else ()
    api_key = _option(options, "api_key", "SATMAP_LANTMATERIET_API_KEY", None)
    page_limit = int(options.get("page_limit", 500))
    max_pages = int(options.get("max_pages", 50))
    return stac.StacSearchOptions(
        url=str(url),
        collections=collections,
        api_key=str(api_key) if api_key else None,
        page_limit=page_limit,
        max_pages=max_pages,
    )


def _basic_auth_header(options: dict[str, Any]) -> str | None:
    user = _option(options, "username", "SATMAP_LANTMATERIET_USERNAME", None)
    password = _option(options, "password", "SATMAP_LANTMATERIET_PASSWORD", None)
    if not user or not password:
        return None
    import base64

    token = base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


def _parse_bbox(value: str) -> tuple[float, float, float, float]:
    parts = [float(p.strip()) for p in value.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must have format xmin,ymin,xmax,ymax")
    minx, miny, maxx, maxy = parts
    if minx >= maxx or miny >= maxy:
        raise ValueError("bbox must satisfy xmin<xmax and ymin<ymax")
    return (minx, miny, maxx, maxy)


def select_year_matches(
    requested_years: list[int],
    available_years: list[int],
    *,
    policy_name: str,
    max_delta: int | None,
) -> dict[int, year_policy.YearMatch]:
    return {
        year: year_policy.select_year(
            year,
            available_years,
            policy=policy_name,
            max_delta=max_delta,
        )
        for year in requested_years
    }


def build_index_manifest(
    *,
    config: IndexConfig,
    items_by_year: dict[int, list[stac.StacItem]],
    matches: dict[int, year_policy.YearMatch],
    available_stac_years: list[int],
    warnings: list[str],
    errors: list[str],
    provider_metadata: dict[str, Any],
) -> IndexManifest:
    requested_years = config.requested_years
    year_statuses: list[YearStatus] = []
    tile_sources_by_year: dict[int, dict[str, str]] = {}
    tile_bboxes_by_year: dict[int, dict[str, list[float]]] = {}
    tile_acquisition_by_year: dict[int, dict[str, TileAcquisitionMetadata]] = {}
    years_excluded_with_reason: dict[int, str] = {}

    for year in requested_years:
        match = matches[year]
        if match.matched_year is None:
            year_statuses.append(
                YearStatus(
                    year=year,
                    typename_exists=False,
                    feature_count=0,
                    status="no_typename",
                    reason=match.reason,
                )
            )
            years_excluded_with_reason[year] = match.reason
            continue
        items = items_by_year.get(match.matched_year, [])
        if not items:
            year_statuses.append(
                YearStatus(
                    year=year,
                    typename_exists=True,
                    feature_count=0,
                    status="zero_features",
                    reason="matched_year_missing_item",
                )
            )
            years_excluded_with_reason[year] = "matched_year_missing_item"
            continue

        sources: dict[str, str] = {}
        bboxes: dict[str, list[float]] = {}
        acquisition: dict[str, TileAcquisitionMetadata] = {}
        for item in items:
            asset = stac.select_asset(item)
            if asset is None:
                continue
            tile_id = item.item_id or f"{match.matched_year}_{asset.key}"
            sources[tile_id] = asset.href
            footprint = item.proj_bbox if item.proj_bbox is not None else item.bbox
            if footprint is not None:
                bboxes[tile_id] = list(footprint)
            acquisition[tile_id] = TileAcquisitionMetadata(
                acquisition_date=item.datetime_iso,
                publication_date=None,
                acquisition_year=match.matched_year,
            )

        if not sources:
            year_statuses.append(
                YearStatus(
                    year=year,
                    typename_exists=True,
                    feature_count=0,
                    status="zero_features",
                    reason="no_downloadable_asset",
                )
            )
            years_excluded_with_reason[year] = "no_downloadable_asset"
            continue

        tile_sources_by_year[year] = sources
        if bboxes:
            tile_bboxes_by_year[year] = bboxes
        tile_acquisition_by_year[year] = acquisition
        year_statuses.append(
            YearStatus(
                year=year,
                typename_exists=True,
                feature_count=len(sources),
                status="has_features",
                reason=None if match.delta == 0 else f"{match.reason}_delta_{match.delta}",
            )
        )

    years_included = sorted(tile_sources_by_year.keys())
    policy = evaluate_year_policy(
        requested_years=requested_years,
        available_years=years_included,
        strict_years=config.strict_years,
        min_years=config.min_years,
    )
    combined_warnings = list(warnings) + list(policy.warnings)
    combined_errors = list(errors) + list(policy.errors)

    if not years_included:
        combined_errors.append(
            "STAC returned no usable assets for the requested bbox/years."
        )

    return IndexManifest(
        provider="lantmateriet",
        year_start=config.year_start,
        year_end=config.year_end,
        bbox=config.bbox,
        srs=config.srs,
        strict_years=config.strict_years,
        min_years=config.min_years,
        wfs_bbox_axes_swapped=False,
        years_requested=requested_years,
        year_statuses=year_statuses,
        years_available_wfs=available_stac_years,
        years_included=years_included,
        years_excluded_with_reason=years_excluded_with_reason,
        common_tile_ids=[],
        tile_sources_by_year=tile_sources_by_year,
        tile_bboxes_by_year=tile_bboxes_by_year,
        tile_acquisition_by_year=tile_acquisition_by_year,
        passed=policy.passed and bool(years_included),
        errors=combined_errors,
        warnings=combined_warnings,
        run_parameters=config.model_dump(mode="json"),
        provider_metadata=provider_metadata,
    )


def build_year_availability_report(
    *,
    config: IndexConfig,
    manifest: IndexManifest,
) -> YearAvailabilityReport:
    return YearAvailabilityReport(
        year_start=config.year_start,
        year_end=config.year_end,
        bbox=config.bbox,
        srs=config.srs,
        wfs_bbox_axes_swapped=False,
        years_requested=manifest.years_requested,
        year_statuses=manifest.year_statuses,
        years_available_wfs=manifest.years_available_wfs,
        years_included=manifest.years_included,
        years_excluded_with_reason=manifest.years_excluded_with_reason,
        strict_years=manifest.strict_years,
        min_years=manifest.min_years,
        passed=manifest.passed,
        errors=manifest.errors,
        warnings=manifest.warnings,
        run_parameters=manifest.run_parameters,
    )


_NON_RETRYABLE_STATUSES = frozenset({400, 401, 403, 404, 410})


async def _download_asset_with_retry(
    client: httpx.AsyncClient,
    url: str,
    output_path: Path,
    *,
    retries: int,
    retry_delay: float,
    sleep_min: float,
    sleep_max: float,
) -> bool:
    attempts = retries + 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        await asyncio.sleep(random.uniform(sleep_min, sleep_max))
        try:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                async with aiofiles.open(output_path, "wb") as file_handle:
                    async for chunk in response.aiter_bytes():
                        await file_handle.write(chunk)
            return True
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            logger.warning("Lantmäteriet download attempt=%s status=%s url=%s", attempt, status, url)
            if status in _NON_RETRYABLE_STATUSES:
                # Retrying a 4xx wastes time and quota — surface the failure now.
                return False
            if attempt >= attempts:
                return False
            await asyncio.sleep(retry_delay * (2 ** (attempt - 1)))
        except (httpx.HTTPError, OSError) as exc:
            logger.warning("Lantmäteriet download attempt=%s failed: %s", attempt, exc)
            if attempt >= attempts:
                return False
            await asyncio.sleep(retry_delay * (2 ** (attempt - 1)))
    return False


def _filename_for_url(url: str, fallback: str) -> str:
    name = Path(urlparse(url).path).name
    return name or fallback


class LantmaterietProvider(Provider):
    name = "lantmateriet"
    default_target_srs = DEFAULT_TARGET_SRS

    def index(self, config: IndexConfig) -> tuple[int, Path]:
        return asyncio.run(self._index_async(config))

    async def _index_async(self, config: IndexConfig) -> tuple[int, Path]:
        options = dict(config.provider_options)
        search_options = _resolve_search_options(options)
        bbox = _parse_bbox(config.bbox)
        bbox_for_search, bbox_crs_for_search = self._bbox_for_search(bbox, config.srs, options)
        datetime_range = options.get(
            "datetime_range",
            f"{config.year_start}-01-01T00:00:00Z/{config.year_end}-12-31T23:59:59Z",
        )
        retry_policy = RetryPolicy(max_attempts=int(options.get("retry_max_attempts", 6)))

        warnings: list[str] = []
        errors: list[str] = []
        items: list[stac.StacItem]
        try:
            items = await stac.search_features(
                search_options,
                bbox=bbox_for_search,
                bbox_crs=bbox_crs_for_search,
                datetime_range=str(datetime_range) if datetime_range else None,
                timeout=float(options.get("timeout", 60.0)),
                retry_policy=retry_policy,
            )
        except httpx.HTTPError as exc:
            items = []
            errors.append(f"STAC search failed: {exc}")

        spectral_filter = options.get("spectral_type", DEFAULT_SPECTRAL_TYPES)
        if isinstance(spectral_filter, str):
            allowed_spectral = {spectral_filter.strip().lower()} if spectral_filter else set()
        elif isinstance(spectral_filter, (list, tuple, set)):
            allowed_spectral = {str(s).strip().lower() for s in spectral_filter if s}
        else:
            allowed_spectral = set()
        if allowed_spectral:
            items = [
                item
                for item in items
                if item.spectral_type is None or item.spectral_type in allowed_spectral
            ]
        grouped = stac.group_items_by_year(items)
        items_by_year = {year: list(year_items) for year, year_items in grouped.items()}
        available_stac_years = sorted(items_by_year.keys())

        policy_name = str(options.get("year_policy", "exact_only"))
        if policy_name not in year_policy.ALLOWED_POLICIES:
            errors.append(
                f"Invalid year_policy {policy_name!r}; expected one of {sorted(year_policy.ALLOWED_POLICIES)}."
            )
            policy_name = "exact_only"
        max_delta_value = options.get("max_year_delta")
        max_delta = int(max_delta_value) if max_delta_value is not None else None

        matches = select_year_matches(
            requested_years=config.requested_years,
            available_years=available_stac_years,
            policy_name=policy_name,
            max_delta=max_delta,
        )

        provider_metadata: dict[str, Any] = {
            "stac_url": search_options.url,
            "stac_collections": list(search_options.collections),
            "year_policy": policy_name,
            "max_year_delta": max_delta,
            "datetime_range": datetime_range,
            "available_stac_years": available_stac_years,
            "spectral_type": list(allowed_spectral) if allowed_spectral else None,
            "tiles_per_year": {year: len(items) for year, items in items_by_year.items()},
        }

        manifest = build_index_manifest(
            config=config,
            items_by_year=items_by_year,
            matches=matches,
            available_stac_years=available_stac_years,
            warnings=warnings,
            errors=errors,
            provider_metadata=provider_metadata,
        )
        availability = build_year_availability_report(config=config, manifest=manifest)

        config.year_availability_output_json.parent.mkdir(parents=True, exist_ok=True)
        config.output_json.parent.mkdir(parents=True, exist_ok=True)
        config.year_availability_output_json.write_text(
            availability.model_dump_json(indent=2), encoding="utf-8"
        )
        config.output_json.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        logger.info(
            "Lantmäteriet index: years_included=%s years_available=%s passed=%s",
            len(manifest.years_included),
            len(available_stac_years),
            manifest.passed,
        )
        return (0 if manifest.passed else 1), config.output_json

    def download(self, config: DownloadConfig) -> tuple[int, Path]:
        return asyncio.run(self._download_async(config))

    async def _download_async(self, config: DownloadConfig) -> tuple[int, Path]:
        index_manifest = IndexManifest.model_validate_json(
            config.index_manifest.read_text(encoding="utf-8")
        )
        options = dict(config.provider_options)

        years_source_map: dict[int, str] = {}
        assets: list[str] = []
        failed: list[str] = []

        jobs: list[tuple[int, str, str, Path]] = []
        for year in index_manifest.years_included:
            sources = index_manifest.tile_sources_by_year.get(year, {})
            for tile_id, url in sources.items():
                filename = _filename_for_url(url, f"{tile_id}.tif")
                output_path = config.download_root / str(year) / filename
                jobs.append((year, tile_id, url, output_path))

        timeout = httpx.Timeout(timeout=config.timeout, connect=min(config.timeout, 20.0))
        limits = httpx.Limits(
            max_connections=config.concurrency, max_keepalive_connections=config.concurrency
        )
        api_key = _option(options, "api_key", "SATMAP_LANTMATERIET_API_KEY", None)
        basic_auth = _basic_auth_header(options)
        headers = {"User-Agent": "satmap_dataset/0.1"}
        if basic_auth:
            headers["Authorization"] = basic_auth
        elif api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        if jobs:
            queue: asyncio.Queue[tuple[int, str, str, Path] | None] = asyncio.Queue()
            for job in jobs:
                queue.put_nowait(job)
            lock = asyncio.Lock()

            async def worker() -> None:
                async with httpx.AsyncClient(
                    follow_redirects=True, timeout=timeout, limits=limits, headers=headers
                ) as client:
                    while True:
                        item = await queue.get()
                        if item is None:
                            queue.task_done()
                            return
                        year, _tile_id, url, output_path = item
                        ok = output_path.exists() and output_path.stat().st_size > 0 and not config.overwrite
                        if not ok:
                            ok = await _download_asset_with_retry(
                                client,
                                url,
                                output_path,
                                retries=config.retries,
                                retry_delay=config.retry_delay,
                                sleep_min=config.sleep_min,
                                sleep_max=config.sleep_max,
                            )
                        async with lock:
                            if ok:
                                assets.append(str(output_path))
                                years_source_map[year] = "stac"
                            else:
                                failed.append(url)
                        queue.task_done()

            workers = [asyncio.create_task(worker()) for _ in range(max(1, config.concurrency))]
            await queue.join()
            for _ in workers:
                queue.put_nowait(None)
            await asyncio.gather(*workers)

        # Optional WMS fallback for years still missing.
        if config.wms_fallback_missing_years and config.bbox is not None:
            missing_years = [
                year for year in index_manifest.years_requested if year not in years_source_map
            ]
            if missing_years:
                fallback_urls = self._build_wms_fallback_urls(
                    config=config,
                    years=missing_years,
                    options=options,
                )
                async with httpx.AsyncClient(
                    follow_redirects=True, timeout=timeout, limits=limits, headers=headers
                ) as client:
                    for year, url in fallback_urls.items():
                        output_path = config.download_root / str(year) / f"wms_{year}.tiff"
                        ok = await _download_asset_with_retry(
                            client,
                            url,
                            output_path,
                            retries=config.retries,
                            retry_delay=config.retry_delay,
                            sleep_min=config.sleep_min,
                            sleep_max=config.sleep_max,
                        )
                        if ok:
                            assets.append(str(output_path))
                            years_source_map[year] = "wms_fallback"
                        else:
                            failed.append(url)

        years_included_effective = sorted(
            year for year, src in years_source_map.items() if src in {"stac", "wms_fallback"}
        )
        manifest = DatasetManifest(
            stage="download",
            provider="lantmateriet",
            years_requested=index_manifest.years_requested,
            years_available_wfs=index_manifest.years_available_wfs,
            years_included=years_included_effective,
            years_excluded_with_reason=index_manifest.years_excluded_with_reason,
            common_tile_ids=index_manifest.common_tile_ids,
            tile_sources_by_year=index_manifest.tile_sources_by_year,
            tile_bboxes_by_year=index_manifest.tile_bboxes_by_year,
            tile_acquisition_by_year=index_manifest.tile_acquisition_by_year,
            assets=sorted(set(assets)),
            source_manifest=str(config.index_manifest),
            mode="stac",
            target_bbox=config.bbox,
            target_srs=config.srs,
            profile=config.profile,
            px_per_meter=config.px_per_meter,
            years_source_map=years_source_map,
            forced_wms_years=[],
            passed=bool(assets) and not failed,
            notes=(
                f"provider=lantmateriet downloaded={len(assets)} failed={len(failed)} "
                f"years_included={years_included_effective}"
            ),
            run_parameters=config.model_dump(mode="json"),
            provider_metadata={
                "wms_fallback_used": [y for y, src in years_source_map.items() if src == "wms_fallback"],
            },
        )
        config.output_json.parent.mkdir(parents=True, exist_ok=True)
        config.output_json.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        logger.info(
            "Lantmäteriet download: assets=%s failed=%s passed=%s",
            len(assets),
            len(failed),
            manifest.passed,
        )
        return (0 if manifest.passed else 1), config.output_json

    @staticmethod
    def _bbox_for_search(
        bbox: tuple[float, float, float, float],
        request_srs: str,
        options: dict[str, Any],
    ) -> tuple[tuple[float, float, float, float], str | None]:
        """Lantmäteriet's STAC requires WGS84 bbox; convert from request CRS.

        Honors `bbox_crs` override in provider_options for non-Lantmäteriet
        STAC endpoints that accept a bbox-crs hint (default: omit when WGS84).
        """
        explicit_crs = options.get("bbox_crs")
        if explicit_crs:
            return bbox, str(explicit_crs)
        if request_srs.upper() == "EPSG:4326":
            return bbox, None
        minx, miny, maxx, maxy = bbox
        ll_min = crs.transform_point(request_srs, "EPSG:4326", minx, miny)
        ll_max = crs.transform_point(request_srs, "EPSG:4326", maxx, maxy)
        wgs_bbox = (
            min(ll_min[0], ll_max[0]),
            min(ll_min[1], ll_max[1]),
            max(ll_min[0], ll_max[0]),
            max(ll_min[1], ll_max[1]),
        )
        return wgs_bbox, None

    @staticmethod
    def _build_wms_fallback_urls(
        *,
        config: DownloadConfig,
        years: list[int],
        options: dict[str, Any],
    ) -> dict[int, str]:
        wms_url = _option(options, "wms_url", "SATMAP_LANTMATERIET_WMS_URL", DEFAULT_WMS_URL)
        wms_layer = _option(options, "wms_layer", "SATMAP_LANTMATERIET_WMS_LAYER", DEFAULT_WMS_LAYER)
        if not wms_url or not wms_layer:
            # Lantmäteriet's annual WMS (Ortofoto Visning, Årsvisa) is a paid
            # product. We refuse to silently fall back without an explicit URL
            # and layer — caller must opt in via provider_options or env vars.
            logger.warning(
                "wms_fallback_missing_years requested but wms_url/wms_layer not "
                "configured; skipping WMS fallback for years=%s",
                years,
            )
            return {}
        wms_version = str(options.get("wms_version", "1.3.0"))
        bbox = _parse_bbox(config.bbox or "")
        width_px = int(options.get("wms_width_px", 2048))
        height_px = int(options.get("wms_height_px", 2048))
        urls: dict[int, str] = {}
        for year in years:
            urls[year] = wms.build_get_map_url(
                str(wms_url),
                layer=str(wms_layer),
                bbox=bbox,
                srs=config.srs,
                width=width_px,
                height=height_px,
                year=year,
                version=wms_version,
            )
        return urls
