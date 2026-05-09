"""Sentinel-2 L2A provider, Element84 Earth Search backend.

Picks one representative scene per requested year by minimising the distance
from a target day-of-year (default Feb 15 — winter for the northern
hemisphere) and filtering by ``eo:cloud_cover``. Downloads the Sentinel-2
``visual`` asset (3-band 10 m True-Color COG, anonymous S3) per chosen year.

The native CRS of each scene is the MGRS UTM zone (e.g. EPSG:32633, EPSG:32635).
Set ``RunConfig.srs`` and ``RunConfig.target_srs`` to the matching UTM zone
for your AOI — the existing render stage does not reproject across CRS, so
mixing source UTM with a different render target produces misaligned pixels.

Configuration via ``provider_options`` / env vars:

- ``stac_url``               (env ``SATMAP_SENTINEL2_STAC_URL``):
                              Earth Search /search endpoint.
- ``stac_collection``        (env ``SATMAP_SENTINEL2_STAC_COLLECTION``):
                              defaults to ``sentinel-2-l2a``.
- ``preferred_asset_key``    asset key to download. Default ``"visual"``
                              (the 3-band 10 m TCI COG).
- ``target_month`` / ``target_day``  desired DOY for representative scene.
                                      Defaults ``(2, 15)``.
- ``max_cloud_cover_pct``    discard scenes above this. Default 20.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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
from satmap_dataset.providers.lantmateriet import crs as crs_helpers
from satmap_dataset.providers.lantmateriet import stac
from satmap_dataset.providers.lantmateriet.provider import _download_asset_with_retry
from satmap_dataset.providers.sentinel2.year_policy import (
    CandidateItem,
    YearPick,
    select_years,
)

DEFAULT_STAC_URL = "https://earth-search.aws.element84.com/v1/search"
DEFAULT_STAC_COLLECTION = "sentinel-2-l2a"
DEFAULT_PREFERRED_ASSET = "visual"
DEFAULT_TARGET_MONTH = 2
DEFAULT_TARGET_DAY = 15
DEFAULT_MAX_CLOUD_COVER_PCT = 20.0
DEFAULT_NATIVE_GSD_M = 10.0

logger = logging.getLogger("satmap_dataset.sentinel2")


def _option(options: dict[str, Any], key: str, env_var: str | None, default: Any) -> Any:
    if key in options and options[key] not in (None, ""):
        return options[key]
    if env_var:
        env_value = os.environ.get(env_var)
        if env_value:
            return env_value
    return default


def _resolve_search_options(options: dict[str, Any]) -> stac.StacSearchOptions:
    url = _option(options, "stac_url", "SATMAP_SENTINEL2_STAC_URL", DEFAULT_STAC_URL)
    collection_value = _option(
        options,
        "stac_collection",
        "SATMAP_SENTINEL2_STAC_COLLECTION",
        DEFAULT_STAC_COLLECTION,
    )
    if isinstance(collection_value, (list, tuple)):
        collections: tuple[str, ...] = tuple(str(c) for c in collection_value if c)
    else:
        collections = (str(collection_value),) if collection_value else ()
    return stac.StacSearchOptions(
        url=str(url),
        collections=collections,
        api_key=None,
        page_limit=int(options.get("page_limit", 100)),
        max_pages=int(options.get("max_pages", 50)),
    )


def _parse_bbox(value: str) -> tuple[float, float, float, float]:
    parts = [float(p.strip()) for p in value.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must have format xmin,ymin,xmax,ymax")
    minx, miny, maxx, maxy = parts
    if minx >= maxx or miny >= maxy:
        raise ValueError("bbox must satisfy xmin<xmax and ymin<ymax")
    return (minx, miny, maxx, maxy)


def _bbox_to_wgs84(
    bbox: tuple[float, float, float, float], src_srs: str
) -> tuple[float, float, float, float]:
    if src_srs.upper() == "EPSG:4326":
        return bbox
    minx, miny, maxx, maxy = bbox
    ll_min = crs_helpers.transform_point(src_srs, "EPSG:4326", minx, miny)
    ll_max = crs_helpers.transform_point(src_srs, "EPSG:4326", maxx, maxy)
    return (
        min(ll_min[0], ll_max[0]),
        min(ll_min[1], ll_max[1]),
        max(ll_min[0], ll_max[0]),
        max(ll_min[1], ll_max[1]),
    )


def _select_visual_asset(item: stac.StacItem, preferred_key: str) -> stac.StacAsset | None:
    if preferred_key:
        for asset in item.assets:
            if asset.key == preferred_key:
                return asset
    return stac.select_asset(item)


def _build_search_query(max_cloud_cover_pct: float | None) -> dict[str, Any] | None:
    if max_cloud_cover_pct is None:
        return None
    return {"eo:cloud_cover": {"lt": float(max_cloud_cover_pct)}}


def _filename_for_url(url: str, fallback: str) -> str:
    name = Path(urlparse(url).path).name
    return name or fallback


class Sentinel2Provider(Provider):
    name = "sentinel2"
    default_target_srs = "EPSG:32633"  # placeholder; set per AOI

    def index(self, config: IndexConfig) -> tuple[int, Path]:
        return asyncio.run(self._index_async(config))

    async def _index_async(self, config: IndexConfig) -> tuple[int, Path]:
        options = dict(config.provider_options)
        search_options = _resolve_search_options(options)
        bbox = _parse_bbox(config.bbox)
        wgs84_bbox = _bbox_to_wgs84(bbox, config.srs)
        target_month = int(options.get("target_month", DEFAULT_TARGET_MONTH))
        target_day = int(options.get("target_day", DEFAULT_TARGET_DAY))
        max_cloud = options.get("max_cloud_cover_pct", DEFAULT_MAX_CLOUD_COVER_PCT)
        max_cloud_value = float(max_cloud) if max_cloud is not None else None
        preferred_asset = str(options.get("preferred_asset_key", DEFAULT_PREFERRED_ASSET))
        datetime_range = options.get(
            "datetime_range",
            f"{config.year_start}-01-01T00:00:00Z/{config.year_end}-12-31T23:59:59Z",
        )
        retry_policy = RetryPolicy(max_attempts=int(options.get("retry_max_attempts", 6)))
        query = _build_search_query(max_cloud_value)

        warnings: list[str] = []
        errors: list[str] = []
        items: list[stac.StacItem]
        try:
            items = await stac.search_features(
                search_options,
                bbox=wgs84_bbox,
                bbox_crs=None,
                datetime_range=str(datetime_range),
                timeout=float(options.get("timeout", 60.0)),
                retry_policy=retry_policy,
                extra_body={"query": query} if query else None,
            )
        except httpx.HTTPError as exc:
            items = []
            errors.append(f"Earth Search query failed: {exc}")

        candidates = [
            CandidateItem(
                item_id=item.item_id,
                datetime_iso=item.datetime_iso or "",
                cloud_cover=item.cloud_cover,
            )
            for item in items
            if item.datetime_iso
        ]
        items_by_id = {item.item_id: item for item in items}
        picks = select_years(
            requested_years=config.requested_years,
            candidates=candidates,
            target_month=target_month,
            target_day=target_day,
            max_cloud_cover_pct=max_cloud_value,
        )

        chosen_epsgs: set[int] = set()
        manifest = self._build_index_manifest(
            config=config,
            picks=picks,
            items_by_id=items_by_id,
            preferred_asset=preferred_asset,
            warnings=warnings,
            errors=errors,
            chosen_epsgs=chosen_epsgs,
            search_url=search_options.url,
            collections=list(search_options.collections),
            target_month=target_month,
            target_day=target_day,
            max_cloud_cover_pct=max_cloud_value,
        )
        availability = YearAvailabilityReport(
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
        config.year_availability_output_json.parent.mkdir(parents=True, exist_ok=True)
        config.output_json.parent.mkdir(parents=True, exist_ok=True)
        config.year_availability_output_json.write_text(
            availability.model_dump_json(indent=2), encoding="utf-8"
        )
        config.output_json.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

        if chosen_epsgs and len(chosen_epsgs) > 1:
            logger.warning(
                "Sentinel-2: chosen scenes span multiple UTM zones %s — render to a single CRS may be inconsistent.",
                sorted(chosen_epsgs),
            )

        logger.info(
            "Sentinel-2 index: years_included=%s years_searched=%s passed=%s",
            len(manifest.years_included),
            len(items),
            manifest.passed,
        )
        return (0 if manifest.passed else 1), config.output_json

    @staticmethod
    def _build_index_manifest(
        *,
        config: IndexConfig,
        picks: dict[int, YearPick],
        items_by_id: dict[str, stac.StacItem],
        preferred_asset: str,
        warnings: list[str],
        errors: list[str],
        chosen_epsgs: set[int],
        search_url: str,
        collections: list[str],
        target_month: int,
        target_day: int,
        max_cloud_cover_pct: float | None,
    ) -> IndexManifest:
        requested_years = config.requested_years
        year_statuses: list[YearStatus] = []
        tile_sources_by_year: dict[int, dict[str, str]] = {}
        tile_bboxes_by_year: dict[int, dict[str, list[float]]] = {}
        tile_acquisition_by_year: dict[int, dict[str, TileAcquisitionMetadata]] = {}
        years_excluded_with_reason: dict[int, str] = {}
        scenes_metadata: list[dict[str, Any]] = []

        for year in requested_years:
            pick = picks.get(year)
            if pick is None or pick.chosen is None:
                reason = pick.reason if pick is not None else "no_pick"
                year_statuses.append(
                    YearStatus(
                        year=year,
                        typename_exists=False,
                        feature_count=0,
                        status="no_typename",
                        reason=reason,
                    )
                )
                years_excluded_with_reason[year] = reason
                continue
            item = items_by_id.get(pick.chosen.item_id)
            if item is None:
                year_statuses.append(
                    YearStatus(
                        year=year,
                        typename_exists=True,
                        feature_count=0,
                        status="zero_features",
                        reason="picked_item_missing",
                    )
                )
                years_excluded_with_reason[year] = "picked_item_missing"
                continue
            asset = _select_visual_asset(item, preferred_asset)
            if asset is None:
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

            if item.epsg is not None:
                chosen_epsgs.add(int(item.epsg))
            tile_id = item.item_id
            tile_sources_by_year[year] = {tile_id: asset.href}
            footprint = item.proj_bbox if item.proj_bbox is not None else item.bbox
            if footprint is not None:
                tile_bboxes_by_year[year] = {tile_id: list(footprint)}
            tile_acquisition_by_year[year] = {
                tile_id: TileAcquisitionMetadata(
                    acquisition_date=item.datetime_iso,
                    publication_date=None,
                    acquisition_year=year,
                )
            }
            year_statuses.append(
                YearStatus(
                    year=year,
                    typename_exists=True,
                    feature_count=1,
                    status="has_features",
                    reason=f"delta_days={pick.delta_days}",
                )
            )
            scenes_metadata.append(
                {
                    "year": year,
                    "item_id": item.item_id,
                    "datetime": item.datetime_iso,
                    "cloud_cover": item.cloud_cover,
                    "epsg": item.epsg,
                    "delta_days_to_target": pick.delta_days,
                }
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
            combined_errors.append("Earth Search returned no usable scenes.")

        provider_metadata: dict[str, Any] = {
            "stac_url": search_url,
            "stac_collections": collections,
            "target_month": target_month,
            "target_day": target_day,
            "max_cloud_cover_pct": max_cloud_cover_pct,
            "scenes": scenes_metadata,
            "chosen_epsgs": sorted(chosen_epsgs),
        }

        return IndexManifest(
            provider="sentinel2",
            year_start=config.year_start,
            year_end=config.year_end,
            bbox=config.bbox,
            srs=config.srs,
            strict_years=config.strict_years,
            min_years=config.min_years,
            wfs_bbox_axes_swapped=False,
            years_requested=requested_years,
            year_statuses=year_statuses,
            years_available_wfs=years_included,
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

    def download(self, config: DownloadConfig) -> tuple[int, Path]:
        return asyncio.run(self._download_async(config))

    async def _download_async(self, config: DownloadConfig) -> tuple[int, Path]:
        index_manifest = IndexManifest.model_validate_json(
            config.index_manifest.read_text(encoding="utf-8")
        )

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
        headers = {"User-Agent": "satmap_dataset/0.1"}

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
                        ok = (
                            output_path.exists()
                            and output_path.stat().st_size > 0
                            and not config.overwrite
                        )
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

            workers = [
                asyncio.create_task(worker()) for _ in range(max(1, config.concurrency))
            ]
            await queue.join()
            for _ in workers:
                queue.put_nowait(None)
            await asyncio.gather(*workers)

        years_included_effective = sorted(years_source_map.keys())
        manifest = DatasetManifest(
            stage="download",
            provider="sentinel2",
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
                f"provider=sentinel2 downloaded={len(assets)} failed={len(failed)} "
                f"years_included={years_included_effective}"
            ),
            run_parameters=config.model_dump(mode="json"),
            provider_metadata=index_manifest.provider_metadata,
        )
        config.output_json.parent.mkdir(parents=True, exist_ok=True)
        config.output_json.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        logger.info(
            "Sentinel-2 download: assets=%s failed=%s passed=%s",
            len(assets),
            len(failed),
            manifest.passed,
        )
        return (0 if manifest.passed else 1), config.output_json
