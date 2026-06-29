"""LROC NAC (Moon) multi-temporal provider — ODE index + download.

Sources lunar NAC observations from the PDS Orbital Data Explorer REST API.
Index enumerates every overlapping NAC observation across a lat/lon bbox and
date range; download pulls the PDS frames. Map projection (ISIS cam2map) and
render are intentionally out of scope for this provider.
"""

from __future__ import annotations

import asyncio
import logging
import random
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import aiofiles
import httpx

from satmap_dataset.config import DownloadConfig, IndexConfig
from satmap_dataset.models import (
    IndexManifest,
    LayerManifest,
    TileAcquisitionMetadata,
    YearAvailabilityReport,
    YearStatus,
)
from satmap_dataset.pipeline.validator import evaluate_year_policy
from satmap_dataset.providers.base import Provider
from satmap_dataset.providers.lroc_nac import crs, ode

logger = logging.getLogger("satmap_dataset.lroc_nac")

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
                async with aiofiles.open(output_path, "wb") as handle:
                    async for chunk in response.aiter_bytes():
                        await handle.write(chunk)
            return True
        except httpx.HTTPStatusError as exc:
            logger.warning("LROC NAC download attempt=%s status=%s url=%s", attempt, exc.response.status_code, url)
            if exc.response.status_code in _NON_RETRYABLE_STATUSES:
                return False
            if attempt >= attempts:
                return False
            await asyncio.sleep(retry_delay * (2 ** (attempt - 1)))
        except (httpx.HTTPError, OSError) as exc:
            logger.warning("LROC NAC download attempt=%s failed: %s", attempt, exc)
            if attempt >= attempts:
                return False
            await asyncio.sleep(retry_delay * (2 ** (attempt - 1)))
    return False


def _ext_for_url(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix
    return suffix or ".IMG"


def _safe_name(pdsid: str) -> str:
    name = pdsid.replace("/", "_").replace("\\", "_").replace("..", "_")
    return name or "tile"

DEFAULT_TARGET_SRS = "IAU_2015:30100"


class LrocNacProvider(Provider):
    name = "lroc_nac"
    default_target_srs = DEFAULT_TARGET_SRS

    def index(self, config: IndexConfig) -> tuple[int, Path]:
        return asyncio.run(self._index_async(config))

    async def _index_async(self, config: IndexConfig) -> tuple[int, Path]:
        options = dict(config.provider_options)
        search_options = ode.OdeSearchOptions(
            url=str(options.get("ode_url", ode.OdeSearchOptions.url)),
            product_type=str(options.get("product_type", "CDRNAC4")),
            loc=str(options.get("loc", "f")),
            limit=int(options.get("page_limit", 100)),
            max_pages=int(options.get("max_pages", 20)),
        )
        westlon, eastlon, minlat, maxlat = crs.normalize_bbox_to_ode(config.bbox, config.srs)
        min_obtime = str(options.get("min_obtime", f"{config.year_start}-01-01"))
        max_obtime = str(options.get("max_obtime", f"{config.year_end}-12-31"))

        warnings: list[str] = []
        errors: list[str] = []
        try:
            products = await ode.search_products(
                search_options,
                westlon=westlon, eastlon=eastlon, minlat=minlat, maxlat=maxlat,
                min_obtime=min_obtime, max_obtime=max_obtime,
                timeout=float(options.get("timeout", 60.0)),
            )
        except Exception as exc:  # noqa: BLE001 — surface transport errors in manifest
            products = []
            errors.append(f"ODE search failed: {exc}")

        max_incidence = options.get("max_incidence_angle")
        if max_incidence is not None:
            limit = float(max_incidence)
            products = [
                p for p in products
                if p.incidence_angle is None or p.incidence_angle <= limit
            ]

        grouped = ode.group_products_by_year(products)
        years_available = sorted(grouped.keys())

        tile_sources_by_year: dict[int, dict[str, str]] = {}
        tile_bboxes_by_year: dict[int, dict[str, list[float]]] = {}
        tile_acquisition_by_year: dict[int, dict[str, TileAcquisitionMetadata]] = {}
        year_statuses: list[YearStatus] = []
        years_excluded: dict[int, str] = {}

        for year in config.requested_years:
            items = grouped.get(year, [])
            if not items:
                year_statuses.append(
                    YearStatus(year=year, typename_exists=year in years_available,
                               feature_count=0, status="zero_features",
                               reason="no_nac_observation")
                )
                years_excluded[year] = "no_nac_observation"
                continue
            sources: dict[str, str] = {}
            bboxes: dict[str, list[float]] = {}
            acquisition: dict[str, TileAcquisitionMetadata] = {}
            for product in items:
                if product.file_url is None:
                    continue
                sources[product.pdsid] = product.file_url
                if product.footprint_bbox is not None:
                    bboxes[product.pdsid] = list(product.footprint_bbox)
                acquisition[product.pdsid] = TileAcquisitionMetadata(
                    acquisition_date=product.observation_time,
                    publication_date=None,
                    acquisition_year=year,
                )
            if not sources:
                year_statuses.append(
                    YearStatus(year=year, typename_exists=True, feature_count=0,
                               status="zero_features", reason="no_downloadable_asset")
                )
                years_excluded[year] = "no_downloadable_asset"
                continue
            tile_sources_by_year[year] = sources
            if bboxes:
                tile_bboxes_by_year[year] = bboxes
            tile_acquisition_by_year[year] = acquisition
            year_statuses.append(
                YearStatus(year=year, typename_exists=True,
                           feature_count=len(sources), status="has_features", reason=None)
            )

        years_included = sorted(tile_sources_by_year.keys())
        policy = evaluate_year_policy(
            requested_years=config.requested_years,
            available_years=years_included,
            strict_years=config.strict_years,
            min_years=config.min_years,
        )
        if not years_included and not errors:
            errors.append("ODE returned no NAC observations for the requested bbox/date range.")

        provider_metadata: dict[str, Any] = {
            "ode_url": search_options.url,
            "product_type": search_options.product_type,
            "loc": search_options.loc,
            "min_obtime": min_obtime,
            "max_obtime": max_obtime,
            "available_years": years_available,
            "observations_per_year": {y: len(grouped[y]) for y in years_available},
            "incidence_by_tile": {
                p.pdsid: p.incidence_angle for p in products if p.incidence_angle is not None
            },
            "map_resolution_by_tile": {
                p.pdsid: p.map_resolution for p in products if p.map_resolution is not None
            },
        }

        manifest = IndexManifest(
            provider="lroc_nac",
            year_start=config.year_start, year_end=config.year_end,
            bbox=config.bbox, srs=config.srs,
            strict_years=config.strict_years, min_years=config.min_years,
            wfs_bbox_axes_swapped=False,
            years_requested=config.requested_years,
            year_statuses=year_statuses,
            years_available_wfs=years_available,
            years_included=years_included,
            years_excluded_with_reason=years_excluded,
            common_tile_ids=[],
            tile_sources_by_year=tile_sources_by_year,
            tile_bboxes_by_year=tile_bboxes_by_year,
            tile_acquisition_by_year=tile_acquisition_by_year,
            passed=policy.passed and bool(years_included),
            errors=list(errors) + list(policy.errors),
            warnings=list(warnings) + list(policy.warnings),
            run_parameters=config.model_dump(mode="json"),
            provider_metadata=provider_metadata,
        )
        availability = YearAvailabilityReport(
            year_start=config.year_start, year_end=config.year_end,
            bbox=config.bbox, srs=config.srs, wfs_bbox_axes_swapped=False,
            years_requested=manifest.years_requested,
            year_statuses=manifest.year_statuses,
            years_available_wfs=manifest.years_available_wfs,
            years_included=manifest.years_included,
            years_excluded_with_reason=manifest.years_excluded_with_reason,
            strict_years=manifest.strict_years, min_years=manifest.min_years,
            passed=manifest.passed, errors=manifest.errors, warnings=manifest.warnings,
            run_parameters=manifest.run_parameters,
        )

        config.output_json.parent.mkdir(parents=True, exist_ok=True)
        config.year_availability_output_json.parent.mkdir(parents=True, exist_ok=True)
        config.year_availability_output_json.write_text(
            availability.model_dump_json(indent=2), encoding="utf-8"
        )
        config.output_json.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        logger.info(
            "LROC NAC index: years_included=%s available=%s passed=%s",
            years_included, years_available, manifest.passed,
        )
        return (0 if manifest.passed else 1), config.output_json

    def download(self, config: DownloadConfig) -> tuple[int, Path]:
        return asyncio.run(self._download_async(config))

    async def _download_async(self, config: DownloadConfig) -> tuple[int, Path]:
        index_manifest = IndexManifest.model_validate_json(
            config.index_manifest.read_text(encoding="utf-8")
        )
        jobs: list[tuple[int, str, str, Path]] = []
        for year in index_manifest.years_included:
            for pdsid, url in index_manifest.tile_sources_by_year.get(year, {}).items():
                output_path = config.download_root / str(year) / f"{_safe_name(pdsid)}{_ext_for_url(url)}"
                jobs.append((year, pdsid, url, output_path))

        assets: list[str] = []
        failed: list[str] = []
        years_source_map: dict[int, str] = {}

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
                        year, _pdsid, url, output_path = item
                        ok = (
                            output_path.exists()
                            and output_path.stat().st_size > 0
                            and not config.overwrite
                        )
                        if not ok:
                            ok = await _download_asset_with_retry(
                                client, url, output_path,
                                retries=config.retries, retry_delay=config.retry_delay,
                                sleep_min=config.sleep_min, sleep_max=config.sleep_max,
                            )
                        async with lock:
                            if ok:
                                assets.append(str(output_path))
                                years_source_map[year] = "ode"
                            else:
                                failed.append(url)
                        queue.task_done()

            workers = [asyncio.create_task(worker()) for _ in range(max(1, config.concurrency))]
            await queue.join()
            for _ in workers:
                queue.put_nowait(None)
            await asyncio.gather(*workers)

        years_included_effective = sorted(years_source_map.keys())
        manifest = LayerManifest(
            layer="lroc_nac_mono",
            role="rgb",
            stage="download",
            provider="lroc_nac",
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
            mode="ode",
            target_bbox=config.bbox,
            target_srs=config.srs,
            profile=config.profile,
            px_per_meter=config.px_per_meter,
            years_source_map=years_source_map,
            forced_wms_years=[],
            passed=bool(assets) and not failed,
            notes=(
                f"provider=lroc_nac downloaded={len(assets)} failed={len(failed)} "
                f"years_included={years_included_effective}"
            ),
            run_parameters=config.model_dump(mode="json"),
            provider_metadata={"failed_urls": failed},
        )
        config.output_json.parent.mkdir(parents=True, exist_ok=True)
        config.output_json.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        logger.info(
            "LROC NAC download: assets=%s failed=%s passed=%s",
            len(assets), len(failed), manifest.passed,
        )
        return (0 if manifest.passed else 1), config.output_json
