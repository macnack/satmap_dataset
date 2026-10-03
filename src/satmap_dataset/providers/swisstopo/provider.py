"""swisstopo SWISSIMAGE Zeitreise provider (historical aerial orthophotos).

Indexes available years from the public WMTS GetCapabilities Time dimension
(typically 1926–present) and downloads each requested year via WMS GetMap
(``TIME=<year>``) as a GeoTIFF tagged in ``EPSG:2056`` (LV95).

No API key. Attribution: © swisstopo. See https://www.geo.admin.ch/terms-of-use

``provider_options`` / env:

- ``wmts_capabilities_url`` (``SATMAP_SWISSTOPO_WMTS_CAPABILITIES_URL``)
- ``wms_url`` (``SATMAP_SWISSTOPO_WMS_URL``)
- ``wms_layer`` (``SATMAP_SWISSTOPO_WMS_LAYER``) default ``ch.swisstopo.swissimage-product``
- ``max_wms_dim_px`` cap on GetMap width/height (default 4096)
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from pathlib import Path
from typing import Any

import httpx

from satmap_dataset.config import DownloadConfig, IndexConfig
from satmap_dataset.models import (
    IndexManifest,
    LayerManifest,
    TileAcquisitionMetadata,
    YearAvailabilityReport,
    YearStatus,
)
from satmap_dataset.pipeline.downloader import BBox, _tag_wms_tile_as_geotiff
from satmap_dataset.pipeline.validator import evaluate_year_policy
from satmap_dataset.providers.base import Provider
from satmap_dataset.providers.lantmateriet.provider import _download_asset_with_retry
from satmap_dataset.providers.swisstopo import capabilities, wms

DEFAULT_NATIVE_SRS = "EPSG:2056"
DEFAULT_MAX_WMS_DIM_PX = 4096
DEFAULT_PX_PER_METER = 1.0  # ~1 m historical; modern SWISSIMAGE is finer

logger = logging.getLogger("satmap_dataset.swisstopo")


def _option(options: dict[str, Any], key: str, env_var: str | None, default: Any) -> Any:
    if key in options and options[key] not in (None, ""):
        return options[key]
    if env_var:
        env_value = os.environ.get(env_var)
        if env_value:
            return env_value
    return default


def _parse_bbox(bbox: str) -> tuple[float, float, float, float]:
    parts = [float(p.strip()) for p in bbox.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must have format xmin,ymin,xmax,ymax")
    xmin, ymin, xmax, ymax = parts
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("bbox must satisfy xmin<xmax and ymin<ymax")
    return (xmin, ymin, xmax, ymax)


def _output_dims(
    bbox: tuple[float, float, float, float],
    *,
    px_per_meter: float,
    max_dim: int,
) -> tuple[int, int]:
    xmin, ymin, xmax, ymax = bbox
    width_m = xmax - xmin
    height_m = ymax - ymin
    width = max(1, int(round(width_m * px_per_meter)))
    height = max(1, int(round(height_m * px_per_meter)))
    longest = max(width, height)
    if longest > max_dim:
        scale = max_dim / float(longest)
        width = max(1, int(math.floor(width * scale)))
        height = max(1, int(math.floor(height * scale)))
    return width, height


def _fetch_capabilities_xml(url: str, *, timeout: float = 90.0) -> bytes:
    headers = {"User-Agent": "satmap_dataset/0.1"}
    with httpx.Client(timeout=timeout, headers=headers, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.content


class SwisstopoProvider(Provider):
    name = "swisstopo"
    default_target_srs = DEFAULT_NATIVE_SRS

    def index(self, config: IndexConfig) -> tuple[int, Path]:
        options = dict(config.provider_options)
        caps_url = str(
            _option(
                options,
                "wmts_capabilities_url",
                "SATMAP_SWISSTOPO_WMTS_CAPABILITIES_URL",
                capabilities.DEFAULT_WMTS_CAPABILITIES_URL,
            )
        )
        layer_id = str(
            _option(
                options,
                "wms_layer",
                "SATMAP_SWISSTOPO_WMS_LAYER",
                capabilities.DEFAULT_LAYER_ID,
            )
        )
        warnings: list[str] = []
        errors: list[str] = []
        available_years: list[int] = []
        try:
            xml_bytes = _fetch_capabilities_xml(
                caps_url, timeout=float(options.get("timeout", 90.0))
            )
            available_years = capabilities.parse_time_years(xml_bytes, layer_id=layer_id)
        except Exception as exc:
            errors.append(f"WMTS GetCapabilities failed: {exc}")

        requested = config.requested_years
        years_included = capabilities.intersect_years(requested, available_years)
        years_excluded = {
            year: "not_in_zeitreise_time_dimension"
            for year in requested
            if year not in set(years_included)
        }

        if config.srs.upper() != DEFAULT_NATIVE_SRS:
            warnings.append(
                f"swisstopo Zeitreise is native {DEFAULT_NATIVE_SRS}; "
                f"got srs={config.srs!r}. Reproject the AOI to LV95 for best results."
            )

        year_statuses: list[YearStatus] = []
        tile_sources_by_year: dict[int, dict[str, str]] = {}
        tile_bboxes_by_year: dict[int, dict[str, list[float]]] = {}
        tile_acquisition_by_year: dict[int, dict[str, TileAcquisitionMetadata]] = {}
        bbox = _parse_bbox(config.bbox)

        for year in requested:
            if year not in years_included:
                year_statuses.append(
                    YearStatus(
                        year=year,
                        typename_exists=False,
                        feature_count=0,
                        status="no_typename",
                        reason=years_excluded.get(year, "unavailable"),
                    )
                )
                continue
            tile_id = f"swissimage_{year}"
            # Placeholder; concrete GetMap URL is built at download with sizing.
            tile_sources_by_year[year] = {tile_id: f"wms://swisstopo/{layer_id}/{year}"}
            tile_bboxes_by_year[year] = {tile_id: list(bbox)}
            tile_acquisition_by_year[year] = {
                tile_id: TileAcquisitionMetadata(
                    acquisition_date=f"{year}-01-01",
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
                    reason="zeitreise_time",
                )
            )

        policy = evaluate_year_policy(
            requested_years=requested,
            available_years=years_included,
            strict_years=config.strict_years,
            min_years=config.min_years,
        )
        combined_errors = list(errors) + list(policy.errors)
        combined_warnings = list(warnings) + list(policy.warnings)
        if not years_included and not combined_errors:
            combined_errors.append("No Zeitreise years intersect the requested range.")

        manifest = IndexManifest(
            provider="swisstopo",
            year_start=config.year_start,
            year_end=config.year_end,
            bbox=config.bbox,
            srs=config.srs,
            strict_years=config.strict_years,
            min_years=config.min_years,
            wfs_bbox_axes_swapped=False,
            years_requested=requested,
            year_statuses=year_statuses,
            years_available_wfs=years_included,
            years_included=years_included,
            years_excluded_with_reason=years_excluded,
            common_tile_ids=[],
            tile_sources_by_year=tile_sources_by_year,
            tile_bboxes_by_year=tile_bboxes_by_year,
            tile_acquisition_by_year=tile_acquisition_by_year,
            passed=policy.passed and bool(years_included) and not errors,
            errors=combined_errors,
            warnings=combined_warnings,
            run_parameters=config.model_dump(mode="json"),
            provider_metadata={
                "wmts_capabilities_url": caps_url,
                "wms_layer": layer_id,
                "available_years_span": (
                    [available_years[0], available_years[-1]] if available_years else []
                ),
                "available_year_count": len(available_years),
            },
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
        logger.info(
            "swisstopo index: years_included=%s available=%s passed=%s",
            len(years_included),
            len(available_years),
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
        wms_url = str(
            _option(options, "wms_url", "SATMAP_SWISSTOPO_WMS_URL", wms.DEFAULT_WMS_URL)
        )
        wms_layer = str(
            _option(
                options,
                "wms_layer",
                "SATMAP_SWISSTOPO_WMS_LAYER",
                wms.DEFAULT_WMS_LAYER,
            )
        )
        wms_version = str(options.get("wms_version", wms.DEFAULT_WMS_VERSION))
        max_dim = int(options.get("max_wms_dim_px", DEFAULT_MAX_WMS_DIM_PX))
        if config.bbox is None:
            raise ValueError("bbox is required for swisstopo WMS download")
        bbox = _parse_bbox(config.bbox)
        width_px, height_px = _output_dims(
            bbox, px_per_meter=float(config.px_per_meter), max_dim=max_dim
        )
        tag_bbox = BBox(min_x=bbox[0], min_y=bbox[1], max_x=bbox[2], max_y=bbox[3])

        years_source_map: dict[int, str] = {}
        assets: list[str] = []
        failed: list[str] = []

        timeout = httpx.Timeout(timeout=config.timeout, connect=min(config.timeout, 20.0))
        limits = httpx.Limits(
            max_connections=config.concurrency, max_keepalive_connections=config.concurrency
        )
        headers = {"User-Agent": "satmap_dataset/0.1"}

        jobs: list[tuple[int, str, Path]] = []
        for year in index_manifest.years_included:
            url = wms.build_get_map_url(
                wms_url,
                layer=wms_layer,
                bbox=bbox,
                srs=config.srs,
                width=width_px,
                height=height_px,
                year=year,
                version=wms_version,
            )
            output_path = config.download_root / str(year) / f"swissimage_{year}.tif"
            jobs.append((year, url, output_path))

        if jobs:
            queue: asyncio.Queue[tuple[int, str, Path] | None] = asyncio.Queue()
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
                        year, url, output_path = item
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
                            if ok:
                                try:
                                    _tag_wms_tile_as_geotiff(
                                        output_path,
                                        tag_bbox,
                                        width_px,
                                        height_px,
                                        config.srs,
                                    )
                                except Exception as exc:
                                    logger.warning(
                                        "WMS geotag failed for %s (%s); keeping raw TIFF",
                                        output_path,
                                        exc,
                                    )
                        async with lock:
                            if ok:
                                assets.append(str(output_path))
                                years_source_map[year] = "wms"
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
        manifest = LayerManifest(
            layer="swisstopo_rgb",
            role="rgb",
            stage="download",
            provider="swisstopo",
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
            mode="wms_tiled",
            target_bbox=config.bbox,
            target_srs=config.srs,
            profile=config.profile,
            px_per_meter=config.px_per_meter,
            years_source_map=years_source_map,
            forced_wms_years=[],
            passed=bool(assets) and not failed,
            notes=(
                f"provider=swisstopo downloaded={len(assets)} failed={len(failed)} "
                f"years_included={years_included_effective} size={width_px}x{height_px}"
            ),
            run_parameters=config.model_dump(mode="json"),
            provider_metadata={
                **(index_manifest.provider_metadata or {}),
                "wms_url": wms_url,
                "wms_layer": wms_layer,
                "wms_width_px": width_px,
                "wms_height_px": height_px,
            },
        )
        config.output_json.parent.mkdir(parents=True, exist_ok=True)
        config.output_json.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        logger.info(
            "swisstopo download: assets=%s failed=%s passed=%s",
            len(assets),
            len(failed),
            manifest.passed,
        )
        return (0 if manifest.passed else 1), config.output_json
