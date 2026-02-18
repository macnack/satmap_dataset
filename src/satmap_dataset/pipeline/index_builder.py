from __future__ import annotations

import asyncio
from pathlib import Path
import random
import logging

from satmap_dataset.config import IndexConfig
from satmap_dataset.geoportal.http import RetryPolicy
from satmap_dataset.geoportal.wfs_client import get_capabilities, get_year_tiles
from satmap_dataset.models import IndexManifest, YearAvailabilityReport, YearStatus
from satmap_dataset.pipeline.aoi_preview import write_osm_preview_html
from satmap_dataset.pipeline.validator import evaluate_year_policy

logger = logging.getLogger("satmap_dataset.index")


def _write_json(path: Path, payload: IndexManifest | YearAvailabilityReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")


def _swap_bbox_axes_str(bbox: str) -> str:
    parts = [float(part.strip()) for part in bbox.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must have format xmin,ymin,xmax,ymax")
    min_x, min_y, max_x, max_y = parts
    return f"{min_y},{min_x},{max_y},{max_x}"


async def _probe_years_wfs_async(
    requested_years: list[int],
    bbox: str,
    srs: str,
) -> tuple[list[YearStatus], dict[int, dict[str, str]], dict[int, dict[str, list[float]]]]:
    logger.info("Index: probing WFS capabilities and yearly tiles for %d years", len(requested_years))
    retry_policy = RetryPolicy(max_attempts=10, backoff_seconds=1.2, jitter_seconds=0.8)
    _, year_to_typename = await get_capabilities(timeout=45.0, retry_policy=retry_policy)
    results: list[YearStatus] = []
    tile_sources_by_year: dict[int, dict[str, str]] = {}
    tile_bboxes_by_year: dict[int, dict[str, list[float]]] = {}
    for year in requested_years:
        # Geoportal WFS is sensitive to bursts even for sequential calls.
        await asyncio.sleep(random.uniform(0.6, 1.8))
        status, year_tiles, year_tile_bboxes = await get_year_tiles(
                year=year,
                bbox=bbox,
                srs=srs,
                year_to_typename=year_to_typename,
                timeout=45.0,
                retry_policy=retry_policy,
            )
        results.append(status)
        tile_sources_by_year[year] = year_tiles
        tile_bboxes_by_year[year] = year_tile_bboxes
        logger.info(
            "Index: year=%s status=%s features=%s tiles=%s",
            year,
            status.status,
            status.feature_count,
            len(year_tiles),
        )
    return results, tile_sources_by_year, tile_bboxes_by_year


def probe_years_wfs(
    aoi: str,
    year_start: int,
    year_end: int,
    srs: str = "EPSG:2180",
) -> list[YearStatus]:
    requested_years = list(range(year_start, year_end + 1))
    statuses, _, _ = asyncio.run(_probe_years_wfs_async(requested_years, aoi, srs))
    return statuses


def probe_years_wfs_with_tiles(
    aoi: str,
    year_start: int,
    year_end: int,
    srs: str = "EPSG:2180",
) -> tuple[list[YearStatus], dict[int, dict[str, str]], dict[int, dict[str, list[float]]]]:
    requested_years = list(range(year_start, year_end + 1))
    return asyncio.run(_probe_years_wfs_async(requested_years, aoi, srs))


def run(config: IndexConfig) -> tuple[int, Path]:
    logger.info(
        "Index: start year_start=%s year_end=%s bbox=%s strict_years=%s min_years=%s",
        config.year_start,
        config.year_end,
        config.bbox,
        config.strict_years,
        config.min_years,
    )
    run_parameters = config.model_dump(mode="json")
    bbox_for_wfs = config.bbox
    wfs_bbox_axes_swapped = False
    if config.experimental_wfs_swap_bbox_axes:
        # Legacy Geoportal clients sometimes expected WFS BBOX in Y/X order.
        # Keep this as explicit opt-in so users can force compatibility when needed.
        bbox_for_wfs = _swap_bbox_axes_str(config.bbox)
        wfs_bbox_axes_swapped = True
        logger.warning(
            "Index: experimental_wfs_swap_bbox_axes enabled; querying WFS with swapped axis order bbox=%s",
            bbox_for_wfs,
        )
    requested_years = config.requested_years
    probe_result = probe_years_wfs_with_tiles(
        aoi=bbox_for_wfs,
        year_start=config.year_start,
        year_end=config.year_end,
        srs=config.srs,
    )
    if len(probe_result) == 3:
        year_statuses, tile_sources_full, tile_bboxes_full = probe_result
    else:
        year_statuses, tile_sources_full = probe_result
        tile_bboxes_full = {}
    years_available_wfs = [entry.year for entry in year_statuses if entry.status == "has_features"]
    years_excluded_with_reason = {
        entry.year: (entry.reason or entry.status)
        for entry in year_statuses
        if entry.status != "has_features"
    }

    policy = evaluate_year_policy(
        requested_years=requested_years,
        available_years=years_available_wfs,
        strict_years=config.strict_years,
        min_years=config.min_years,
    )
    warnings = list(policy.warnings)
    errors = list(policy.errors)
    if config.experimental_wfs_swap_bbox_axes:
        warnings.append(
            "experimental_wfs_swap_bbox_axes enabled: WFS queried with swapped axis order ymin,xmin,ymax,xmax."
        )
    warnings.append(
        "WFS tile index and WMS TIME rendering are not equivalent: WFS reflects downloadable source tiles per year, while WMS TIME can render years even when WFS returns no features."
    )

    tile_sources_by_year = {year: tile_sources_full.get(year, {}) for year in policy.years_included}
    tile_bboxes_by_year = {year: tile_bboxes_full.get(year, {}) for year in policy.years_included}
    common_tile_ids: list[str] = []
    if policy.years_included:
        tile_id_sets = [set(tile_sources_by_year.get(year, {}).keys()) for year in policy.years_included]
        if tile_id_sets:
            common_tile_ids = sorted(set.intersection(*tile_id_sets))
    if not common_tile_ids:
        warnings.append(
            "No common_tile_ids found for included years; downstream stages will use per-year tile sets."
        )

    if not years_available_wfs:
        errors.append(
            "WFS returned zero features for the provided bbox and years. Verify bbox axis order (xmin,ymin,xmax,ymax) and SRS."
        )
    aoi_preview_html: str | None = None
    preview_warning: str | None = None
    preview_path = config.output_json.parent / "aoi_preview.html"
    try:
        write_osm_preview_html(
            bbox=config.bbox,
            srs=config.srs,
            output_path=preview_path,
        )
        aoi_preview_html = str(preview_path)
        logger.info("Index: wrote AOI preview=%s", preview_path)
    except Exception as exc:
        preview_warning = f"AOI preview was not generated: {exc}"
        warnings.append(preview_warning)
        logger.warning("Index: AOI preview generation failed: %s", exc)
    logger.info(
        "Index: years_included=%s common_tile_ids=%s",
        len(policy.years_included),
        len(common_tile_ids),
    )

    manifest = IndexManifest(
        year_start=config.year_start,
        year_end=config.year_end,
        bbox=config.bbox,
        srs=config.srs,
        strict_years=config.strict_years,
        min_years=config.min_years,
        wfs_bbox_axes_swapped=wfs_bbox_axes_swapped,
        years_requested=requested_years,
        year_statuses=year_statuses,
        years_available_wfs=years_available_wfs,
        years_included=policy.years_included,
        years_excluded_with_reason=years_excluded_with_reason,
        common_tile_ids=common_tile_ids,
        tile_sources_by_year=tile_sources_by_year,
        tile_bboxes_by_year=tile_bboxes_by_year,
        passed=policy.passed and bool(years_available_wfs),
        errors=errors,
        warnings=warnings,
        aoi_preview_html=aoi_preview_html,
        run_parameters=run_parameters,
    )
    year_report = YearAvailabilityReport(
        year_start=config.year_start,
        year_end=config.year_end,
        bbox=config.bbox,
        srs=config.srs,
        wfs_bbox_axes_swapped=wfs_bbox_axes_swapped,
        years_requested=requested_years,
        year_statuses=year_statuses,
        years_available_wfs=years_available_wfs,
        years_included=policy.years_included,
        years_excluded_with_reason=years_excluded_with_reason,
        strict_years=config.strict_years,
        min_years=config.min_years,
        passed=policy.passed,
        errors=policy.errors,
        warnings=policy.warnings + ([preview_warning] if preview_warning else []),
        aoi_preview_html=aoi_preview_html,
        run_parameters=run_parameters,
    )
    _write_json(config.year_availability_output_json, year_report)
    _write_json(config.output_json, manifest)
    logger.info("Index: wrote %s and %s", config.output_json, config.year_availability_output_json)
    logger.info("Index: finished passed=%s", manifest.passed)

    return (0 if manifest.passed else 1), config.output_json
