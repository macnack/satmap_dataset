from __future__ import annotations

from pathlib import Path
import logging

from satmap_dataset.config import (
    DownloadConfig,
    IndexConfig,
    RenderConfig,
    RunConfig,
    ValidateConfig,
)
from satmap_dataset.models import DatasetManifest, IndexManifest
from satmap_dataset.pipeline import downloader, index_builder, render, validator

logger = logging.getLogger("satmap_dataset.run")


def _load_index_manifest(path: Path) -> IndexManifest | None:
    if not path.exists():
        return None
    try:
        return IndexManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_dataset_manifest(path: Path) -> DatasetManifest | None:
    if not path.exists():
        return None
    try:
        return DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _can_reuse_index(manifest: IndexManifest, config: RunConfig) -> bool:
    return (
        manifest.passed
        and manifest.year_start == config.year_start
        and manifest.year_end == config.year_end
        and manifest.bbox == config.bbox
        and manifest.srs == config.srs
        and manifest.strict_years == config.strict_years
        and manifest.min_years == config.min_years
    )


def _asset_exists(asset: str, dataset_manifest_path: Path) -> bool:
    p = Path(asset)
    if p.is_absolute():
        return p.exists()
    if p.exists():
        return p.exists()
    return (dataset_manifest_path.parent / p).exists()


def _same_path_ref(a: str | None, b: Path, base: Path) -> bool:
    if not a:
        return False
    pa = Path(a)
    if not pa.is_absolute():
        pa = (base / pa).resolve()
    else:
        pa = pa.resolve()
    return pa == b.resolve()


def _can_reuse_download(manifest: DatasetManifest, config: RunConfig, index_output: Path, download_output: Path) -> bool:
    if not manifest.passed or manifest.stage != "download":
        return False
    if manifest.mode != config.mode:
        return False
    if not _same_path_ref(manifest.source_manifest, index_output, download_output.parent):
        return False
    if manifest.profile != config.profile:
        return False
    if config.profile == "reference":
        if manifest.target_bbox != config.bbox:
            return False
        if (manifest.target_srs or "").upper() != config.srs.upper():
            return False
    if not manifest.assets:
        return False
    return all(_asset_exists(asset, download_output) for asset in manifest.assets)


def _write_wms_only_index(config: RunConfig, index_output: Path, year_availability_output: Path) -> None:
    from satmap_dataset.models import YearAvailabilityReport, YearStatus

    requested_years = config.requested_years
    statuses = [
        YearStatus(
            year=year,
            typename_exists=False,
            feature_count=0,
            status="no_typename",
            reason="WFS probe skipped in mode=wms_tiled.",
        )
        for year in requested_years
    ]
    index_manifest = IndexManifest(
        year_start=config.year_start,
        year_end=config.year_end,
        bbox=config.bbox,
        srs=config.srs,
        strict_years=config.strict_years,
        min_years=config.min_years,
        years_requested=requested_years,
        year_statuses=statuses,
        years_available_wfs=[],
        years_included=requested_years,
        years_excluded_with_reason={},
        common_tile_ids=[],
        tile_sources_by_year={},
        passed=True,
        errors=[],
        warnings=["WFS index step skipped for mode=wms_tiled."],
    )
    year_report = YearAvailabilityReport(
        year_start=config.year_start,
        year_end=config.year_end,
        bbox=config.bbox,
        srs=config.srs,
        years_requested=requested_years,
        year_statuses=statuses,
        years_available_wfs=[],
        years_included=requested_years,
        years_excluded_with_reason={},
        strict_years=config.strict_years,
        min_years=config.min_years,
        passed=True,
        errors=[],
        warnings=["WFS year availability probe skipped for mode=wms_tiled."],
    )
    index_output.parent.mkdir(parents=True, exist_ok=True)
    year_availability_output.parent.mkdir(parents=True, exist_ok=True)
    index_output.write_text(index_manifest.model_dump_json(indent=2), encoding="utf-8")
    year_availability_output.write_text(year_report.model_dump_json(indent=2), encoding="utf-8")


def run(config: RunConfig) -> tuple[int, Path]:
    logger.info("Run: start year_start=%s year_end=%s bbox=%s", config.year_start, config.year_end, config.bbox)
    artifacts_dir = config.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    index_output = artifacts_dir / "index_manifest.json"
    year_availability_output = artifacts_dir / "year_availability_report.json"
    download_output = artifacts_dir / "dataset_manifest_download.json"
    render_output = artifacts_dir / "dataset_manifest_render.json"
    validate_output = artifacts_dir / "validation_report.json"

    existing_index = _load_index_manifest(index_output)
    if config.mode == "wms_tiled":
        if existing_index and _can_reuse_index(existing_index, config):
            logger.info("Run: reusing existing WMS-only index manifest=%s", index_output)
        else:
            _write_wms_only_index(config, index_output, year_availability_output)
            logger.info("Run: wrote WMS-only index manifest=%s", index_output)
    elif existing_index and _can_reuse_index(existing_index, config):
        logger.info("Run: reusing existing index manifest=%s", index_output)
    else:
        index_config = IndexConfig(
            year_start=config.year_start,
            year_end=config.year_end,
            bbox=config.bbox,
            srs=config.srs,
            strict_years=config.strict_years,
            experimental_wfs_swap_bbox_axes=config.experimental_wfs_swap_bbox_axes,
            min_years=config.min_years,
            output_json=index_output,
            year_availability_output_json=year_availability_output,
        )
        index_code, _ = index_builder.run(index_config)
        if index_code != 0:
            logger.error("Run: index failed output=%s", index_output)
            return index_code, index_output

    existing_download = _load_dataset_manifest(download_output)
    if existing_download and _can_reuse_download(existing_download, config, index_output, download_output):
        logger.info("Run: reusing existing download manifest=%s", download_output)
    else:
        download_config = DownloadConfig(
            index_manifest=index_output,
            download_root=config.download_root,
            mode=config.mode,
            profile=config.profile,
            bbox=config.bbox,
            srs=config.srs,
            px_per_meter=config.px_per_meter,
            wms_fallback_missing_years=config.wms_fallback_missing_years,
            concurrency=config.concurrency,
            retries=config.retries,
            retry_delay=config.retry_delay,
            timeout=config.timeout,
            sleep_min=config.sleep_min,
            sleep_max=config.sleep_max,
            overwrite=config.overwrite,
            output_json=download_output,
        )
        download_code, _ = downloader.run(download_config)
        if download_code != 0:
            logger.error("Run: download failed output=%s", download_output)
            return download_code, download_output

    render_config = RenderConfig(
        dataset_manifest=download_output,
        render_root=config.render_root,
        mode=config.mode,
        profile=config.profile,
        px_per_meter=config.px_per_meter,
        target_width=config.target_width,
        target_height=config.target_height,
        auto_size_from_bbox=config.auto_size_from_bbox,
        target_bbox=config.target_bbox,
        target_srs=config.target_srs,
        resample_method=config.resample_method,
        tile_size=config.tile_size,
        compression=config.compression,
        overview_levels=config.overview_levels,
        wms_fallback_missing_years=config.wms_fallback_missing_years,
        wfs_global_calibration=config.wfs_global_calibration,
        calibration_reference_year=config.calibration_reference_year,
        calibration_transform=config.calibration_transform,
        calibration_max_error_px=config.calibration_max_error_px,
        disable_color_norm=config.disable_color_norm,
        experimental_force_srgb_from_ycbcr=config.experimental_force_srgb_from_ycbcr,
        experimental_per_year_color_norm=config.experimental_per_year_color_norm,
        output_json=render_output,
    )
    render_code, _ = render.run(render_config)
    if render_code != 0:
        logger.error("Run: render failed output=%s", render_output)
        return render_code, render_output

    validate_config = ValidateConfig(
        dataset_manifest=render_output,
        requested_years=config.requested_years,
        strict_years=config.strict_years,
        min_years=config.min_years,
        output_json=validate_output,
    )
    validate_code, _ = validator.run(validate_config)
    logger.info("Run: finished validate_code=%s output=%s", validate_code, validate_output)
    return validate_code, validate_output
