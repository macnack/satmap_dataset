"""Unified location pipeline: RGB (grid) → optional DEM/OSM → optional validate."""

from __future__ import annotations

import logging
from pathlib import Path

from satmap_dataset.config import PipelineConfig, ValidateConfig
from satmap_dataset.io.config_hash import run_config_hash
from satmap_dataset.layers import get_layer
from satmap_dataset.models import PipelineManifest
from satmap_dataset.pipeline import validator
from satmap_dataset.progress_report import report_log, report_progress

logger = logging.getLogger("satmap_dataset.orchestrator")


def _write_json(model: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2), encoding="utf-8")  # type: ignore[attr-defined]


def run(config: PipelineConfig) -> tuple[int, Path]:
    """Run the unified pipeline for one location / AOI.

    Always writes ``<artifacts_dir>/pipeline_manifest.json`` and returns its path
    as the stage artifact (CLI last-line contract). RGB failure short-circuits;
    DEM/OSM/validate failures accumulate via ``max(exit_code)``.
    """
    rgb_config = config.rgb
    artifacts_dir = Path(rgb_config.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    pipeline_output = artifacts_dir / "pipeline_manifest.json"
    rgb_output = artifacts_dir / "rgb_layer_manifest.json"

    layers_requested = list(config.layers_requested)
    total_steps = len(layers_requested) + int(config.run_validate)
    step = 0
    layer_artifacts: dict[str, str] = {}
    layers_completed: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []
    failed_artifact: str | None = None
    grid = None
    overall = 0

    report_progress(step, max(total_steps, 1), "RGB layer (index → download → render)…")
    report_log("Starting RGB layer")
    rgb_name = f"{rgb_config.provider}_rgb"
    rgb_layer = get_layer(rgb_name)
    code, rgb_manifest = rgb_layer.produce(rgb_config, grid=None)
    _write_json(rgb_manifest, rgb_output)
    layer_artifacts[rgb_name] = str(rgb_output)
    if code != 0:
        errors.append(f"RGB layer failed with exit code {code}")
        failed_artifact = rgb_manifest.source_manifest or str(rgb_output)
        manifest = PipelineManifest(
            layers_requested=layers_requested,
            layers_completed=[],
            layer_artifacts=layer_artifacts,
            run_validate=config.run_validate,
            failed_artifact=failed_artifact,
            passed=False,
            errors=errors,
            run_parameters=_run_parameters(config),
        )
        _write_json(manifest, pipeline_output)
        logger.error("orchestrator: RGB failed code=%s artifact=%s", code, failed_artifact)
        return code, pipeline_output

    layers_completed.append(rgb_name)
    grid = rgb_manifest.grid
    overall = code

    if config.run_dem and config.dem is not None:
        step += 1
        report_progress(step, total_steps, "DEM layer…")
        report_log("Starting DEM layer")
        dem_code, dem_manifest = get_layer("dem").produce(config.dem, grid)
        dem_path = Path(config.dem.output_json)
        _write_json(dem_manifest, dem_path)
        layer_artifacts["dem"] = str(dem_path)
        if dem_code == 0:
            layers_completed.append("dem")
        else:
            errors.append(f"DEM layer failed with exit code {dem_code}")
        overall = max(overall, dem_code)

    if config.run_osm and config.osm is not None:
        step += 1
        report_progress(step, total_steps, "OSM label layer…")
        report_log("Starting OSM layer")
        osm_code, osm_manifest = get_layer("osm").produce(config.osm, grid)
        osm_path = Path(config.osm.output_json)
        _write_json(osm_manifest, osm_path)
        layer_artifacts["osm"] = str(osm_path)
        if osm_code == 0:
            layers_completed.append("osm")
        else:
            errors.append(f"OSM layer failed with exit code {osm_code}")
        overall = max(overall, osm_code)

    validation_report: str | None = None
    if config.run_validate:
        step += 1
        report_progress(step, total_steps, "Validating RGB output…")
        report_log("Starting validation")
        validate_output = artifacts_dir / "validation_report.json"
        validate_config = ValidateConfig(
            dataset_manifest=rgb_output,
            requested_years=rgb_config.requested_years,
            strict_years=rgb_config.strict_years,
            min_years=rgb_config.min_years,
            output_json=validate_output,
            config_hash=run_config_hash(rgb_config),
        )
        validate_code, _ = validator.run(validate_config)
        validation_report = str(validate_output)
        if validate_code != 0:
            errors.append(f"Validation failed with exit code {validate_code}")
            failed_artifact = failed_artifact or validation_report
        overall = max(overall, validate_code)

    report_progress(total_steps, total_steps, "Pipeline finished")
    manifest = PipelineManifest(
        layers_requested=layers_requested,
        layers_completed=layers_completed,
        layer_artifacts=layer_artifacts,
        grid=grid,
        run_validate=config.run_validate,
        validation_report=validation_report,
        failed_artifact=failed_artifact,
        passed=overall == 0,
        errors=errors,
        warnings=warnings,
        run_parameters=_run_parameters(config),
    )
    _write_json(manifest, pipeline_output)
    logger.info(
        "orchestrator: finished code=%s layers=%s output=%s",
        overall,
        layers_completed,
        pipeline_output,
    )
    return overall, pipeline_output


def _run_parameters(config: PipelineConfig) -> dict:
    return {
        "layers_requested": config.layers_requested,
        "run_dem": config.run_dem,
        "run_osm": config.run_osm,
        "run_validate": config.run_validate,
        "rgb": config.rgb.model_dump(mode="json"),
        "dem": config.dem.model_dump(mode="json") if config.dem is not None else None,
        "osm": config.osm.model_dump(mode="json") if config.osm is not None else None,
    }


def run_rgb_only(rgb_config, *, run_validate: bool = True) -> tuple[int, Path]:
    """Convenience: RGB (+ optional validate) with no DEM/OSM layers."""
    return run(
        PipelineConfig(
            rgb=rgb_config,
            run_dem=False,
            run_osm=False,
            run_validate=run_validate,
        )
    )
