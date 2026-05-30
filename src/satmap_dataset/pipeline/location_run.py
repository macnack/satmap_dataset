from __future__ import annotations

import logging
from pathlib import Path

from satmap_dataset.config import DemConfig, OsmConfig, RunConfig, ValidateConfig
from satmap_dataset.layers import get_layer
from satmap_dataset.models import LayerManifest
from satmap_dataset.pipeline import validator

logger = logging.getLogger("satmap_dataset.location_run")


def _write_manifest(manifest: LayerManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def run_location(
    *,
    rgb_config: RunConfig,
    dem_config: DemConfig | None = None,
    osm_config: OsmConfig | None = None,
    artifacts_dir: Path,
    run_dem: bool = True,
    run_osm: bool = True,
    validate: bool = True,
) -> tuple[int, Path]:
    """Produce RGB + DEM + OSM for one location, aligned to a single grid.

    The RGB layer defines the ReferenceGrid; that grid is then handed to the
    DEM and OSM layers so every modality lands on the same NN-ready raster grid
    without any layer re-reading another's manifest from disk.

    Returns (exit_code, rgb_manifest_path). The exit code is the most severe
    (highest) among RGB (+ optional validation), DEM, and OSM, so a later
    failure is never masked by an earlier one.
    """
    artifacts_dir = Path(artifacts_dir)
    rgb_layer = get_layer(f"{rgb_config.provider}_rgb")
    code, rgb_manifest = rgb_layer.produce(rgb_config, grid=None)
    rgb_output = artifacts_dir / "rgb_layer_manifest.json"
    _write_manifest(rgb_manifest, rgb_output)
    if code != 0:
        logger.error("run_location: RGB layer failed code=%s", code)
        return code, rgb_output

    grid = rgb_manifest.grid
    overall = code

    if run_dem and dem_config is not None:
        dem_code, dem_manifest = get_layer("dem").produce(dem_config, grid)
        _write_manifest(dem_manifest, dem_config.output_json)
        overall = max(overall, dem_code)

    if run_osm and osm_config is not None:
        osm_code, osm_manifest = get_layer("osm").produce(osm_config, grid)
        _write_manifest(osm_manifest, osm_config.output_json)
        overall = max(overall, osm_code)

    if validate:
        validate_output = artifacts_dir / "validation_report.json"
        validate_config = ValidateConfig(
            dataset_manifest=rgb_output,
            requested_years=rgb_config.requested_years,
            strict_years=rgb_config.strict_years,
            min_years=rgb_config.min_years,
            output_json=validate_output,
        )
        validate_code, _ = validator.run(validate_config)
        overall = max(overall, validate_code)

    logger.info("run_location: finished code=%s rgb=%s", overall, rgb_output)
    return overall, rgb_output
