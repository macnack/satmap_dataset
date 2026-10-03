"""Multi-layer location run (thin wrapper over the unified orchestrator)."""

from __future__ import annotations

import logging
from pathlib import Path

from satmap_dataset.config import DemConfig, OsmConfig, PipelineConfig, RunConfig
from satmap_dataset.pipeline import orchestrator

logger = logging.getLogger("satmap_dataset.location_run")


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
    """Produce RGB + optional DEM/OSM for one location, aligned to one grid.

    Preserves the historical return contract: artifact path is always
    ``<artifacts_dir>/rgb_layer_manifest.json``.
    """
    artifacts_dir = Path(artifacts_dir)
    # Keep rgb artifacts_dir aligned with the location artifacts root.
    if rgb_config.artifacts_dir != artifacts_dir:
        rgb_config = rgb_config.model_copy(update={"artifacts_dir": artifacts_dir})

    code, _pipeline_path = orchestrator.run(
        PipelineConfig(
            rgb=rgb_config,
            dem=dem_config,
            osm=osm_config,
            run_dem=bool(run_dem and dem_config is not None),
            run_osm=bool(run_osm and osm_config is not None),
            run_validate=validate,
        )
    )
    rgb_output = artifacts_dir / "rgb_layer_manifest.json"
    logger.info("run_location: finished code=%s rgb=%s", code, rgb_output)
    return code, rgb_output
