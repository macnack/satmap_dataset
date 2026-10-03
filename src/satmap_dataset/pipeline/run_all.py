"""RGB-only run entry (thin wrapper over the unified orchestrator)."""

from __future__ import annotations

import logging
from pathlib import Path

from satmap_dataset.config import PipelineConfig, RunConfig
from satmap_dataset.models import PipelineManifest
from satmap_dataset.pipeline import orchestrator, rgb_pipeline

# Re-export reuse helpers for tests / back-compat.
_can_reuse_index = rgb_pipeline._can_reuse_index
_can_reuse_download = rgb_pipeline._can_reuse_download
_index_manifest_has_swapped_tile_bboxes = rgb_pipeline._index_manifest_has_swapped_tile_bboxes
_run_rgb_pipeline = rgb_pipeline.run_rgb_pipeline
run_rgb_pipeline = rgb_pipeline.run_rgb_pipeline

logger = logging.getLogger("satmap_dataset.run")


def run(config: RunConfig) -> tuple[int, Path]:
    """Index → download → render → validate for RGB only.

    Preserves the historical return contract: on success (or validate failure)
    the artifact path is ``validation_report.json``; on RGB short-circuit it is
    the failing stage path when available.
    """
    code, pipeline_path = orchestrator.run(
        PipelineConfig(rgb=config, run_dem=False, run_osm=False, run_validate=True)
    )
    validate_output = config.artifacts_dir / "validation_report.json"
    if code != 0:
        try:
            pm = PipelineManifest.model_validate_json(
                pipeline_path.read_text(encoding="utf-8")
            )
        except Exception:
            return code, pipeline_path
        if pm.failed_artifact:
            return code, Path(pm.failed_artifact)
        if validate_output.exists():
            return code, validate_output
        return code, pipeline_path
    logger.info("Run: finished validate_code=%s output=%s", code, validate_output)
    return code, validate_output
