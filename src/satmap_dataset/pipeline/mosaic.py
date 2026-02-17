from __future__ import annotations

from pathlib import Path

from satmap_dataset.config import MosaicConfig
from satmap_dataset.models import DatasetManifest


def _read_dataset_manifest(path: Path) -> DatasetManifest:
    return DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: DatasetManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")


def run(config: MosaicConfig) -> tuple[int, Path]:
    source_manifest = _read_dataset_manifest(config.dataset_manifest)
    years = source_manifest.years_included
    assets = list(source_manifest.assets)
    passed = source_manifest.passed and bool(years) and bool(assets)

    manifest = DatasetManifest(
        stage="mosaic",
        years_requested=source_manifest.years_requested,
        years_available_wfs=source_manifest.years_available_wfs,
        years_included=years,
        years_excluded_with_reason=source_manifest.years_excluded_with_reason,
        common_tile_ids=source_manifest.common_tile_ids,
        tile_sources_by_year=source_manifest.tile_sources_by_year,
        assets=assets,
        source_manifest=str(config.dataset_manifest),
        target_width=30000,
        target_height=30000,
        pixel_profile="RGB_U8",
        passed=passed,
        notes="Mosaic stage currently passes through downloaded TIFF asset list.",
    )
    _write_json(config.output_json, manifest)

    return (0 if manifest.passed else 1), config.output_json
