from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.models import (
    DatasetManifest,
    IndexManifest,
    ValidationReport,
    YearAvailabilityReport,
    YearStatus,
)
from satmap_dataset.config import RenderConfig
import pytest


def test_year_status_json_roundtrip() -> None:
    source = YearStatus(
        year=2023,
        typename_exists=True,
        feature_count=12,
        status="has_features",
        reason=None,
    )
    payload = source.model_dump_json()
    restored = YearStatus.model_validate_json(payload)

    assert restored == source
    assert restored.year == 2023
    assert restored.typename_exists is True


def test_manifest_models_json_roundtrip() -> None:
    index_manifest = IndexManifest(
        year_start=2015,
        year_end=2016,
        bbox="210300,521900,210500,522100",
        srs="EPSG:2180",
        strict_years=False,
        min_years=2,
        years_requested=[2015, 2016],
        year_statuses=[
            YearStatus(year=2015, typename_exists=True, feature_count=5, status="has_features"),
            YearStatus(
                year=2016,
                typename_exists=True,
                feature_count=0,
                status="zero_features",
                reason="No features",
            ),
        ],
        years_available_wfs=[2015],
        years_included=[2015],
        years_excluded_with_reason={2016: "No features"},
        common_tile_ids=["placeholder"],
        tile_sources_by_year={2015: {"placeholder": "https://example.com/2015.tif"}},
        passed=False,
        errors=["min_years policy failed"],
        run_parameters={"bbox": "210300,521900,210500,522100", "min_years": 2},
    )
    dataset_manifest = DatasetManifest(
        stage="download",
        years_requested=[2015, 2016],
        years_available_wfs=[2015],
        years_included=[2015],
        years_excluded_with_reason={2016: "No features"},
        common_tile_ids=["placeholder"],
        tile_sources_by_year={2015: {"placeholder": "https://example.com/2015.tif"}},
        assets=["downloads/2015/placeholder_tile.tif"],
        source_manifest="artifacts/index_manifest.json",
        profile="reference",
        px_per_meter=15.0,
        years_source_map={2015: "wfs"},
        color_qc_by_year={2015: {"mean_rgb": [120.0, 121.0, 118.0], "delta_to_wms_reference": None}},
        passed=True,
        run_parameters={"mode": "hybrid", "bbox": "210300,521900,210500,522100"},
    )
    validation_report = ValidationReport(
        requested_years=[2015, 2016],
        years_included=[2015],
        years_excluded_with_reason={2016: "No features"},
        missing_years=[2016],
        strict_years=False,
        min_years=1,
        passed=True,
        run_parameters={"strict_years": False, "min_years": 1},
    )
    availability_report = YearAvailabilityReport(
        year_start=2015,
        year_end=2016,
        bbox="210300,521900,210500,522100",
        srs="EPSG:2180",
        years_requested=[2015, 2016],
        year_statuses=index_manifest.year_statuses,
        years_available_wfs=[2015],
        years_included=[2015],
        years_excluded_with_reason={2016: "No features"},
        strict_years=False,
        min_years=1,
        passed=True,
        run_parameters={"strict_years": False, "min_years": 1},
    )

    restored_index = IndexManifest.model_validate_json(index_manifest.model_dump_json())
    restored_dataset = DatasetManifest.model_validate_json(dataset_manifest.model_dump_json())
    restored_report = ValidationReport.model_validate_json(validation_report.model_dump_json())
    restored_availability = YearAvailabilityReport.model_validate_json(
        availability_report.model_dump_json()
    )

    assert restored_index.year_start == 2015
    assert restored_index.run_parameters["min_years"] == 2
    assert restored_dataset.stage == "download"
    assert restored_dataset.run_parameters["mode"] == "hybrid"
    assert restored_dataset.profile == "reference"
    assert restored_report.missing_years == [2016]
    assert restored_report.run_parameters["min_years"] == 1
    assert restored_availability.years_available_wfs == [2015]
    assert restored_availability.run_parameters["min_years"] == 1


def test_required_fields_and_types() -> None:
    year_status_fields = YearStatus.model_fields
    assert year_status_fields["year"].is_required()
    assert year_status_fields["typename_exists"].is_required()
    assert year_status_fields["year"].annotation is int
    assert year_status_fields["typename_exists"].annotation is bool

    index_fields = IndexManifest.model_fields
    assert index_fields["year_statuses"].is_required()
    assert index_fields["years_available_wfs"].is_required()
    assert index_fields["passed"].is_required()

    dataset_fields = DatasetManifest.model_fields
    assert dataset_fields["stage"].annotation is not None
    assert dataset_fields["years_included"].annotation is not None

    report_fields = ValidationReport.model_fields
    assert report_fields["requested_years"].is_required()
    assert report_fields["years_included"].is_required()
    assert report_fields["passed"].is_required()


def test_render_config_requires_both_target_dimensions() -> None:
    with pytest.raises(ValueError):
        RenderConfig(
            dataset_manifest=Path("artifacts/dataset_manifest_download.json"),
            target_width=3000,
            target_height=None,
        )


def test_render_config_requires_dimensions_when_auto_size_disabled() -> None:
    with pytest.raises(ValueError):
        RenderConfig(
            dataset_manifest=Path("artifacts/dataset_manifest_download.json"),
            auto_size_from_bbox=False,
        )


def test_render_config_accepts_jpeg95_compression() -> None:
    config = RenderConfig(
        dataset_manifest=Path("artifacts/dataset_manifest_download.json"),
        compression="jpeg95",
    )
    assert config.compression == "jpeg95"
