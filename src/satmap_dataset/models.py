from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


class YearStatus(BaseModel):
    year: int = Field(..., ge=1900)
    typename_exists: bool
    feature_count: int = Field(default=0, ge=0)
    status: Literal["no_typename", "zero_features", "has_features"] = "zero_features"
    reason: str | None = None


class IndexManifest(BaseModel):
    kind: Literal["index_manifest"] = "index_manifest"
    generated_at: datetime = Field(default_factory=_utc_now)
    year_start: int
    year_end: int
    bbox: str
    srs: str
    strict_years: bool = False
    min_years: int = 2
    years_requested: list[int]
    year_statuses: list[YearStatus]
    years_available_wfs: list[int]
    years_included: list[int]
    years_excluded_with_reason: dict[int, str] = Field(default_factory=dict)
    common_tile_ids: list[str] = Field(default_factory=list)
    tile_sources_by_year: dict[int, dict[str, str]] = Field(default_factory=dict)
    passed: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class DatasetManifest(BaseModel):
    kind: Literal["dataset_manifest"] = "dataset_manifest"
    stage: Literal["download", "mosaic", "render", "run"] = "download"
    generated_at: datetime = Field(default_factory=_utc_now)
    years_requested: list[int] = Field(default_factory=list)
    years_available_wfs: list[int] = Field(default_factory=list)
    years_included: list[int] = Field(default_factory=list)
    years_excluded_with_reason: dict[int, str] = Field(default_factory=dict)
    common_tile_ids: list[str] = Field(default_factory=list)
    tile_sources_by_year: dict[int, dict[str, str]] = Field(default_factory=dict)
    assets: list[str] = Field(default_factory=list)
    source_manifest: str | None = None
    mode: Literal["wms_tiled", "wfs_render", "hybrid"] = "hybrid"
    target_width: int | None = None
    target_height: int | None = None
    target_bbox: str | None = None
    target_srs: str | None = None
    profile: Literal["train", "reference"] | None = None
    px_per_meter: float | None = None
    years_source_map: dict[int, Literal["wfs", "wms", "wms_fallback"]] = Field(default_factory=dict)
    coverage_ratio_by_year: dict[int, float] = Field(default_factory=dict)
    color_qc_by_year: dict[int, dict[str, float | list[float] | None]] = Field(default_factory=dict)
    resample_method: str | None = None
    render_backend: Literal["pyvips"] | None = None
    asset_stats: dict[str, dict[str, int | str | None]] = Field(default_factory=dict)
    pixel_profile: str = "RGB_U8"
    wfs_global_calibration: bool = False
    calibration_reference_year: int | None = None
    calibration_transform: str | None = None
    calibration_source_axis_mode: Literal["normal", "swapped"] | None = None
    calibration_max_error_px: float | None = None
    calibration_fit_error_px: float | None = None
    calibration_matrix: list[float] | None = None
    calibration_report_path: str | None = None
    calibration_status_by_year: dict[int, Literal["applied", "skipped_wms", "disabled"]] = Field(default_factory=dict)
    calibration_error_px_by_year: dict[int, float] = Field(default_factory=dict)
    render_cache_signature: str | None = None
    diagnostics_report_path: str | None = None
    diagnostics_quicklook_dir: str | None = None
    passed: bool = True
    notes: str | None = None


class ValidationReport(BaseModel):
    kind: Literal["validation_report"] = "validation_report"
    generated_at: datetime = Field(default_factory=_utc_now)
    requested_years: list[int]
    years_included: list[int]
    years_excluded_with_reason: dict[int, str] = Field(default_factory=dict)
    missing_years: list[int]
    strict_years: bool = False
    min_years: int = 2
    passed: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class YearAvailabilityReport(BaseModel):
    kind: Literal["year_availability_report"] = "year_availability_report"
    generated_at: datetime = Field(default_factory=_utc_now)
    year_start: int
    year_end: int
    bbox: str
    srs: str
    years_requested: list[int]
    year_statuses: list[YearStatus]
    years_available_wfs: list[int]
    years_included: list[int]
    years_excluded_with_reason: dict[int, str] = Field(default_factory=dict)
    strict_years: bool = False
    min_years: int = 2
    passed: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
