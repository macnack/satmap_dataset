from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


class YearStatus(BaseModel):
    year: int = Field(..., ge=1900)
    typename_exists: bool
    feature_count: int = Field(default=0, ge=0)
    status: Literal["no_typename", "zero_features", "has_features"] = "zero_features"
    reason: str | None = None


class TileAcquisitionMetadata(BaseModel):
    acquisition_date: str | None = None
    publication_date: str | None = None
    acquisition_year: int | None = None


class IndexManifest(BaseModel):
    kind: Literal["index_manifest"] = "index_manifest"
    generated_at: datetime = Field(default_factory=_utc_now)
    provider: str = "geoportal"
    year_start: int
    year_end: int
    bbox: str
    srs: str
    strict_years: bool = False
    min_years: int = 1
    wfs_bbox_axes_swapped: bool = False
    years_requested: list[int]
    year_statuses: list[YearStatus]
    years_available_wfs: list[int]
    years_included: list[int]
    years_excluded_with_reason: dict[int, str] = Field(default_factory=dict)
    common_tile_ids: list[str] = Field(default_factory=list)
    tile_sources_by_year: dict[int, dict[str, str]] = Field(default_factory=dict)
    tile_bboxes_by_year: dict[int, dict[str, list[float]]] = Field(default_factory=dict)
    tile_acquisition_by_year: dict[int, dict[str, TileAcquisitionMetadata]] = Field(default_factory=dict)
    passed: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    aoi_preview_html: str | None = None
    aoi_preview_png: str | None = None
    run_parameters: dict[str, Any] = Field(default_factory=dict)
    provider_metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetManifest(BaseModel):
    kind: Literal["dataset_manifest"] = "dataset_manifest"
    stage: Literal["download", "mosaic", "render", "run"] = "download"
    generated_at: datetime = Field(default_factory=_utc_now)
    provider: str = "geoportal"
    years_requested: list[int] = Field(default_factory=list)
    years_available_wfs: list[int] = Field(default_factory=list)
    years_included: list[int] = Field(default_factory=list)
    years_excluded_with_reason: dict[int, str] = Field(default_factory=dict)
    common_tile_ids: list[str] = Field(default_factory=list)
    tile_sources_by_year: dict[int, dict[str, str]] = Field(default_factory=dict)
    tile_bboxes_by_year: dict[int, dict[str, list[float]]] = Field(default_factory=dict)
    tile_acquisition_by_year: dict[int, dict[str, TileAcquisitionMetadata]] = Field(default_factory=dict)
    assets: list[str] = Field(default_factory=list)
    source_manifest: str | None = None
    mode: Literal["wms_tiled", "wfs_render", "hybrid", "stac"] = "hybrid"
    target_width: int | None = None
    target_height: int | None = None
    target_bbox: str | None = None
    target_srs: str | None = None
    profile: Literal["train", "reference"] | None = None
    px_per_meter: float | None = None
    years_source_map: dict[int, str] = Field(default_factory=dict)
    forced_wms_years: list[int] = Field(default_factory=list)
    color_qc_by_year: dict[int, dict[str, float | list[float] | None]] = Field(default_factory=dict)
    resample_method: str | None = None
    render_backend: Literal["pyvips"] | None = None
    asset_stats: dict[str, dict[str, int | str | None]] = Field(default_factory=dict)
    pixel_profile: str = "RGB_U8"
    render_cache_signature: str | None = None
    diagnostics_report_path: str | None = None
    diagnostics_quicklook_dir: str | None = None
    passed: bool = True
    notes: str | None = None
    run_parameters: dict[str, Any] = Field(default_factory=dict)
    provider_metadata: dict[str, Any] = Field(default_factory=dict)


class ValidationReport(BaseModel):
    kind: Literal["validation_report"] = "validation_report"
    generated_at: datetime = Field(default_factory=_utc_now)
    requested_years: list[int]
    years_included: list[int]
    years_excluded_with_reason: dict[int, str] = Field(default_factory=dict)
    missing_years: list[int]
    strict_years: bool = False
    min_years: int = 1
    passed: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    run_parameters: dict[str, Any] = Field(default_factory=dict)


class YearAvailabilityReport(BaseModel):
    kind: Literal["year_availability_report"] = "year_availability_report"
    generated_at: datetime = Field(default_factory=_utc_now)
    year_start: int
    year_end: int
    bbox: str
    srs: str
    wfs_bbox_axes_swapped: bool = False
    years_requested: list[int]
    year_statuses: list[YearStatus]
    years_available_wfs: list[int]
    years_included: list[int]
    years_excluded_with_reason: dict[int, str] = Field(default_factory=dict)
    strict_years: bool = False
    min_years: int = 1
    passed: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    aoi_preview_html: str | None = None
    aoi_preview_png: str | None = None
    run_parameters: dict[str, Any] = Field(default_factory=dict)


class DemProductAsset(BaseModel):
    product: Literal["nmt", "nmpt"]
    coverage_id: str
    endpoint: str
    native_path: str | None = None
    native_width: int | None = None
    native_height: int | None = None
    aligned_path: str | None = None
    aligned_width: int | None = None
    aligned_height: int | None = None
    tile_count: int = Field(default=0, ge=0)
    nodata: float | None = None  # not yet populated; would require WCS DescribeCoverage
    passed: bool = False
    errors: list[str] = Field(default_factory=list)


class DemManifest(BaseModel):
    kind: Literal["dem_manifest"] = "dem_manifest"
    stage: Literal["dem"] = "dem"
    generated_at: datetime = Field(default_factory=_utc_now)
    provider: str = "geoportal"
    bbox: str
    srs: str
    vertical_datum: str
    products: list[DemProductAsset] = Field(default_factory=list)
    align_to_render: bool = True
    passed: bool = False
    notes: str | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    run_parameters: dict[str, Any] = Field(default_factory=dict)
