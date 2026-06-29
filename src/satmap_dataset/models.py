from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError


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


class ReferenceGrid(BaseModel):
    """The shared NN-ready grid every modality (RGB/DEM/OSM) aligns to.

    Computed once by the grid-defining (RGB) layer and passed explicitly to the
    other layers, replacing the previous implicit coupling where DEM/OSM stages
    re-read the render manifest JSON to discover the target grid.
    """

    bbox: str  # "xmin,ymin,xmax,ymax" in `srs` axis order
    width: int = Field(..., ge=1)
    height: int = Field(..., ge=1)
    srs: str
    px_per_meter: float | None = None
    # year -> snapshot date (max per-year acquisition date); consumed by OSM.
    year_date_map: dict[int, str] = Field(default_factory=dict)

    @classmethod
    def from_render_manifest(cls, manifest: Any) -> "ReferenceGrid":
        """Build a grid from a render-stage manifest's target_* fields.

        Duck-typed: reads `target_bbox`/`target_width`/`target_height`/
        `target_srs`/`px_per_meter` and derives `year_date_map` as the max
        acquisition date per year from `tile_acquisition_by_year` (mirrors the
        historical `osm._read_year_date_map` logic).
        """
        target_bbox = getattr(manifest, "target_bbox", None)
        target_width = getattr(manifest, "target_width", None)
        target_height = getattr(manifest, "target_height", None)
        if not target_bbox or target_width is None or target_height is None:
            raise ValueError(
                "Render manifest is missing target_bbox/target_width/target_height; "
                "cannot derive a ReferenceGrid."
            )
        year_date_map: dict[int, str] = {}
        tile_acq = getattr(manifest, "tile_acquisition_by_year", None) or {}
        for year, acq in tile_acq.items():
            dates = [
                v.acquisition_date
                for v in acq.values()
                if getattr(v, "acquisition_date", None) is not None
            ]
            if dates:
                year_date_map[int(year)] = max(dates)
        return cls(
            bbox=str(target_bbox),
            width=int(target_width),
            height=int(target_height),
            srs=str(getattr(manifest, "target_srs", None) or "EPSG:2180"),
            px_per_meter=getattr(manifest, "px_per_meter", None),
            year_date_map=year_date_map,
        )


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
    mode: Literal["wms_tiled", "wfs_render", "hybrid", "wcs", "stac"] = "hybrid"
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


class DemYearAsset(BaseModel):
    year: int = Field(..., ge=1900)
    native_path: str | None = None
    native_width: int | None = None
    native_height: int | None = None
    aligned_path: str | None = None
    aligned_width: int | None = None
    aligned_height: int | None = None
    tile_count: int = Field(default=0, ge=0)
    mean_height_error: float | None = None  # not populated yet (would require extending wfs_client return)
    godla: list[str] = Field(default_factory=list)
    passed: bool = False
    errors: list[str] = Field(default_factory=list)


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
    years: list[DemYearAsset] = Field(default_factory=list)


class DemAvailabilityEntry(BaseModel):
    product: Literal["nmt", "nmpt"]
    datum: Literal["evrf2007", "kron86"]
    year: int = Field(..., ge=1900)
    godla: list[str] = Field(default_factory=list)
    tile_count: int = Field(default=0, ge=0)
    formats: list[str] = Field(default_factory=list)
    coverage: Literal["full", "partial", "none"] = "none"
    coverage_pct: float = 0.0
    acquisition_dates: list[str] = Field(default_factory=list)


class DemAvailabilityReport(BaseModel):
    kind: Literal["dem_availability"] = "dem_availability"
    generated_at: datetime = Field(default_factory=_utc_now)
    provider: str = "geoportal"
    aoi_bbox: str
    srs: str
    entries: list[DemAvailabilityEntry] = Field(default_factory=list)
    errors: dict[str, str] = Field(default_factory=dict)
    full_coverage_options: list[dict[str, Any]] = Field(default_factory=list)
    run_parameters: dict[str, Any] = Field(default_factory=dict)


class OsmCategoryAsset(BaseModel):
    feature_count: int = Field(default=0, ge=0)
    raster_path: str | None = None


class OsmYearAsset(BaseModel):
    year: int
    snapshot_date: str
    categories: dict[str, OsmCategoryAsset] = Field(default_factory=dict)
    passed: bool = False
    errors: list[str] = Field(default_factory=list)


class LayerYearAsset(BaseModel):
    """Per-year output of a single layer (one modality), aligned to the grid."""

    year: int = Field(..., ge=1900)
    snapshot_date: str | None = None
    source: str | None = None  # e.g. "wfs" | "wms" | "stac" | "skorowidz" | "wcs"
    assets: dict[str, str] = Field(default_factory=dict)  # band/category -> raster path
    native_paths: dict[str, str] = Field(default_factory=dict)  # DEM product -> native path
    feature_counts: dict[str, int] = Field(default_factory=dict)  # OSM category -> count
    color_qc: dict[str, float | list[float] | None] = Field(default_factory=dict)
    acquisition: dict[str, TileAcquisitionMetadata] = Field(default_factory=dict)
    passed: bool = False
    errors: list[str] = Field(default_factory=list)


class LayerManifest(BaseModel):
    """Unified terminal manifest for every modality (RGB / DEM / OSM labels).

    Replaces the previous per-modality DatasetManifest / DemManifest /
    OsmManifest. Carries the shared ReferenceGrid plus per-year provenance so a
    downstream multi-band assembler can compose layers without re-deriving the
    grid or re-reading stage-specific schemas.
    """

    kind: Literal["layer_manifest"] = "layer_manifest"
    layer: str  # registry name: "geoportal_rgb" | "lantmateriet_rgb" | "dem" | "osm"
    role: Literal["rgb", "dem", "labels"]
    stage: Literal["download", "render", "run", "dem", "osm"] = "run"
    generated_at: datetime = Field(default_factory=_utc_now)
    provider: str | None = None  # None for provider-less layers (osm)
    grid: ReferenceGrid | None = None
    bands: list[str] = Field(default_factory=list)
    years_requested: list[int] = Field(default_factory=list)
    years_included: list[int] = Field(default_factory=list)
    years_excluded_with_reason: dict[int, str] = Field(default_factory=dict)
    years_source_map: dict[int, str] = Field(default_factory=dict)
    years: list[LayerYearAsset] = Field(default_factory=list)
    assets: list[str] = Field(default_factory=list)  # flat raster list (validator/reuse)
    source_manifest: str | None = None
    pixel_profile: str = "RGB_U8"
    target_srs: str | None = None  # convenience mirror of grid.srs
    render_cache_signature: str | None = None
    diagnostics_report_path: str | None = None
    diagnostics_quicklook_dir: str | None = None
    passed: bool = False
    notes: str | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    run_parameters: dict[str, Any] = Field(default_factory=dict)
    provider_metadata: dict[str, Any] = Field(default_factory=dict)

    # --- Internal download->render handoff fields (geoportal/RGB plumbing) ---
    # Empty for terminal DEM/OSM layers; carried so the RGB pipeline's
    # index/download/render stages exchange tile-level detail through one schema.
    years_available_wfs: list[int] = Field(default_factory=list)
    common_tile_ids: list[str] = Field(default_factory=list)
    tile_sources_by_year: dict[int, dict[str, str]] = Field(default_factory=dict)
    tile_bboxes_by_year: dict[int, dict[str, list[float]]] = Field(default_factory=dict)
    tile_acquisition_by_year: dict[int, dict[str, TileAcquisitionMetadata]] = Field(
        default_factory=dict
    )
    mode: Literal["wms_tiled", "wfs_render", "hybrid", "stac", "ode"] = "hybrid"
    target_width: int | None = None
    target_height: int | None = None
    target_bbox: str | None = None
    profile: Literal["train", "reference"] | None = None
    px_per_meter: float | None = None
    forced_wms_years: list[int] = Field(default_factory=list)
    color_qc_by_year: dict[int, dict[str, float | list[float] | None]] = Field(
        default_factory=dict
    )
    resample_method: str | None = None
    render_backend: Literal["pyvips"] | None = None
    asset_stats: dict[str, dict[str, int | str | None]] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str) -> "LayerManifest":
        """Parse a LayerManifest from disk with a clear error for stale formats.

        A pre-migration manifest (kind ``dataset_manifest``/``dem_manifest``/
        ``osm_manifest``) fails the ``layer_manifest`` discriminator; surface a
        re-run hint instead of a raw pydantic ValidationError.
        """
        text = Path(path).read_text(encoding="utf-8")
        try:
            return cls.model_validate_json(text)
        except ValidationError as exc:
            try:
                kind = json.loads(text).get("kind")
            except Exception:
                kind = None
            if kind and kind != "layer_manifest":
                raise ValueError(
                    f"{path} has kind={kind!r}, not a LayerManifest — this looks like a "
                    "pre-migration manifest. Re-run the producing stage (render/dem/osm) "
                    "to regenerate it."
                ) from exc
            raise


class RawExportManifest(BaseModel):
    """On-disk JSON contract for `raw_export_manifest.json` (the raw-export stage artifact)."""

    kind: Literal["raw_export_manifest"] = "raw_export_manifest"
    stage: Literal["raw_export"] = "raw_export"
    generated_at: datetime = Field(default_factory=_utc_now)
    provider: str
    area: str
    raw_root: str
    epsg: int | None = None
    epsg_provider_mismatch: bool = False
    link_mode: str = "symlink"
    cell_mode: str = "footprint"
    equalize_gsd: bool = True
    min_coverage: float | None = None
    cell_size_m: list[float] | None = None
    exported_tile_counts_by_year: dict[int, int] = Field(default_factory=dict)
    cells_produced: int = 0
    seasons_kept: int = 0
    seasons_dropped: int = 0
    per_area_manifest_path: str | None = None
    cell_dirs: list[str] = Field(default_factory=list)
    source_download_manifest: str | None = None
    passed: bool = False
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    run_parameters: dict[str, Any] = Field(default_factory=dict)


class CellEntry(BaseModel):
    name: str
    ix: int
    iy: int
    bbox: str  # "xmin,ymin,xmax,ymax" in `srs` axis order
    bbox_wgs84: str  # "lon_min,lat_min,lon_max,lat_max"
    center_lat: float
    center_lon: float
    download_status: str | None = None  # None|"ok"|"failed"|"skipped"


class TrajectoryManifest(BaseModel):
    track_path: str
    point_count: int = Field(..., ge=0)
    srs: str
    cell_m: float = Field(..., gt=0.0)
    year_start: int = Field(..., ge=1900)
    year_end: int = Field(..., ge=1900)
    union_bbox_2180: str
    cell_count: int = Field(..., ge=0)
    cells: list[CellEntry] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=_utc_now)
