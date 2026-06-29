from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

PROVIDER_GEOPORTAL = "geoportal"
PROVIDER_LANTMATERIET = "lantmateriet"
PROVIDER_SENTINEL2 = "sentinel2"
PROVIDER_NLS = "nls"
PROVIDER_LROC_NAC = "lroc_nac"
ALLOWED_PROVIDERS = {
    PROVIDER_GEOPORTAL,
    PROVIDER_LANTMATERIET,
    PROVIDER_SENTINEL2,
    PROVIDER_NLS,
    PROVIDER_LROC_NAC,
}


def _validate_bbox(value: str) -> str:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must have format xmin,ymin,xmax,ymax")

    try:
        xmin, ymin, xmax, ymax = (float(part) for part in parts)
    except ValueError as exc:
        raise ValueError("bbox coordinates must be numeric") from exc

    if xmin >= xmax or ymin >= ymax:
        raise ValueError("bbox must satisfy xmin<xmax and ymin<ymax")

    return value


_NLS_NATIVE_SRS = "EPSG:3067"


def _validate_provider_srs(provider: str, srs: str) -> None:
    if provider == "nls" and srs.upper() != _NLS_NATIVE_SRS:
        raise ValueError(
            f"provider='nls' requires srs='{_NLS_NATIVE_SRS}' (NLS WCS/OAPIF "
            f"only accept TM35FIN coordinates); got srs={srs!r}. "
            "Reproject your bbox to EPSG:3067 before configuring an NLS run."
        )
    if provider == "lroc_nac" and not srs.upper().startswith("IAU_2015:301"):
        raise ValueError(
            f"provider='lroc_nac' requires a lunar IAU_2015:301xx CRS "
            f"(e.g. 'IAU_2015:30100'); got srs={srs!r}."
        )


class IndexConfig(BaseModel):
    year_start: int = Field(..., ge=1900)
    year_end: int = Field(..., ge=1900)
    bbox: str
    srs: str = "EPSG:2180"
    strict_years: bool = False
    experimental_wfs_swap_bbox_axes: bool = False
    min_years: int = Field(default=1, ge=1)
    output_json: Path = Path("artifacts/index_manifest.json")
    year_availability_output_json: Path = Path("artifacts/year_availability_report.json")
    provider: str = PROVIDER_GEOPORTAL
    provider_options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        if value not in ALLOWED_PROVIDERS:
            raise ValueError(f"provider must be one of {sorted(ALLOWED_PROVIDERS)}")
        return value

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value: str) -> str:
        return _validate_bbox(value)

    @model_validator(mode="after")
    def validate_year_range(self) -> "IndexConfig":
        if self.year_end < self.year_start:
            raise ValueError("year_end must be >= year_start")
        _validate_provider_srs(self.provider, self.srs)
        return self

    @property
    def requested_years(self) -> list[int]:
        return list(range(self.year_start, self.year_end + 1))


class DownloadConfig(BaseModel):
    index_manifest: Path = Path("artifacts/index_manifest.json")
    download_root: Path = Path("downloads")
    mode: str = "hybrid"
    profile: str = "train"
    bbox: str | None = None
    srs: str = "EPSG:2180"
    px_per_meter: float = Field(default=15.0, gt=0.0)
    wms_fallback_missing_years: bool = True
    force_wms_years: list[int] = Field(default_factory=list)
    concurrency: int = Field(default=6, ge=1, le=64)
    retries: int = Field(default=3, ge=0, le=20)
    retry_delay: float = Field(default=1.0, gt=0.0)
    timeout: float = Field(default=120.0, gt=0.0)
    sleep_min: float = Field(default=0.6, ge=0.0)
    sleep_max: float = Field(default=2.2, ge=0.0)
    overwrite: bool = False
    output_json: Path = Path("artifacts/dataset_manifest_download.json")
    provider: str = PROVIDER_GEOPORTAL
    provider_options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        if value not in ALLOWED_PROVIDERS:
            raise ValueError(f"provider must be one of {sorted(ALLOWED_PROVIDERS)}")
        return value

    @model_validator(mode="after")
    def validate_sleep_range(self) -> "DownloadConfig":
        if self.sleep_max < self.sleep_min:
            raise ValueError("sleep_max must be >= sleep_min")
        _validate_provider_srs(self.provider, self.srs)
        if self.provider == PROVIDER_GEOPORTAL:
            allowed_modes = {"wms_tiled", "wfs_render", "hybrid"}
            if self.mode not in allowed_modes:
                raise ValueError(f"mode must be one of {sorted(allowed_modes)} for provider=geoportal")
        allowed_profiles = {"train", "reference"}
        if self.profile not in allowed_profiles:
            raise ValueError(f"profile must be one of {sorted(allowed_profiles)}")
        if self.provider == PROVIDER_GEOPORTAL:
            if (self.profile == "reference" or self.mode in {"wms_tiled", "hybrid"}) and self.bbox is None:
                raise ValueError("bbox is required for profile='reference' and for modes using WMS")
        if self.bbox is not None:
            _validate_bbox(self.bbox)
        self.force_wms_years = sorted(set(self.force_wms_years))
        if self.provider == PROVIDER_GEOPORTAL and self.mode == "wfs_render" and self.force_wms_years:
            raise ValueError("force_wms_years requires mode 'hybrid' or 'wms_tiled'")
        return self


class MosaicConfig(BaseModel):
    dataset_manifest: Path = Path("artifacts/dataset_manifest_download.json")
    target_width: int = Field(default=30000, ge=1)
    target_height: int = Field(default=30000, ge=1)
    pixel_profile: str = "RGB_U8"
    output_json: Path = Path("artifacts/dataset_manifest_mosaic.json")


class RenderConfig(BaseModel):
    dataset_manifest: Path = Path("artifacts/dataset_manifest_download.json")
    render_root: Path = Path("rendered")
    mode: str = "hybrid"
    profile: str = "train"
    px_per_meter: float = Field(default=15.0, gt=0.0)
    target_width: int | None = Field(default=None, ge=1)
    target_height: int | None = Field(default=None, ge=1)
    auto_size_from_bbox: bool = True
    target_bbox: str | None = None
    target_srs: str = "EPSG:2180"
    resample_method: str = "bilinear"
    tile_size: int = Field(default=512, ge=64)
    compression: str = "deflate"
    overview_levels: list[int] = Field(default_factory=lambda: [2, 4, 8, 16])
    wms_fallback_missing_years: bool = True
    disable_color_norm: bool = False
    experimental_force_srgb_from_ycbcr: bool = False
    experimental_per_year_color_norm: bool = False
    output_json: Path = Path("artifacts/dataset_manifest_render.json")

    @field_validator("target_bbox")
    @classmethod
    def validate_target_bbox(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return _validate_bbox(value)

    @field_validator("resample_method")
    @classmethod
    def validate_resample_method(cls, value: str) -> str:
        allowed = {"bilinear", "nearest"}
        if value not in allowed:
            raise ValueError(f"resample_method must be one of {sorted(allowed)}")
        return value

    @field_validator("compression")
    @classmethod
    def validate_compression(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized == "deflate":
            return normalized
        if normalized == "jpeg":
            return normalized
        if re.fullmatch(r"jpeg([1-9]\d?|100)", normalized):
            return normalized
        raise ValueError("compression must be 'deflate', 'jpeg', or 'jpegNN' (e.g. jpeg95)")

    @field_validator("overview_levels")
    @classmethod
    def validate_overviews(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("overview_levels must not be empty")
        if any(level <= 1 for level in value):
            raise ValueError("overview_levels must contain integers > 1")
        return sorted(set(value))

    @model_validator(mode="after")
    def validate_render_profile(self) -> "RenderConfig":
        allowed_modes = {"wms_tiled", "wfs_render", "hybrid", "stac"}
        if self.mode not in allowed_modes:
            raise ValueError(f"mode must be one of {sorted(allowed_modes)}")
        allowed_profiles = {"train", "reference"}
        if self.profile not in allowed_profiles:
            raise ValueError(f"profile must be one of {sorted(allowed_profiles)}")
        if (self.target_width is None) != (self.target_height is None):
            raise ValueError("target_width and target_height must be set together")
        if not self.auto_size_from_bbox and (self.target_width is None or self.target_height is None):
            raise ValueError("target_width and target_height are required when auto_size_from_bbox is False")
        return self


class ValidateConfig(BaseModel):
    dataset_manifest: Path = Path("artifacts/dataset_manifest_render.json")
    requested_years: list[int] = Field(default_factory=list)
    strict_years: bool = False
    min_years: int = Field(default=1, ge=1)
    output_json: Path = Path("artifacts/validation_report.json")

    @field_validator("requested_years")
    @classmethod
    def sort_unique_years(cls, value: list[int]) -> list[int]:
        return sorted(set(value))


class RunConfig(BaseModel):
    year_start: int = Field(..., ge=1900)
    year_end: int = Field(..., ge=1900)
    bbox: str
    srs: str = "EPSG:2180"
    strict_years: bool = False
    experimental_wfs_swap_bbox_axes: bool = False
    min_years: int = Field(default=1, ge=1)
    mode: str = "hybrid"
    profile: str = "train"
    px_per_meter: float = Field(default=15.0, gt=0.0)
    wms_fallback_missing_years: bool = True
    force_wms_years: list[int] = Field(default_factory=list)
    disable_color_norm: bool = False
    target_width: int | None = Field(default=None, ge=1)
    target_height: int | None = Field(default=None, ge=1)
    auto_size_from_bbox: bool = True
    pixel_profile: str = "RGB_U8"
    render_root: Path = Path("rendered")
    target_bbox: str | None = None
    target_srs: str = "EPSG:2180"
    resample_method: str = "bilinear"
    tile_size: int = Field(default=512, ge=64)
    compression: str = "deflate"
    overview_levels: list[int] = Field(default_factory=lambda: [2, 4, 8, 16])
    experimental_force_srgb_from_ycbcr: bool = False
    experimental_per_year_color_norm: bool = False
    download_root: Path = Path("downloads")
    concurrency: int = Field(default=6, ge=1, le=64)
    retries: int = Field(default=3, ge=0, le=20)
    retry_delay: float = Field(default=1.0, gt=0.0)
    timeout: float = Field(default=120.0, gt=0.0)
    sleep_min: float = Field(default=0.6, ge=0.0)
    sleep_max: float = Field(default=2.2, ge=0.0)
    overwrite: bool = False
    artifacts_dir: Path = Path("artifacts")
    provider: str = PROVIDER_GEOPORTAL
    provider_options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value: str) -> str:
        return _validate_bbox(value)

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        if value not in ALLOWED_PROVIDERS:
            raise ValueError(f"provider must be one of {sorted(ALLOWED_PROVIDERS)}")
        return value

    @model_validator(mode="after")
    def validate_year_range(self) -> "RunConfig":
        if self.year_end < self.year_start:
            raise ValueError("year_end must be >= year_start")
        if self.sleep_max < self.sleep_min:
            raise ValueError("sleep_max must be >= sleep_min")
        if self.target_bbox is not None:
            _validate_bbox(self.target_bbox)
        if self.provider == PROVIDER_GEOPORTAL:
            allowed_modes = {"wms_tiled", "wfs_render", "hybrid"}
            if self.mode not in allowed_modes:
                raise ValueError(f"mode must be one of {sorted(allowed_modes)} for provider=geoportal")
        allowed_profiles = {"train", "reference"}
        if self.profile not in allowed_profiles:
            raise ValueError(f"profile must be one of {sorted(allowed_profiles)}")
        self.force_wms_years = sorted(set(self.force_wms_years))
        if self.provider == PROVIDER_GEOPORTAL and self.mode == "wfs_render" and self.force_wms_years:
            raise ValueError("force_wms_years requires mode 'hybrid' or 'wms_tiled'")
        allowed_resample = {"bilinear", "nearest"}
        if self.resample_method not in allowed_resample:
            raise ValueError(f"resample_method must be one of {sorted(allowed_resample)}")
        compression = self.compression.strip().lower()
        if compression not in {"deflate", "jpeg"} and not re.fullmatch(r"jpeg([1-9]\d?|100)", compression):
            raise ValueError("compression must be 'deflate', 'jpeg', or 'jpegNN' (e.g. jpeg95)")
        self.compression = compression
        if not self.overview_levels or any(level <= 1 for level in self.overview_levels):
            raise ValueError("overview_levels must contain integers > 1")
        if (self.target_width is None) != (self.target_height is None):
            raise ValueError("target_width and target_height must be set together")
        if not self.auto_size_from_bbox and (self.target_width is None or self.target_height is None):
            raise ValueError("target_width and target_height are required when auto_size_from_bbox is False")
        return self

    @property
    def requested_years(self) -> list[int]:
        return list(range(self.year_start, self.year_end + 1))


class DemConfig(BaseModel):
    bbox: str
    srs: str = "EPSG:2180"
    transport: str = "wcs"
    year_start: int | None = Field(default=None, ge=1900)
    year_end: int | None = Field(default=None, ge=1900)
    products: list[str] = Field(default_factory=lambda: ["nmt", "nmpt"])
    vertical_datum: str = "evrf2007"
    dem_root: Path = Path("dem")
    align_to_render: bool = True
    render_manifest: Path | None = None
    target_bbox: str | None = None
    target_width: int | None = Field(default=None, ge=1)
    target_height: int | None = Field(default=None, ge=1)
    px_per_meter: float = Field(default=1.0, gt=0.0)
    max_request_px: int = Field(default=2048, ge=1)
    overwrite: bool = False
    timeout: float = Field(default=120.0, gt=0.0)
    retries: int = Field(default=6, ge=1, le=20)
    retry_delay: float = Field(default=1.0, gt=0.0)
    sleep_min: float = Field(default=0.6, ge=0.0)
    sleep_max: float = Field(default=2.2, ge=0.0)
    location_name: str | None = None
    output_json: Path = Path("dem/dem_manifest.json")
    provider: str = PROVIDER_GEOPORTAL
    provider_options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("transport")
    @classmethod
    def validate_transport(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized not in {"wcs", "skorowidz"}:
            raise ValueError("transport must be 'wcs' or 'skorowidz'")
        return normalized

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value: str) -> str:
        return _validate_bbox(value)

    @field_validator("target_bbox")
    @classmethod
    def validate_target_bbox(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return _validate_bbox(value)

    @field_validator("products")
    @classmethod
    def validate_products(cls, value: list[str]) -> list[str]:
        allowed = {"nmt", "nmpt"}
        normalized = [str(item).strip().lower() for item in value]
        if not normalized:
            raise ValueError("products must not be empty")
        bad = [item for item in normalized if item not in allowed]
        if bad:
            raise ValueError(f"products must be a subset of {sorted(allowed)}; got {bad}")
        seen: list[str] = []
        for item in normalized:
            if item not in seen:
                seen.append(item)
        return seen

    @field_validator("vertical_datum")
    @classmethod
    def validate_vertical_datum(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        allowed = {"evrf2007", "kron86"}
        if normalized not in allowed:
            raise ValueError(f"vertical_datum must be one of {sorted(allowed)}")
        return normalized

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        if value not in ALLOWED_PROVIDERS:
            raise ValueError(f"provider must be one of {sorted(ALLOWED_PROVIDERS)}")
        return value

    @model_validator(mode="after")
    def validate_invariants(self) -> "DemConfig":
        if self.sleep_max < self.sleep_min:
            raise ValueError("sleep_max must be >= sleep_min")
        if (self.target_width is None) != (self.target_height is None):
            raise ValueError("target_width and target_height must be set together")
        if self.transport == "skorowidz":
            if self.year_start is None or self.year_end is None:
                raise ValueError("year_start and year_end are required when transport='skorowidz'")
            if self.year_end < self.year_start:
                raise ValueError("year_end must be >= year_start")
        return self

    @property
    def requested_years(self) -> list[int]:
        if self.year_start is None or self.year_end is None:
            return []
        return list(range(self.year_start, self.year_end + 1))


class DemAvailabilityConfig(BaseModel):
    bbox: str
    srs: str = "EPSG:2180"
    products: list[str] = Field(default_factory=lambda: ["nmt", "nmpt"])
    datums: list[str] = Field(default_factory=lambda: ["evrf2007", "kron86"])
    year_start: int | None = Field(default=None, ge=1900)
    year_end: int | None = Field(default=None, ge=1900)
    location_name: str | None = None
    timeout: float = Field(default=45.0, gt=0.0)
    retries: int = Field(default=6, ge=1, le=20)
    retry_delay: float = Field(default=1.0, gt=0.0)
    sleep_min: float = Field(default=0.6, ge=0.0)
    sleep_max: float = Field(default=2.2, ge=0.0)
    output_json: Path = Path("artifacts/dem_availability.json")
    provider: str = PROVIDER_GEOPORTAL
    provider_options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value: str) -> str:
        return _validate_bbox(value)

    @field_validator("products")
    @classmethod
    def validate_products(cls, value: list[str]) -> list[str]:
        allowed = {"nmt", "nmpt"}
        normalized = [str(v).strip().lower() for v in value]
        if not normalized:
            raise ValueError("products must not be empty")
        bad = [v for v in normalized if v not in allowed]
        if bad:
            raise ValueError(f"products must be a subset of {sorted(allowed)}; got {bad}")
        seen: list[str] = []
        for v in normalized:
            if v not in seen:
                seen.append(v)
        return seen

    @field_validator("datums")
    @classmethod
    def validate_datums(cls, value: list[str]) -> list[str]:
        allowed = {"evrf2007", "kron86"}
        normalized = [str(v).strip().lower() for v in value]
        if not normalized:
            raise ValueError("datums must not be empty")
        bad = [v for v in normalized if v not in allowed]
        if bad:
            raise ValueError(f"datums must be a subset of {sorted(allowed)}; got {bad}")
        seen: list[str] = []
        for v in normalized:
            if v not in seen:
                seen.append(v)
        return seen

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        if value not in ALLOWED_PROVIDERS:
            raise ValueError(f"provider must be one of {sorted(ALLOWED_PROVIDERS)}")
        return value

    @model_validator(mode="after")
    def validate_invariants(self) -> "DemAvailabilityConfig":
        if self.sleep_max < self.sleep_min:
            raise ValueError("sleep_max must be >= sleep_min")
        if self.year_start is not None and self.year_end is not None and self.year_end < self.year_start:
            raise ValueError("year_end must be >= year_start")
        return self

    @property
    def requested_years(self) -> list[int] | None:
        if self.year_start is None or self.year_end is None:
            return None
        return list(range(self.year_start, self.year_end + 1))


_OSM_ALLOWED_CATEGORIES: frozenset[str] = frozenset({"buildings", "roads", "paths", "green", "water"})


class OsmConfig(BaseModel):
    bbox: str
    srs: str = "EPSG:2180"
    osm_root: Path = Path("osm")
    output_json: Path = Path("osm/osm_manifest.json")
    render_manifest: Path | None = None
    year_date_map: dict[int, str] | None = None
    categories: list[str] = Field(default_factory=lambda: ["buildings", "roads", "paths", "green", "water"])
    target_bbox: str | None = None
    target_width: int | None = Field(default=None, ge=1)
    target_height: int | None = Field(default=None, ge=1)
    overpass_url: str = "https://overpass.kumi.systems/api/interpreter"
    timeout: float = Field(default=60.0, gt=0.0)
    retries: int = Field(default=3, ge=1, le=20)
    retry_delay: float = Field(default=2.0, gt=0.0)
    sleep_min: float = Field(default=1.0, ge=0.0)
    sleep_max: float = Field(default=3.0, ge=0.0)
    overwrite: bool = False
    location_name: str | None = None
    provider_options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value: str) -> str:
        return _validate_bbox(value)

    @field_validator("categories")
    @classmethod
    def validate_categories(cls, value: list[str]) -> list[str]:
        normalized = [str(item).strip().lower() for item in value]
        if not normalized:
            raise ValueError("categories must not be empty")
        bad = [c for c in normalized if c not in _OSM_ALLOWED_CATEGORIES]
        if bad:
            raise ValueError(f"unknown categories {bad}; allowed: {sorted(_OSM_ALLOWED_CATEGORIES)}")
        seen: list[str] = []
        for c in normalized:
            if c not in seen:
                seen.append(c)
        return seen

    @model_validator(mode="after")
    def validate_invariants(self) -> "OsmConfig":
        if self.sleep_max < self.sleep_min:
            raise ValueError("sleep_max must be >= sleep_min")
        if (self.target_width is None) != (self.target_height is None):
            raise ValueError("target_width and target_height must be set together")
        if self.target_bbox is not None:
            _validate_bbox(self.target_bbox)
        return self


def _default_raw_root() -> Path:
    env = os.environ.get("SATMAP_RAW_ROOT")
    if env:
        return Path(env).expanduser()
    return Path("~/Github/sat_data_raw").expanduser()


class RawExportConfig(BaseModel):
    """Input config for the opt-in raw-export stage.

    Exports native download tiles into <raw_root>/<provider>/<area>/<year>/ and
    ingests co-located season-cell stacks. sentinel2 is rejected (not a raw
    orthophoto tile provider).
    """

    provider: str
    area: str
    download_root: Path
    download_manifest: Path | None = None
    raw_root: Path = Field(default_factory=_default_raw_root)
    min_coverage: float | None = None
    link_mode: str = "symlink"
    cell_mode: str = "footprint"
    equalize_gsd: bool = True
    cell_size_m: float | None = None
    aoi_bbox: str | None = None
    min_aoi_overlap: float = 0.25
    artifacts_dir: Path = Path("artifacts")
    output_json: Path = Path("artifacts/raw_export_manifest.json")

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        if value == PROVIDER_SENTINEL2:
            raise ValueError("provider 'sentinel2' is not a raw-orthophoto-tile provider")
        if value not in ALLOWED_PROVIDERS:
            raise ValueError(f"provider must be one of {sorted(ALLOWED_PROVIDERS)}")
        return value

    @field_validator("link_mode")
    @classmethod
    def validate_link_mode(cls, value: str) -> str:
        if value not in {"symlink", "copy"}:
            raise ValueError("link_mode must be 'symlink' or 'copy'")
        return value

    @field_validator("cell_mode")
    @classmethod
    def validate_cell_mode(cls, value: str) -> str:
        if value not in {"footprint", "world_window"}:
            raise ValueError("cell_mode must be 'footprint' or 'world_window'")
        return value

    @field_validator("min_coverage")
    @classmethod
    def validate_min_coverage(cls, value: float | None) -> float | None:
        if value is not None and not (0.0 < value <= 1.0):
            raise ValueError("min_coverage must be in (0, 1]")
        return value

    @field_validator("cell_size_m")
    @classmethod
    def validate_cell_size(cls, value: float | None) -> float | None:
        if value is not None and value <= 0.0:
            raise ValueError("cell_size_m must be > 0")
        return value

    @field_validator("aoi_bbox")
    @classmethod
    def validate_aoi_bbox(cls, value: str | None) -> str | None:
        return _validate_bbox(value) if value is not None else None

    @field_validator("min_aoi_overlap")
    @classmethod
    def validate_min_aoi_overlap(cls, value: float) -> float:
        if not (0.0 <= value <= 1.0):
            raise ValueError("min_aoi_overlap must be in [0, 1]")
        return value

    @field_validator("area")
    @classmethod
    def validate_area(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("area must be non-empty")
        return value


class TrajectoryConfig(BaseModel):
    track_path: Path
    output_dir: Path
    cell_km: float = Field(default=1.0, gt=0.0)
    srs: str = "EPSG:2180"
    year_start: int = Field(default=2020, ge=1900)
    year_end: int = Field(default=2025, ge=1900)
    download: bool = False
    preview: bool = True
    mode: str = "hybrid"
    profile: str = "train"
    wms_fallback_missing_years: bool = True
    concurrency: int = Field(default=6, ge=1, le=64)
    retries: int = Field(default=3, ge=0, le=20)
    retry_delay: float = Field(default=1.0, gt=0.0)
    timeout: float = Field(default=120.0, gt=0.0)
    sleep_min: float = Field(default=0.6, ge=0.0)
    sleep_max: float = Field(default=2.2, ge=0.0)
    overwrite: bool = False

    @model_validator(mode="after")
    def _validate(self) -> "TrajectoryConfig":
        if self.year_end < self.year_start:
            raise ValueError("year_end must be >= year_start")
        if self.sleep_max < self.sleep_min:
            raise ValueError("sleep_max must be >= sleep_min")
        return self
