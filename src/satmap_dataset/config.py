from __future__ import annotations

from pathlib import Path
import re

from pydantic import BaseModel, Field, field_validator, model_validator


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

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value: str) -> str:
        return _validate_bbox(value)

    @model_validator(mode="after")
    def validate_year_range(self) -> "IndexConfig":
        if self.year_end < self.year_start:
            raise ValueError("year_end must be >= year_start")
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

    @model_validator(mode="after")
    def validate_sleep_range(self) -> "DownloadConfig":
        if self.sleep_max < self.sleep_min:
            raise ValueError("sleep_max must be >= sleep_min")
        allowed_modes = {"wms_tiled", "wfs_render", "hybrid"}
        if self.mode not in allowed_modes:
            raise ValueError(f"mode must be one of {sorted(allowed_modes)}")
        allowed_profiles = {"train", "reference"}
        if self.profile not in allowed_profiles:
            raise ValueError(f"profile must be one of {sorted(allowed_profiles)}")
        if (self.profile == "reference" or self.mode in {"wms_tiled", "hybrid"}) and self.bbox is None:
            raise ValueError("bbox is required for profile='reference' and for modes using WMS")
        if self.bbox is not None:
            _validate_bbox(self.bbox)
        self.force_wms_years = sorted(set(self.force_wms_years))
        if self.mode == "wfs_render" and self.force_wms_years:
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
        allowed_modes = {"wms_tiled", "wfs_render", "hybrid"}
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

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value: str) -> str:
        return _validate_bbox(value)

    @model_validator(mode="after")
    def validate_year_range(self) -> "RunConfig":
        if self.year_end < self.year_start:
            raise ValueError("year_end must be >= year_start")
        if self.sleep_max < self.sleep_min:
            raise ValueError("sleep_max must be >= sleep_min")
        if self.target_bbox is not None:
            _validate_bbox(self.target_bbox)
        allowed_modes = {"wms_tiled", "wfs_render", "hybrid"}
        if self.mode not in allowed_modes:
            raise ValueError(f"mode must be one of {sorted(allowed_modes)}")
        allowed_profiles = {"train", "reference"}
        if self.profile not in allowed_profiles:
            raise ValueError(f"profile must be one of {sorted(allowed_profiles)}")
        self.force_wms_years = sorted(set(self.force_wms_years))
        if self.mode == "wfs_render" and self.force_wms_years:
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
