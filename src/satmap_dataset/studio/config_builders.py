"""Build pipeline configs from studio UI fields."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from satmap_dataset.config import (
    DemAvailabilityConfig,
    DemConfig,
    IndexConfig,
    OsmConfig,
    RawExportConfig,
    RunConfig,
)
from satmap_dataset.studio.geo import utm_epsg_for_lon_lat

PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "geoportal": {
        "base_json": "configs/run/base.json",
        "srs": "EPSG:2180",
        "target_srs": "EPSG:2180",
    },
    "lantmateriet": {
        "base_json": "configs/run/base_lantmateriet.json",
        "srs": "EPSG:3006",
        "target_srs": "EPSG:3006",
    },
    "nls": {
        "base_json": "configs/run/base_nls.json",
        "srs": "EPSG:3067",
        "target_srs": "EPSG:3067",
    },
    "sentinel2": {
        "base_json": "configs/run/base_sentinel2.json",
        "srs": "EPSG:32633",
        "target_srs": "EPSG:32633",
    },
}


def repo_root_from_base(base_json: Path) -> Path:
    resolved = base_json.resolve()
    if len(resolved.parents) >= 3:
        return resolved.parents[2]
    return Path.cwd().resolve()


def resolve_base_json(provider: str, repo_root: Path) -> Path:
    preset = PROVIDER_PRESETS.get(provider)
    if preset is None:
        raise ValueError(f"Unknown provider: {provider!r}")
    path = Path(preset["base_json"])
    if not path.is_absolute():
        path = repo_root / path
    return path


def build_location_payload(
    *,
    location_name: str,
    center_lat: float,
    center_lon: float,
    area_km2: float,
    provider: str,
    year_start: int,
    year_end: int,
    px_per_meter: float,
    profile: str,
    mode: str | None = None,
    wms_fallback_missing_years: bool | None = None,
    min_years: int | None = None,
    strict_years: bool | None = None,
    concurrency: int | None = None,
    sleep_min: float | None = None,
    sleep_max: float | None = None,
    provider_options: dict[str, Any] | None = None,
    cell_mode: str | None = None,
    equalize_gsd: bool | None = None,
    raw_root: str | None = None,
) -> dict[str, Any]:
    preset = PROVIDER_PRESETS[provider]
    srs = preset["srs"]
    target_srs = preset["target_srs"]
    if provider == "sentinel2":
        utm = utm_epsg_for_lon_lat(center_lon, center_lat)
        srs = utm
        target_srs = utm

    payload: dict[str, Any] = {
        "location_name": location_name,
        "center_lat": center_lat,
        "center_lon": center_lon,
        "area_km2": area_km2,
        "provider": provider,
        "srs": srs,
        "target_srs": target_srs,
        "year_start": year_start,
        "year_end": year_end,
        "px_per_meter": px_per_meter,
        "profile": profile,
    }
    if mode is not None:
        payload["mode"] = mode
    if wms_fallback_missing_years is not None:
        payload["wms_fallback_missing_years"] = wms_fallback_missing_years
    if min_years is not None:
        payload["min_years"] = min_years
    if strict_years is not None:
        payload["strict_years"] = strict_years
    if concurrency is not None:
        payload["concurrency"] = concurrency
    if sleep_min is not None:
        payload["sleep_min"] = sleep_min
    if sleep_max is not None:
        payload["sleep_max"] = sleep_max
    if provider_options:
        payload["provider_options"] = provider_options
    if cell_mode is not None:
        payload["cell_mode"] = cell_mode
    if equalize_gsd is not None:
        payload["equalize_gsd"] = equalize_gsd
    if raw_root is not None:
        payload["raw_root"] = raw_root
    return payload


def merge_base_and_location_payload(
    base_json: Path,
    location_payload: dict[str, Any],
    *,
    resolve_bbox: bool = True,
) -> dict[str, Any]:
    from satmap_dataset.cli import (
        _apply_location_paths_policy,
        _load_params_json_dict,
        _resolve_json_center_bbox,
    )

    base_payload = _load_params_json_dict(base_json)
    merged: dict[str, object] = dict(base_payload)
    merged.update(location_payload)
    repo_root = repo_root_from_base(base_json)
    merged = _apply_location_paths_policy(merged, repo_root)
    if resolve_bbox:
        merged = _resolve_json_center_bbox(merged, required=True)
    return merged


def build_index_config(location_payload: dict[str, Any], base_json: Path) -> IndexConfig:
    merged = merge_base_and_location_payload(base_json, location_payload)
    artifacts_dir = Path(str(merged.get("artifacts_dir")))
    merged.setdefault("output_json", str(artifacts_dir / "index_manifest.json"))
    merged.setdefault(
        "year_availability_output_json",
        str(artifacts_dir / "year_availability_report.json"),
    )
    return IndexConfig.model_validate(merged)


def build_run_config(location_payload: dict[str, Any], base_json: Path) -> RunConfig:
    merged = merge_base_and_location_payload(base_json, location_payload)
    return RunConfig.model_validate(merged)


def build_dem_config(location_payload: dict[str, Any], base_json: Path) -> DemConfig:
    merged = merge_base_and_location_payload(base_json, location_payload)
    dem_root = Path(str(merged.get("dem_root", "dem")))
    merged.setdefault("output_json", str(dem_root / "dem_manifest.json"))
    artifacts_dir = merged.get("artifacts_dir")
    if artifacts_dir is not None and merged.get("align_to_render", True):
        merged.setdefault(
            "render_manifest",
            str(Path(str(artifacts_dir)) / "dataset_manifest_render.json"),
        )
    return DemConfig.model_validate(merged)


def build_osm_config(location_payload: dict[str, Any], base_json: Path) -> OsmConfig:
    merged = merge_base_and_location_payload(base_json, location_payload)
    osm_root = Path(str(merged.get("osm_root", "osm")))
    merged.setdefault("output_json", str(osm_root / "osm_manifest.json"))
    return OsmConfig.model_validate(merged)


def build_dem_availability_config(
    location_payload: dict[str, Any],
    base_json: Path,
) -> DemAvailabilityConfig:
    merged = merge_base_and_location_payload(base_json, location_payload)
    artifacts_dir = Path(str(merged.get("artifacts_dir")))
    merged.setdefault("output_json", str(artifacts_dir / "dem_availability.json"))
    return DemAvailabilityConfig.model_validate(merged)


def build_raw_export_config(location_payload: dict[str, Any], base_json: Path) -> RawExportConfig:
    from satmap_dataset.cli import _slugify_location_name

    merged = merge_base_and_location_payload(base_json, location_payload, resolve_bbox=False)
    location_name = merged.get("location_name")
    if location_name is None:
        raise ValueError("location_name is required for raw export")
    merged.setdefault("area", _slugify_location_name(str(location_name)))
    artifacts_dir = Path(str(merged.get("artifacts_dir")))
    merged.setdefault(
        "download_manifest",
        str(artifacts_dir / "dataset_manifest_download.json"),
    )
    merged.setdefault("output_json", str(artifacts_dir / "raw_export_manifest.json"))
    allowed = set(RawExportConfig.model_fields)
    cleaned = {k: v for k, v in merged.items() if k in allowed}
    return RawExportConfig.model_validate(cleaned)


def write_location_json(
    location_payload: dict[str, Any],
    repo_root: Path,
    *,
    slug: str,
) -> Path:
    out_dir = repo_root / "configs/run/locations"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{slug}.json"
    # Only persist location-centric fields for CLI replay
    persist_keys = {
        "location_name",
        "center_lat",
        "center_lon",
        "area_km2",
        "provider",
        "srs",
        "target_srs",
        "year_start",
        "year_end",
        "px_per_meter",
        "profile",
        "cell_mode",
        "equalize_gsd",
    }
    to_write = {k: v for k, v in location_payload.items() if k in persist_keys and v is not None}
    path.write_text(json.dumps(to_write, indent=2) + "\n", encoding="utf-8")
    return path


def write_merged_run_json(
    location_payload: dict[str, Any],
    base_json: Path,
    repo_root: Path,
    *,
    slug: str,
) -> Path:
    merged = merge_base_and_location_payload(base_json, location_payload)
    out_dir = repo_root / "configs/run/generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{slug}.run.json"
    path.write_text(json.dumps(merged, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def validate_location_payload(location_payload: dict[str, Any], base_json: Path) -> None:
    """Raise ValidationError if configs cannot be built."""
    build_index_config(location_payload, base_json)
