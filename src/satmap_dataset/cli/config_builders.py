from __future__ import annotations

from pathlib import Path

import typer

from satmap_dataset.config import (
    DemAvailabilityConfig,
    DemConfig,
    DownloadConfig,
    IndexConfig,
    OsmConfig,
    RawExportConfig,
    RenderConfig,
    RunConfig,
    ValidateConfig,
)
from satmap_dataset.cli.common import (
    _apply_location_paths_policy,
    _load_params_json_dict,
    _resolve_json_center_bbox,
    _slugify_location_name,
)


def _build_run_config_from_base_and_location(*, base_json: Path, location_json: Path) -> RunConfig:
    base_payload = _load_params_json_dict(base_json)
    location_payload = _load_params_json_dict(location_json)
    merged: dict[str, object] = dict(base_payload)
    merged.update(location_payload)
    repo_root = base_json.resolve().parents[2] if len(base_json.resolve().parents) >= 3 else Path.cwd().resolve()
    merged = _apply_location_paths_policy(merged, repo_root)
    merged = _resolve_json_center_bbox(merged, required=True)
    return RunConfig.model_validate(merged)


def _build_index_config_from_base_and_location(*, base_json: Path, location_json: Path) -> IndexConfig:
    base_payload = _load_params_json_dict(base_json)
    location_payload = _load_params_json_dict(location_json)
    merged: dict[str, object] = dict(base_payload)
    merged.update(location_payload)
    repo_root = base_json.resolve().parents[2] if len(base_json.resolve().parents) >= 3 else Path.cwd().resolve()
    merged = _apply_location_paths_policy(merged, repo_root)
    merged = _resolve_json_center_bbox(merged, required=True)
    artifacts_dir = Path(str(merged.get("artifacts_dir")))
    merged.setdefault("output_json", str(artifacts_dir / "index_manifest.json"))
    merged.setdefault("year_availability_output_json", str(artifacts_dir / "year_availability_report.json"))
    return IndexConfig.model_validate(merged)


def _build_download_config_from_base_and_location(*, base_json: Path, location_json: Path) -> DownloadConfig:
    base_payload = _load_params_json_dict(base_json)
    location_payload = _load_params_json_dict(location_json)
    merged: dict[str, object] = dict(base_payload)
    merged.update(location_payload)
    repo_root = base_json.resolve().parents[2] if len(base_json.resolve().parents) >= 3 else Path.cwd().resolve()
    merged = _apply_location_paths_policy(merged, repo_root)
    merged = _resolve_json_center_bbox(merged, required=False)
    artifacts_dir = Path(str(merged.get("artifacts_dir")))
    merged.setdefault("index_manifest", str(artifacts_dir / "index_manifest.json"))
    merged.setdefault("output_json", str(artifacts_dir / "dataset_manifest_download.json"))
    return DownloadConfig.model_validate(merged)


def _build_validate_config_from_base_and_location(*, base_json: Path, location_json: Path) -> ValidateConfig:
    base_payload = _load_params_json_dict(base_json)
    location_payload = _load_params_json_dict(location_json)
    merged: dict[str, object] = dict(base_payload)
    merged.update(location_payload)
    repo_root = base_json.resolve().parents[2] if len(base_json.resolve().parents) >= 3 else Path.cwd().resolve()
    merged = _apply_location_paths_policy(merged, repo_root)
    artifacts_dir = Path(str(merged.get("artifacts_dir")))

    requested_years: list[int] = []
    if "requested_years" in merged:
        raw = merged.get("requested_years")
        if isinstance(raw, list):
            requested_years = [int(value) for value in raw]
    elif "year_start" in merged and "year_end" in merged:
        year_start = int(merged["year_start"])
        year_end = int(merged["year_end"])
        requested_years = list(range(year_start, year_end + 1))

    payload = {
        "dataset_manifest": str(merged.get("dataset_manifest", artifacts_dir / "dataset_manifest_render.json")),
        "requested_years": requested_years,
        "strict_years": bool(merged.get("strict_years", False)),
        "min_years": int(merged.get("min_years", 1)),
        "output_json": str(merged.get("validation_output_json", artifacts_dir / "validation_report.json")),
    }
    return ValidateConfig.model_validate(payload)


def _build_render_config_from_base_and_location(*, base_json: Path, location_json: Path) -> RenderConfig:
    base_payload = _load_params_json_dict(base_json)
    location_payload = _load_params_json_dict(location_json)
    merged: dict[str, object] = dict(base_payload)
    merged.update(location_payload)
    repo_root = base_json.resolve().parents[2] if len(base_json.resolve().parents) >= 3 else Path.cwd().resolve()
    merged = _apply_location_paths_policy(merged, repo_root)
    artifacts_dir = Path(str(merged.get("artifacts_dir")))
    merged.setdefault("dataset_manifest", str(artifacts_dir / "dataset_manifest_download.json"))
    merged.setdefault("output_json", str(artifacts_dir / "dataset_manifest_render.json"))
    return RenderConfig.model_validate(merged)


def _build_raw_export_config_from_base_and_location(*, base_json: Path, location_json: Path) -> RawExportConfig:
    base_payload = _load_params_json_dict(base_json)
    location_payload = _load_params_json_dict(location_json)
    merged: dict[str, object] = dict(base_payload)
    merged.update(location_payload)
    repo_root = base_json.resolve().parents[2] if len(base_json.resolve().parents) >= 3 else Path.cwd().resolve()
    merged = _apply_location_paths_policy(merged, repo_root)
    location_name = merged.get("location_name")
    if location_name is None:
        raise typer.BadParameter("location JSON must set 'location_name'")
    merged.setdefault("area", _slugify_location_name(str(location_name)))
    artifacts_dir = Path(str(merged.get("artifacts_dir")))
    merged.setdefault("download_manifest", str(artifacts_dir / "dataset_manifest_download.json"))
    merged.setdefault("output_json", str(artifacts_dir / "raw_export_manifest.json"))
    # Resolve the AOI (same center+area as the rest of the pipeline) so world_window
    # can clip godło sheets that over-cover beyond it. Optional: skip if unresolvable.
    if "aoi_bbox" not in merged:
        try:
            resolved = _resolve_json_center_bbox(dict(merged), required=False)
            if resolved.get("bbox"):
                merged["aoi_bbox"] = resolved["bbox"]
        except (typer.BadParameter, ValueError, KeyError):
            pass
    # base.json carries many keys for other stages; keep only RawExportConfig fields.
    allowed = set(RawExportConfig.model_fields)
    cleaned = {k: v for k, v in merged.items() if k in allowed}
    return RawExportConfig.model_validate(cleaned)


def _build_dem_config_from_base_and_location(*, base_json: Path, location_json: Path) -> DemConfig:
    base_payload = _load_params_json_dict(base_json)
    location_payload = _load_params_json_dict(location_json)
    merged: dict[str, object] = dict(base_payload)
    merged.update(location_payload)
    repo_root = base_json.resolve().parents[2] if len(base_json.resolve().parents) >= 3 else Path.cwd().resolve()
    merged = _apply_location_paths_policy(merged, repo_root)
    merged = _resolve_json_center_bbox(merged, required=True)
    dem_root = Path(str(merged.get("dem_root", "dem")))
    merged.setdefault("output_json", str(dem_root / "dem_manifest.json"))
    artifacts_dir = merged.get("artifacts_dir")
    if artifacts_dir is not None and merged.get("align_to_render", True):
        merged.setdefault("render_manifest", str(Path(str(artifacts_dir)) / "dataset_manifest_render.json"))
    return DemConfig.model_validate(merged)


def _build_dem_availability_config_from_base_and_location(*, base_json: Path, location_json: Path) -> DemAvailabilityConfig:
    base_payload = _load_params_json_dict(base_json)
    location_payload = _load_params_json_dict(location_json)
    merged: dict[str, object] = dict(base_payload)
    merged.update(location_payload)
    # Availability is a discovery tool: report ALL advertised years by default.
    # base.json's year_start/year_end scope the download/run pipeline, not discovery,
    # so ignore them here unless the LOCATION file sets an explicit range.
    if "year_start" not in location_payload:
        merged.pop("year_start", None)
    if "year_end" not in location_payload:
        merged.pop("year_end", None)
    repo_root = base_json.resolve().parents[2] if len(base_json.resolve().parents) >= 3 else Path.cwd().resolve()
    merged = _apply_location_paths_policy(merged, repo_root)
    merged = _resolve_json_center_bbox(merged, required=True)
    artifacts_dir = merged.get("artifacts_dir")
    if artifacts_dir is not None:
        merged.setdefault("output_json", str(Path(str(artifacts_dir)) / "dem_availability.json"))
    return DemAvailabilityConfig.model_validate(merged)


def _build_osm_config_from_base_and_location(*, base_json: Path, location_json: Path) -> OsmConfig:
    base_payload = _load_params_json_dict(base_json)
    location_payload = _load_params_json_dict(location_json)
    merged: dict[str, object] = dict(base_payload)
    merged.update(location_payload)
    repo_root = base_json.resolve().parents[2] if len(base_json.resolve().parents) >= 3 else Path.cwd().resolve()
    merged = _apply_location_paths_policy(merged, repo_root)
    merged = _resolve_json_center_bbox(merged, required=True)
    osm_root = Path(str(merged.get("osm_root", "osm")))
    merged.setdefault("output_json", str(osm_root / "osm_manifest.json"))
    artifacts_dir = merged.get("artifacts_dir")
    if artifacts_dir is not None:
        merged.setdefault("render_manifest", str(Path(str(artifacts_dir)) / "dataset_manifest_render.json"))
    return OsmConfig.model_validate(merged)

