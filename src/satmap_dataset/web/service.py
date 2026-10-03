"""Location discovery, config merge, and status assembly for satmap-web."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from satmap_dataset.cli import _slugify_location_name
from satmap_dataset.studio.config_builders import (
    PROVIDER_PRESETS,
    merge_base_and_location_payload,
    resolve_base_json,
)
from satmap_dataset.web.jobs import Job, JobRegistry, JobStatus
from satmap_dataset.web.status import (
    derive_pipeline_dag,
    derive_year_timeline,
    load_json_mapping,
    location_artifact_paths,
)


def default_repo_root() -> Path:
    # web/service.py → web → satmap_dataset → src → repo
    return Path(__file__).resolve().parents[3]


def locations_dir(repo_root: Path | None = None) -> Path:
    root = repo_root or default_repo_root()
    return root / "configs" / "run" / "locations"


def list_location_files(repo_root: Path | None = None) -> list[Path]:
    directory = locations_dir(repo_root)
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.json"))


def load_location_payload(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Location JSON must be an object: {path}")
    return data


def resolve_location_path(location_id: str, repo_root: Path | None = None) -> Path:
    """Resolve a location id (stem or filename) to a JSON path."""
    root = repo_root or default_repo_root()
    stem = location_id.removesuffix(".json")
    path = locations_dir(root) / f"{stem}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Unknown location: {location_id}")
    return path


def location_summary(path: Path, repo_root: Path | None = None) -> dict[str, Any]:
    root = repo_root or default_repo_root()
    payload = load_location_payload(path)
    provider = str(payload.get("provider") or "geoportal")
    try:
        base_json = resolve_base_json(provider, root)
    except ValueError:
        base_json = root / "configs" / "run" / "base.json"
        provider = "geoportal"

    # Prefer not failing the list view if bbox/proj is unavailable.
    try:
        merged = merge_base_and_location_payload(base_json, payload, resolve_bbox=True)
    except Exception:  # noqa: BLE001
        merged = merge_base_and_location_payload(base_json, payload, resolve_bbox=False)

    # Bbox resolution may drop center/area; keep display fields from pre-resolve merge.
    display = merge_base_and_location_payload(base_json, payload, resolve_bbox=False)

    location_name = str(merged.get("location_name") or path.stem)
    slug = _slugify_location_name(location_name)
    paths = location_artifact_paths(root, slug)

    year_start = int(merged.get("year_start") or 2015)
    year_end = int(merged.get("year_end") or year_start)

    return {
        "id": path.stem,
        "file": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
        "location_name": location_name,
        "slug": slug,
        "provider": str(merged.get("provider") or provider),
        "year_start": year_start,
        "year_end": year_end,
        "center_lat": display.get("center_lat"),
        "center_lon": display.get("center_lon"),
        "area_km2": display.get("area_km2") or display.get("square_km"),
        "srs": merged.get("srs"),
        "profile": merged.get("profile"),
        "mode": merged.get("mode"),
        "bbox": merged.get("bbox"),
        "paths": {k: str(v) for k, v in paths.items()},
        "base_json": str(base_json.relative_to(root)) if base_json.is_relative_to(root) else str(base_json),
        "providers": sorted(PROVIDER_PRESETS.keys()),
    }


def location_status(
    location_id: str,
    *,
    repo_root: Path | None = None,
    jobs: JobRegistry | None = None,
) -> dict[str, Any]:
    root = repo_root or default_repo_root()
    path = resolve_location_path(location_id, root)
    summary = location_summary(path, root)
    paths = location_artifact_paths(root, summary["slug"])

    index_payload = load_json_mapping(paths["index_manifest"]) or load_json_mapping(
        paths["year_availability"]
    )
    download_payload = load_json_mapping(paths["download_manifest"])
    render_payload = load_json_mapping(paths["render_manifest"])

    active: Job | None = jobs.active_for_location(location_id) if jobs else None
    job_running = bool(active and active.state.status == JobStatus.RUNNING)

    timeline = derive_year_timeline(
        year_start=int(summary["year_start"]),
        year_end=int(summary["year_end"]),
        index_payload=index_payload,
        download_payload=download_payload,
        render_payload=render_payload,
        download_root=paths["download_root"],
        render_root=paths["render_root"],
    )
    dag = derive_pipeline_dag(
        artifacts_dir=paths["artifacts_dir"],
        dem_manifest=paths["dem_manifest"],
        osm_manifest=paths["osm_manifest"],
        run_dem=paths["dem_manifest"].is_file() or paths["dem_manifest"].parent.is_dir(),
        run_osm=paths["osm_manifest"].is_file() or paths["osm_manifest"].parent.is_dir(),
        validate=True,
        raw_export=paths["raw_export_manifest"].is_file(),
        job_name=active.name if active else None,
        job_running=job_running,
        progress_label=active.state.progress_label if active else None,
    )

    artifact_flags = {
        key: paths[key].is_file()
        for key in (
            "index_manifest",
            "year_availability",
            "download_manifest",
            "render_manifest",
            "validation_report",
            "pipeline_manifest",
            "raw_export_manifest",
            "dem_manifest",
            "osm_manifest",
        )
    }

    return {
        "location": summary,
        "timeline": [cell.to_dict() for cell in timeline],
        "dag": dag.to_dict(),
        "artifacts": artifact_flags,
        "active_job": active.snapshot() if active else None,
    }


def cli_deeplink(location_id: str, *, command: str = "run", repo_root: Path | None = None) -> dict[str, str]:
    root = repo_root or default_repo_root()
    path = resolve_location_path(location_id, root)
    summary = location_summary(path, root)
    rel = summary["file"]
    base = summary["base_json"]
    if command == "index":
        cmd = f'python -m satmap_dataset.cli index-location-json "{rel}" --base-json "{base}"'
        just = f'just index-location-json location_json={rel}'
    else:
        cmd = f'python -m satmap_dataset.cli run-location-json "{rel}" --base-json "{base}"'
        just = f'just run-location-json location_json={rel}'
    return {"command": cmd, "just": just, "location_id": location_id, "kind": command}


def start_pipeline_job(
    location_id: str,
    *,
    kind: str,
    jobs: JobRegistry,
    repo_root: Path | None = None,
) -> Job:
    root = repo_root or default_repo_root()
    path = resolve_location_path(location_id, root)
    summary = location_summary(path, root)
    payload = load_location_payload(path)
    provider = str(payload.get("provider") or summary["provider"])
    base_json = resolve_base_json(provider, root)

    if jobs.active_for_location(location_id) is not None:
        raise RuntimeError(f"A job is already running for {location_id}")

    if kind == "index":
        from satmap_dataset.pipeline import index_builder
        from satmap_dataset.studio.config_builders import build_index_config

        config = build_index_config(payload, base_json)

        def run_fn() -> tuple[int, Any]:
            return index_builder.run(config)

        job_name = "index"
    elif kind == "run":
        from satmap_dataset.pipeline import run_all
        from satmap_dataset.studio.config_builders import build_run_config

        config = build_run_config(payload, base_json)

        def run_fn() -> tuple[int, Any]:
            return run_all.run(config)

        job_name = "run"
    else:
        raise ValueError(f"Unsupported job kind: {kind!r}")

    job = jobs.create(name=job_name, location_id=location_id)
    job.start(run_fn)
    return job
