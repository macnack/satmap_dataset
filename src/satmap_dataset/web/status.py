"""Year timeline + pipeline DAG status derivation for satmap-web.

Vendored pure helpers aligned with studio PR #16 (`studio/timeline.py`,
`studio/pipeline_dag.py`) so this branch stays merge-friendly until those
modules land on main. HTML renderers are omitted — the React UI consumes JSON.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

YearLevel = Literal["rendered", "downloaded", "available", "missing"]
StageStatus = Literal["pending", "running", "done", "failed", "skipped"]

_YEAR_FROM_ASSET = re.compile(r"(?:^|[/\\])year_(\d{4})\.(?:tif|tiff)$", re.IGNORECASE)


@dataclass(frozen=True)
class YearCell:
    year: int
    level: YearLevel
    requested: bool = True
    available: bool = False
    downloaded: bool = False
    rendered: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StageNode:
    id: str
    label: str
    status: StageStatus
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PipelineDag:
    stages: tuple[StageNode, ...]
    edges: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": [s.to_dict() for s in self.stages],
            "edges": [{"from": a, "to": b} for a, b in self.edges],
        }


def _as_int_set(values: Iterable[Any] | None) -> set[int]:
    out: set[int] = set()
    if not values:
        return out
    for value in values:
        try:
            out.add(int(value))
        except (TypeError, ValueError):
            continue
    return out


def _years_from_asset_paths(assets: Sequence[Any] | None) -> set[int]:
    years: set[int] = set()
    if not assets:
        return years
    for asset in assets:
        text = str(asset).replace("\\", "/")
        match = _YEAR_FROM_ASSET.search(text)
        if match:
            years.add(int(match.group(1)))
            continue
        parts = text.split("/")
        for part in parts:
            if part.isdigit() and len(part) == 4:
                years.add(int(part))
                break
    return years


def _available_years_from_index(payload: Mapping[str, Any] | None) -> set[int]:
    if not payload:
        return set()
    available = _as_int_set(payload.get("years_available_wfs")) | _as_int_set(
        payload.get("years_included")
    )
    for status in payload.get("year_statuses") or []:
        if not isinstance(status, Mapping):
            continue
        try:
            year = int(status["year"])
        except (KeyError, TypeError, ValueError):
            continue
        if status.get("status") == "has_features" or int(status.get("feature_count") or 0) > 0:
            available.add(year)
    return available


def _years_from_stage_manifest(payload: Mapping[str, Any] | None) -> set[int]:
    if not payload:
        return set()
    years = _as_int_set(payload.get("years_included"))
    years |= _years_from_asset_paths(payload.get("assets"))
    return years


def _years_present_under_download_root(download_root: Path | None) -> set[int]:
    if download_root is None or not download_root.is_dir():
        return set()
    years: set[int] = set()
    for child in download_root.iterdir():
        if child.is_dir() and child.name.isdigit() and len(child.name) == 4:
            if any(child.iterdir()):
                years.add(int(child.name))
    return years


def _years_present_under_render_root(render_root: Path | None) -> set[int]:
    if render_root is None or not render_root.is_dir():
        return set()
    years: set[int] = set()
    for path in render_root.iterdir():
        if not path.is_file():
            continue
        match = _YEAR_FROM_ASSET.search(path.name)
        if match:
            years.add(int(match.group(1)))
    return years


def load_json_mapping(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def derive_year_timeline(
    *,
    year_start: int,
    year_end: int,
    index_payload: Mapping[str, Any] | None = None,
    download_payload: Mapping[str, Any] | None = None,
    render_payload: Mapping[str, Any] | None = None,
    download_root: Path | None = None,
    render_root: Path | None = None,
) -> list[YearCell]:
    if year_end < year_start:
        year_start, year_end = year_end, year_start

    available = _available_years_from_index(index_payload)
    downloaded = _years_from_stage_manifest(download_payload) | _years_present_under_download_root(
        download_root
    )
    rendered = _years_from_stage_manifest(render_payload) | _years_present_under_render_root(
        render_root
    )

    cells: list[YearCell] = []
    for year in range(year_start, year_end + 1):
        is_available = year in available
        is_downloaded = year in downloaded
        is_rendered = year in rendered
        if is_rendered:
            level: YearLevel = "rendered"
        elif is_downloaded:
            level = "downloaded"
        elif is_available:
            level = "available"
        else:
            level = "missing"
        cells.append(
            YearCell(
                year=year,
                level=level,
                requested=True,
                available=is_available,
                downloaded=is_downloaded,
                rendered=is_rendered,
            )
        )
    return cells


def _status_from_manifest(path: Path | None) -> tuple[StageStatus, str | None]:
    if path is None or not path.is_file():
        return "pending", None
    payload = load_json_mapping(path)
    if payload is None:
        return "failed", "unreadable manifest"
    passed = payload.get("passed")
    errors = payload.get("errors")
    if isinstance(errors, list) and errors and passed is not True:
        return "failed", f"{len(errors)} error(s)"
    if isinstance(errors, dict) and errors and passed is not True:
        return "failed", f"{len(errors)} error(s)"
    if passed is False:
        return "failed", "passed=false"
    if passed is True:
        return "done", None
    return "done", None


def _index_status(artifacts_dir: Path) -> tuple[StageStatus, str | None]:
    index_path = artifacts_dir / "index_manifest.json"
    avail_path = artifacts_dir / "year_availability_report.json"
    if index_path.is_file():
        return _status_from_manifest(index_path)
    if avail_path.is_file():
        return _status_from_manifest(avail_path)
    return "pending", None


_RUNNING_HINTS: list[tuple[str, re.Pattern[str]]] = [
    ("raw_export", re.compile(r"raw[\s_-]?export", re.I)),
    ("validate", re.compile(r"validat", re.I)),
    ("dem", re.compile(r"\bdem\b", re.I)),
    ("osm", re.compile(r"\bosm\b", re.I)),
    ("render", re.compile(r"render|mosaic", re.I)),
    ("download", re.compile(r"download", re.I)),
    ("index", re.compile(r"\bindex\b|availab|orthophoto", re.I)),
    ("rgb", re.compile(r"\brgb\b", re.I)),
]


def _first_pending(stage_statuses: Mapping[str, StageStatus], order: Sequence[str]) -> str | None:
    for stage_id in order:
        if stage_statuses.get(stage_id) == "pending":
            return stage_id
    return None


def infer_running_stage(
    *,
    job_name: str | None,
    job_running: bool,
    progress_label: str | None,
    stage_statuses: Mapping[str, StageStatus],
) -> str | None:
    if not job_running:
        return None

    label = progress_label or ""
    if job_name == "index":
        return "index"
    if job_name == "dem_availability":
        return "dem"

    if re.search(r"\brgb\b", label, re.I):
        return _first_pending(stage_statuses, ("index", "download", "render")) or "render"

    for stage_id, pattern in _RUNNING_HINTS:
        if stage_id in {"rgb", "index"}:
            continue
        if pattern.search(label):
            return stage_id

    if job_name in {"run", "location_run"}:
        return _first_pending(
            stage_statuses,
            ("index", "download", "render", "dem", "osm", "validate", "raw_export"),
        )
    if re.search(r"\bindex\b|availab|orthophoto", label, re.I):
        return "index"
    return None


def derive_pipeline_dag(
    *,
    artifacts_dir: Path,
    dem_manifest: Path | None = None,
    osm_manifest: Path | None = None,
    run_dem: bool = False,
    run_osm: bool = False,
    validate: bool = True,
    raw_export: bool = False,
    job_name: str | None = None,
    job_running: bool = False,
    progress_label: str | None = None,
) -> PipelineDag:
    artifacts_dir = Path(artifacts_dir)

    dem_path = Path(dem_manifest) if dem_manifest is not None else None
    if dem_path is not None and dem_path.is_dir():
        dem_path = dem_path / "dem_manifest.json"
    osm_path = Path(osm_manifest) if osm_manifest is not None else None
    if osm_path is not None and osm_path.is_dir():
        osm_path = osm_path / "osm_manifest.json"

    raw_status: dict[str, tuple[StageStatus, str | None]] = {
        "index": _index_status(artifacts_dir),
        "download": _status_from_manifest(artifacts_dir / "dataset_manifest_download.json"),
        "render": _status_from_manifest(artifacts_dir / "dataset_manifest_render.json"),
    }

    if validate:
        raw_status["validate"] = _status_from_manifest(artifacts_dir / "validation_report.json")
    else:
        raw_status["validate"] = ("skipped", "disabled")

    if run_dem:
        raw_status["dem"] = _status_from_manifest(dem_path) if dem_path else ("pending", None)
    else:
        raw_status["dem"] = ("skipped", "disabled")

    if run_osm:
        raw_status["osm"] = _status_from_manifest(osm_path) if osm_path else ("pending", None)
    else:
        raw_status["osm"] = ("skipped", "disabled")

    if raw_export:
        raw_status["raw_export"] = _status_from_manifest(artifacts_dir / "raw_export_manifest.json")
    else:
        raw_status["raw_export"] = ("skipped", "disabled")

    base_statuses = {k: v[0] for k, v in raw_status.items()}
    running_id = infer_running_stage(
        job_name=job_name,
        job_running=job_running,
        progress_label=progress_label,
        stage_statuses=base_statuses,
    )
    if running_id and running_id in raw_status and raw_status[running_id][0] in {
        "pending",
        "running",
    }:
        detail = progress_label or raw_status[running_id][1]
        raw_status[running_id] = ("running", detail)

    labels = {
        "index": "index",
        "download": "download",
        "render": "render",
        "validate": "validate",
        "dem": "dem",
        "osm": "osm",
        "raw_export": "raw-export",
    }
    order = ["index", "download", "render", "validate", "dem", "osm", "raw_export"]
    stages = tuple(
        StageNode(id=sid, label=labels[sid], status=raw_status[sid][0], detail=raw_status[sid][1])
        for sid in order
    )
    edges: list[tuple[str, str]] = [
        ("index", "download"),
        ("download", "render"),
        ("render", "validate"),
        ("render", "dem"),
        ("render", "osm"),
        ("download", "raw_export"),
    ]
    return PipelineDag(stages=stages, edges=tuple(edges))


def location_artifact_paths(repo_root: Path, slug: str) -> dict[str, Path]:
    artifacts = repo_root / f"artifacts_{slug}"
    return {
        "artifacts_dir": artifacts,
        "download_root": repo_root / f"downloads_{slug}",
        "render_root": repo_root / f"rendered_{slug}",
        "dem_manifest": repo_root / f"dem_{slug}" / "dem_manifest.json",
        "osm_manifest": repo_root / f"osm_{slug}" / "osm_manifest.json",
        "index_manifest": artifacts / "index_manifest.json",
        "year_availability": artifacts / "year_availability_report.json",
        "download_manifest": artifacts / "dataset_manifest_download.json",
        "render_manifest": artifacts / "dataset_manifest_render.json",
        "validation_report": artifacts / "validation_report.json",
        "raw_export_manifest": artifacts / "raw_export_manifest.json",
        "pipeline_manifest": artifacts / "pipeline_manifest.json",
    }
