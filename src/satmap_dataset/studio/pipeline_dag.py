"""Pipeline DAG helpers for satmap-studio (pure status derivation + HTML)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

StageStatus = Literal["pending", "running", "done", "failed", "skipped"]


@dataclass(frozen=True)
class StageNode:
    id: str
    label: str
    status: StageStatus
    detail: str | None = None


@dataclass(frozen=True)
class PipelineDag:
    stages: tuple[StageNode, ...]
    edges: tuple[tuple[str, str], ...]


def _read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _status_from_manifest(path: Path | None) -> tuple[StageStatus, str | None]:
    """Map a stage artifact to done/failed/pending."""
    if path is None or not path.is_file():
        return "pending", None
    payload = _read_json(path)
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
    # File exists without explicit passed → treat as done (partial/legacy).
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
    # RGB blob covers index→download→render when label is generic
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
    """Best-effort map of an active studio job onto a DAG stage id."""
    if not job_running:
        return None

    label = progress_label or ""
    if job_name == "index":
        return "index"
    if job_name == "dem_availability":
        return "dem"

    # RGB layer progress is a single blob covering index→download→render.
    if re.search(r"\brgb\b", label, re.I):
        return _first_pending(stage_statuses, ("index", "download", "render")) or "render"

    for stage_id, pattern in _RUNNING_HINTS:
        if stage_id in {"rgb", "index"}:
            # Avoid matching the word "index" inside the RGB progress string.
            continue
        if pattern.search(label):
            return stage_id

    if job_name == "location_run":
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
    run_dem: bool = True,
    run_osm: bool = True,
    validate: bool = True,
    raw_export: bool = False,
    job_name: str | None = None,
    job_running: bool = False,
    progress_label: str | None = None,
) -> PipelineDag:
    """Derive stage statuses from artifacts + optional active job."""
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
    # Drop edges into skipped-only optionals? Keep edges; renderer greys skipped nodes.
    return PipelineDag(stages=stages, edges=tuple(edges))


_STATUS_STYLE: dict[StageStatus, tuple[str, str, str]] = {
    # bg, border, text
    "pending": ("#f4f4f4", "#b0b0b0", "#555555"),
    "running": ("#fff4d6", "#d4a017", "#5a4200"),
    "done": ("#e5f6ec", "#1b7f4a", "#145c36"),
    "failed": ("#fde8e8", "#c0392b", "#8b1e14"),
    "skipped": ("#f0f0f0", "#cccccc", "#999999"),
}


def dag_html(dag: PipelineDag, *, title: str | None = "Pipeline") -> str:
    """Simple HTML DAG: core row + optional row (Streamlit ``unsafe_allow_html``)."""
    by_id = {s.id: s for s in dag.stages}

    def _node(stage: StageNode) -> str:
        bg, border, fg = _STATUS_STYLE[stage.status]
        tip = stage.detail or stage.status
        return (
            f'<div title="{stage.label}: {tip}" style="min-width:88px;padding:8px 10px;'
            f'text-align:center;border-radius:6px;background:{bg};border:1.5px solid {border};'
            f'color:{fg};font:600 12px/1.2 system-ui,sans-serif;">'
            f"{stage.label}<div style='font-weight:500;font-size:10px;margin-top:3px;'>"
            f"{stage.status}</div></div>"
        )

    def _arrow() -> str:
        return (
            '<div style="align-self:center;color:#888;font-size:16px;padding:0 4px;" '
            'aria-hidden="true">→</div>'
        )

    core_ids = ["index", "download", "render", "validate"]
    core_parts: list[str] = []
    for i, sid in enumerate(core_ids):
        if i:
            core_parts.append(_arrow())
        core_parts.append(_node(by_id[sid]))

    optional_parts: list[str] = []
    for sid in ("dem", "osm", "raw_export"):
        stage = by_id[sid]
        if stage.status == "skipped" and not optional_parts:
            # Still show skipped optionals so the graph shape is stable.
            pass
        optional_parts.append(_node(stage))

    legend = (
        '<span style="margin-right:10px;">pending</span>'
        '<span style="margin-right:10px;color:#d4a017;">running</span>'
        '<span style="margin-right:10px;color:#1b7f4a;">done</span>'
        '<span style="margin-right:10px;color:#c0392b;">failed</span>'
        '<span style="color:#999;">skipped</span>'
    )
    heading = f"<div style='font-weight:600;margin-bottom:6px;'>{title}</div>" if title else ""
    optional_row = (
        "<div style='display:flex;flex-wrap:wrap;gap:8px;margin-top:10px;align-items:center;'>"
        "<span style='font-size:11px;color:#777;margin-right:4px;'>optional</span>"
        + "".join(optional_parts)
        + "</div>"
    )
    return (
        f"<div style='margin:4px 0 14px 0;'>{heading}"
        f"<div style='font-size:12px;color:#555;margin-bottom:8px;'>{legend}</div>"
        f"<div style='display:flex;flex-wrap:wrap;align-items:stretch;gap:0;'>"
        f"{''.join(core_parts)}</div>"
        f"{optional_row}</div>"
    )


def location_artifact_paths(repo_root: Path, slug: str) -> dict[str, Path]:
    """Conventional on-disk roots for a location slug."""
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
    }
