"""Unit tests for studio pipeline DAG status derivation."""

from __future__ import annotations

import json
from pathlib import Path

from satmap_dataset.studio.pipeline_dag import (
    dag_html,
    derive_pipeline_dag,
    infer_running_stage,
    location_artifact_paths,
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_derive_pipeline_dag_from_manifests(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts_demo"
    dem = tmp_path / "dem_demo" / "dem_manifest.json"
    osm = tmp_path / "osm_demo" / "osm_manifest.json"
    _write(artifacts / "index_manifest.json", {"passed": True, "years_included": [2020]})
    _write(
        artifacts / "dataset_manifest_download.json",
        {"kind": "layer_manifest", "passed": True, "years_included": [2020]},
    )
    _write(
        artifacts / "dataset_manifest_render.json",
        {"kind": "layer_manifest", "passed": True, "years_included": [2020]},
    )
    _write(artifacts / "validation_report.json", {"passed": False, "errors": ["bad"]})
    _write(dem, {"passed": True})
    # osm missing → pending

    dag = derive_pipeline_dag(
        artifacts_dir=artifacts,
        dem_manifest=dem,
        osm_manifest=osm,
        run_dem=True,
        run_osm=True,
        validate=True,
        raw_export=False,
    )
    by_id = {s.id: s for s in dag.stages}
    assert by_id["index"].status == "done"
    assert by_id["download"].status == "done"
    assert by_id["render"].status == "done"
    assert by_id["validate"].status == "failed"
    assert by_id["dem"].status == "done"
    assert by_id["osm"].status == "pending"
    assert by_id["raw_export"].status == "skipped"


def test_derive_skips_optional_stages(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts_x"
    artifacts.mkdir()
    dag = derive_pipeline_dag(
        artifacts_dir=artifacts,
        run_dem=False,
        run_osm=False,
        validate=False,
        raw_export=False,
    )
    by_id = {s.id: s for s in dag.stages}
    assert by_id["dem"].status == "skipped"
    assert by_id["osm"].status == "skipped"
    assert by_id["validate"].status == "skipped"
    assert by_id["raw_export"].status == "skipped"
    assert by_id["index"].status == "pending"


def test_infer_running_stage_from_label() -> None:
    statuses = {
        "index": "done",
        "download": "pending",
        "render": "pending",
        "validate": "pending",
        "dem": "pending",
        "osm": "pending",
        "raw_export": "skipped",
    }
    assert (
        infer_running_stage(
            job_name="location_run",
            job_running=True,
            progress_label="Downloading tiles…",
            stage_statuses=statuses,
        )
        == "download"
    )
    assert (
        infer_running_stage(
            job_name="index",
            job_running=True,
            progress_label="whatever",
            stage_statuses=statuses,
        )
        == "index"
    )
    assert (
        infer_running_stage(
            job_name="location_run",
            job_running=True,
            progress_label="RGB layer (index → download → render)…",
            stage_statuses=statuses,
        )
        == "download"
    )


def test_running_overrides_pending(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts_y"
    _write(artifacts / "index_manifest.json", {"passed": True})
    dag = derive_pipeline_dag(
        artifacts_dir=artifacts,
        run_dem=False,
        run_osm=False,
        validate=True,
        raw_export=False,
        job_name="location_run",
        job_running=True,
        progress_label="RGB layer (index → download → render)…",
    )
    by_id = {s.id: s for s in dag.stages}
    assert by_id["index"].status == "done"
    assert by_id["download"].status == "running"


def test_dag_html_and_paths(tmp_path: Path) -> None:
    paths = location_artifact_paths(tmp_path, "poznan")
    assert paths["artifacts_dir"] == tmp_path / "artifacts_poznan"
    assert paths["dem_manifest"] == tmp_path / "dem_poznan" / "dem_manifest.json"
    dag = derive_pipeline_dag(artifacts_dir=paths["artifacts_dir"], raw_export=True)
    html = dag_html(dag)
    assert "index" in html
    assert "raw-export" in html
    assert "pending" in html
