"""API smoke tests for satmap-web."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from satmap_dataset.web.app import create_app
from satmap_dataset.web.status import derive_pipeline_dag, derive_year_timeline


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    locations = tmp_path / "configs" / "run" / "locations"
    locations.mkdir(parents=True)
    (tmp_path / "configs" / "run" / "base.json").write_text(
        json.dumps(
            {
                "year_start": 2018,
                "year_end": 2020,
                "mode": "hybrid",
                "profile": "train",
                "srs": "EPSG:2180",
                "area_km2": 4.0,
                "px_per_meter": 4.0,
                "target_srs": "EPSG:2180",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (locations / "demo_town.json").write_text(
        json.dumps(
            {
                "location_name": "Demo Town",
                "center_lat": 52.4,
                "center_lon": 16.9,
                "provider": "geoportal",
                "year_start": 2018,
                "year_end": 2020,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    artifacts = tmp_path / "artifacts_demo_town"
    artifacts.mkdir()
    (artifacts / "index_manifest.json").write_text(
        json.dumps(
            {
                "kind": "index_manifest",
                "years_included": [2018, 2019],
                "years_available_wfs": [2018, 2019],
                "year_statuses": [
                    {"year": 2018, "status": "has_features", "feature_count": 2},
                    {"year": 2019, "status": "has_features", "feature_count": 1},
                ],
            }
        ),
        encoding="utf-8",
    )
    (artifacts / "dataset_manifest_download.json").write_text(
        json.dumps({"kind": "dataset_manifest", "years_included": [2018], "assets": []}),
        encoding="utf-8",
    )
    downloads = tmp_path / "downloads_demo_town" / "2018"
    downloads.mkdir(parents=True)
    (downloads / "tile.tif").write_bytes(b"fake")
    return tmp_path


@pytest.fixture()
def client(repo: Path) -> TestClient:
    app = create_app(repo_root=repo)
    return TestClient(app)


def test_health(client: TestClient) -> None:
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["name"] == "satmap-web"


def test_list_and_status(client: TestClient) -> None:
    res = client.get("/api/locations")
    assert res.status_code == 200
    locations = res.json()["locations"]
    assert any(item["id"] == "demo_town" for item in locations)

    status = client.get("/api/locations/demo_town/status")
    assert status.status_code == 200
    payload = status.json()
    assert payload["location"]["provider"] == "geoportal"
    assert payload["location"]["year_start"] == 2018
    levels = {cell["year"]: cell["level"] for cell in payload["timeline"]}
    assert levels[2018] == "downloaded"
    assert levels[2019] in {"available", "downloaded"}
    assert levels[2020] == "missing"
    stage_map = {s["id"]: s["status"] for s in payload["dag"]["stages"]}
    assert stage_map["index"] == "done"
    assert stage_map["download"] == "done"
    assert stage_map["render"] == "pending"

    cli = client.get("/api/locations/demo_town/cli?kind=run")
    assert cli.status_code == 200
    assert "run-location-json" in cli.json()["command"]


def test_unknown_location(client: TestClient) -> None:
    assert client.get("/api/locations/nope/status").status_code == 404


def test_derive_year_timeline_unit() -> None:
    cells = derive_year_timeline(
        year_start=2020,
        year_end=2022,
        index_payload={"years_available_wfs": [2020, 2021]},
        download_payload={"years_included": [2020]},
        render_payload={"assets": ["rendered/year_2020.tif"]},
    )
    by_year = {c.year: c for c in cells}
    assert by_year[2020].level == "rendered"
    assert by_year[2021].level == "available"
    assert by_year[2022].level == "missing"


def test_derive_pipeline_dag_unit(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "index_manifest.json").write_text("{}", encoding="utf-8")
    dag = derive_pipeline_dag(artifacts_dir=artifacts, validate=True)
    statuses = {s.id: s.status for s in dag.stages}
    assert statuses["index"] == "done"
    assert statuses["download"] == "pending"
    assert statuses["validate"] == "pending"
