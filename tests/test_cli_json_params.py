from __future__ import annotations

import json
import sys
from pathlib import Path

from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset import cli
from satmap_dataset.config import DownloadConfig, IndexConfig, RenderConfig, RunConfig


def test_download_json_command_loads_config_and_runs(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, DownloadConfig] = {}

    def fake_run(config: DownloadConfig):
        captured["config"] = config
        return 0, config.output_json

    monkeypatch.setattr(cli.downloader, "run", fake_run)

    params = {
        "index_manifest": str(tmp_path / "index_manifest.json"),
        "download_root": str(tmp_path / "downloads"),
        "mode": "wfs_render",
        "profile": "train",
        "output_json": str(tmp_path / "dataset_manifest_download.json"),
    }
    params_path = tmp_path / "download_params.json"
    params_path.write_text(json.dumps(params), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cli.app, ["download-json", str(params_path)])

    assert result.exit_code == 0
    assert "dataset_manifest_download.json" in result.stdout
    assert "config" in captured
    assert captured["config"].mode == "wfs_render"


def test_download_json_center_mode_converts_to_bbox(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, DownloadConfig] = {}

    def fake_run(config: DownloadConfig):
        captured["config"] = config
        return 0, config.output_json

    monkeypatch.setattr(cli.downloader, "run", fake_run)
    monkeypatch.setattr(cli, "_bbox_from_center_latlon", lambda lat, lon, square_km: "1,2,3,4")

    params = {
        "index_manifest": str(tmp_path / "index_manifest.json"),
        "download_root": str(tmp_path / "downloads"),
        "mode": "hybrid",
        "profile": "reference",
        "center_lat": 52.4012627,
        "center_lon": 16.9517999,
        "square_km": 2.0,
        "srs": "EPSG:2180",
        "output_json": str(tmp_path / "dataset_manifest_download.json"),
    }
    params_path = tmp_path / "download_center_params.json"
    params_path.write_text(json.dumps(params), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cli.app, ["download-json", str(params_path)])

    assert result.exit_code == 0
    assert "config" in captured
    assert captured["config"].bbox == "1,2,3,4"


def test_index_json_supports_area_km2_alias(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, IndexConfig] = {}

    def fake_run(config: IndexConfig):
        captured["config"] = config
        return 0, config.output_json

    monkeypatch.setattr(cli.index_builder, "run", fake_run)
    monkeypatch.setattr(cli, "_bbox_from_center_latlon", lambda lat, lon, square_km: "10,20,30,40")

    params = {
        "year_start": 2015,
        "year_end": 2016,
        "center_lat": 52.4012627,
        "center_lon": 16.9517999,
        "area_km2": 2.0,
        "srs": "EPSG:2180",
        "output_json": str(tmp_path / "index_manifest.json"),
        "year_availability_output_json": str(tmp_path / "year_availability_report.json"),
    }
    params_path = tmp_path / "index_center_params.json"
    params_path.write_text(json.dumps(params), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cli.app, ["index-json", str(params_path)])

    assert result.exit_code == 0
    assert "config" in captured
    assert captured["config"].bbox == "10,20,30,40"


def test_run_location_json_uses_base_and_generates_paths(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, RunConfig] = {}

    def fake_run(config: RunConfig):
        captured["config"] = config
        return 0, config.artifacts_dir / "validation_report.json"

    monkeypatch.setattr(cli.run_all, "run", fake_run)
    monkeypatch.setattr(cli, "_bbox_from_center_latlon", lambda lat, lon, square_km: "100,200,300,400")

    base = {
        "year_start": 2015,
        "year_end": 2016,
        "mode": "hybrid",
        "profile": "reference",
        "srs": "EPSG:2180",
        "area_km2": 4.0,
    }
    location = {
        "location_name": "Żelazny Most",
        "center_lat": 51.514264,
        "center_lon": 16.162344,
    }
    base_path = tmp_path / "configs" / "run" / "base.json"
    location_path = tmp_path / "configs" / "run" / "locations" / "zelazny_most.json"
    base_path.parent.mkdir(parents=True, exist_ok=True)
    location_path.parent.mkdir(parents=True, exist_ok=True)
    base_path.write_text(json.dumps(base), encoding="utf-8")
    location_path.write_text(json.dumps(location), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["run-location-json", str(location_path), "--base-json", str(base_path)],
    )

    assert result.exit_code == 0
    assert "config" in captured
    cfg = captured["config"]
    assert cfg.bbox == "100,200,300,400"
    assert str(cfg.download_root).endswith("downloads_zelazny_most")
    assert str(cfg.render_root).endswith("rendered_zelazny_most")
    assert str(cfg.artifacts_dir).endswith("artifacts_zelazny_most")


def test_render_location_json_uses_base_and_generates_paths(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, RenderConfig] = {}

    def fake_run(config: RenderConfig):
        captured["config"] = config
        return 0, config.output_json

    monkeypatch.setattr(cli.render, "run", fake_run)

    base = {
        "mode": "hybrid",
        "profile": "reference",
        "srs": "EPSG:2180",
        "compression": "jpeg95",
    }
    location = {
        "location_name": "Żelazny Most",
        "center_lat": 51.514264,
        "center_lon": 16.162344,
    }
    base_path = tmp_path / "configs" / "run" / "base.json"
    location_path = tmp_path / "configs" / "run" / "locations" / "zelazny_most.json"
    base_path.parent.mkdir(parents=True, exist_ok=True)
    location_path.parent.mkdir(parents=True, exist_ok=True)
    base_path.write_text(json.dumps(base), encoding="utf-8")
    location_path.write_text(json.dumps(location), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["render-location-json", str(location_path), "--base-json", str(base_path)],
    )

    assert result.exit_code == 0
    assert "config" in captured
    cfg = captured["config"]
    assert cfg.compression == "jpeg95"
    assert str(cfg.render_root).endswith("rendered_zelazny_most")
    assert str(cfg.dataset_manifest).endswith("artifacts_zelazny_most/dataset_manifest_download.json")
    assert str(cfg.output_json).endswith("artifacts_zelazny_most/dataset_manifest_render.json")


def test_run_all_location_json_runs_each_file(monkeypatch, tmp_path: Path) -> None:
    captured: list[RunConfig] = []

    def fake_run(config: RunConfig):
        captured.append(config)
        return 0, config.artifacts_dir / "validation_report.json"

    monkeypatch.setattr(cli.run_all, "run", fake_run)
    monkeypatch.setattr(cli, "_bbox_from_center_latlon", lambda lat, lon, square_km: "100,200,300,400")

    base = {
        "year_start": 2015,
        "year_end": 2016,
        "mode": "hybrid",
        "profile": "reference",
        "srs": "EPSG:2180",
        "area_km2": 4.0,
    }
    locations_dir = tmp_path / "configs" / "run" / "locations"
    base_path = tmp_path / "configs" / "run" / "base.json"
    base_path.parent.mkdir(parents=True, exist_ok=True)
    locations_dir.mkdir(parents=True, exist_ok=True)
    base_path.write_text(json.dumps(base), encoding="utf-8")
    (locations_dir / "poznan.json").write_text(
        json.dumps(
            {
                "location_name": "Poznan",
                "center_lat": 52.4012627,
                "center_lon": 16.9517999,
            }
        ),
        encoding="utf-8",
    )
    (locations_dir / "zelazny_most.json").write_text(
        json.dumps(
            {
                "location_name": "Żelazny Most",
                "center_lat": 51.514264,
                "center_lon": 16.162344,
            }
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "run-all-location-json",
            "--locations-dir",
            str(locations_dir),
            "--base-json",
            str(base_path),
        ],
    )

    assert result.exit_code == 0
    assert len(captured) == 2


def test_index_all_location_json_runs_each_file(monkeypatch, tmp_path: Path) -> None:
    captured: list[object] = []

    def fake_run(config):
        captured.append(config)
        return 0, Path(config.output_json)

    monkeypatch.setattr(cli.index_builder, "run", fake_run)
    monkeypatch.setattr(cli, "_bbox_from_center_latlon", lambda lat, lon, square_km: "100,200,300,400")

    base = {
        "year_start": 2015,
        "year_end": 2016,
        "srs": "EPSG:2180",
        "area_km2": 4.0,
    }
    locations_dir = tmp_path / "configs" / "run" / "locations"
    base_path = tmp_path / "configs" / "run" / "base.json"
    base_path.parent.mkdir(parents=True, exist_ok=True)
    locations_dir.mkdir(parents=True, exist_ok=True)
    base_path.write_text(json.dumps(base), encoding="utf-8")
    (locations_dir / "a.json").write_text(
        json.dumps({"location_name": "A", "center_lat": 52.4, "center_lon": 16.9}), encoding="utf-8"
    )
    (locations_dir / "b.json").write_text(
        json.dumps({"location_name": "B", "center_lat": 51.5, "center_lon": 16.1}), encoding="utf-8"
    )

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["index-all-location-json", "--locations-dir", str(locations_dir), "--base-json", str(base_path)],
    )

    assert result.exit_code == 0
    assert len(captured) == 2


def test_download_all_location_json_runs_each_file(monkeypatch, tmp_path: Path) -> None:
    captured: list[object] = []

    def fake_run(config):
        captured.append(config)
        return 0, Path(config.output_json)

    monkeypatch.setattr(cli.downloader, "run", fake_run)
    monkeypatch.setattr(cli, "_bbox_from_center_latlon", lambda lat, lon, square_km: "100,200,300,400")

    base = {
        "mode": "hybrid",
        "profile": "reference",
        "srs": "EPSG:2180",
        "area_km2": 4.0,
    }
    locations_dir = tmp_path / "configs" / "run" / "locations"
    base_path = tmp_path / "configs" / "run" / "base.json"
    base_path.parent.mkdir(parents=True, exist_ok=True)
    locations_dir.mkdir(parents=True, exist_ok=True)
    base_path.write_text(json.dumps(base), encoding="utf-8")
    (locations_dir / "a.json").write_text(
        json.dumps({"location_name": "A", "center_lat": 52.4, "center_lon": 16.9}), encoding="utf-8"
    )
    (locations_dir / "b.json").write_text(
        json.dumps({"location_name": "B", "center_lat": 51.5, "center_lon": 16.1}), encoding="utf-8"
    )

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["download-all-location-json", "--locations-dir", str(locations_dir), "--base-json", str(base_path)],
    )

    assert result.exit_code == 0
    assert len(captured) == 2


def test_validate_all_location_json_runs_each_file(monkeypatch, tmp_path: Path) -> None:
    captured: list[object] = []

    def fake_run(config):
        captured.append(config)
        return 0, Path(config.output_json)

    monkeypatch.setattr(cli.validator, "run", fake_run)

    base = {
        "year_start": 2015,
        "year_end": 2016,
        "strict_years": False,
        "min_years": 1,
    }
    locations_dir = tmp_path / "configs" / "run" / "locations"
    base_path = tmp_path / "configs" / "run" / "base.json"
    base_path.parent.mkdir(parents=True, exist_ok=True)
    locations_dir.mkdir(parents=True, exist_ok=True)
    base_path.write_text(json.dumps(base), encoding="utf-8")
    (locations_dir / "a.json").write_text(
        json.dumps({"location_name": "A", "center_lat": 52.4, "center_lon": 16.9}), encoding="utf-8"
    )
    (locations_dir / "b.json").write_text(
        json.dumps({"location_name": "B", "center_lat": 51.5, "center_lon": 16.1}), encoding="utf-8"
    )

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["validate-all-location-json", "--locations-dir", str(locations_dir), "--base-json", str(base_path)],
    )

    assert result.exit_code == 0
    assert len(captured) == 2
