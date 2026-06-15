import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from satmap_dataset.cli import app
from satmap_dataset.config import DemConfig, OsmConfig, RunConfig

runner = CliRunner()


def _write_base_and_location(tmp_path):
    base = tmp_path / "base.json"
    base.write_text(json.dumps({
        "year_start": 2015, "year_end": 2016, "mode": "hybrid",
        "profile": "reference", "srs": "EPSG:2180", "area_km2": 4.0,
    }))
    loc = tmp_path / "location_poznan.json"
    loc.write_text(json.dumps({
        "location_name": "Poznań", "center_lat": 52.4, "center_lon": 16.9,
    }))
    return base, loc


def test_location_run_json_wires_all_three_layers(tmp_path, monkeypatch):
    captured = {}

    def _fake_run_location(**kwargs):
        captured.update(kwargs)
        out = Path(kwargs["artifacts_dir"]) / "rgb_layer_manifest.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{}")
        return (0, out)

    monkeypatch.setattr(
        "satmap_dataset.cli.location_run.run_location", _fake_run_location
    )
    base, loc = _write_base_and_location(tmp_path)

    result = runner.invoke(app, ["location-run-json", str(loc), "--base-json", str(base)])

    assert result.exit_code == 0, result.stdout
    assert isinstance(captured["rgb_config"], RunConfig)
    assert isinstance(captured["dem_config"], DemConfig)
    assert isinstance(captured["osm_config"], OsmConfig)
    assert captured["run_dem"] is True
    assert captured["run_osm"] is True
    assert captured["validate"] is True
    assert captured["artifacts_dir"] == captured["rgb_config"].artifacts_dir
    assert "artifacts_poznan" in str(captured["artifacts_dir"])
    assert result.stdout.strip().splitlines()[-1].endswith("rgb_layer_manifest.json")


def test_location_run_json_no_dem_no_osm(tmp_path, monkeypatch):
    captured = {}

    def _fake_run_location(**kwargs):
        captured.update(kwargs)
        out = Path(kwargs["artifacts_dir"]) / "rgb_layer_manifest.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{}")
        return (0, out)

    monkeypatch.setattr(
        "satmap_dataset.cli.location_run.run_location", _fake_run_location
    )
    base, loc = _write_base_and_location(tmp_path)

    result = runner.invoke(app, [
        "location-run-json", str(loc), "--base-json", str(base),
        "--no-dem", "--no-osm", "--no-validate",
    ])

    assert result.exit_code == 0, result.stdout
    assert captured["dem_config"] is None
    assert captured["osm_config"] is None
    assert captured["run_dem"] is False
    assert captured["run_osm"] is False
    assert captured["validate"] is False


def test_location_run_json_bad_location_exit_2(tmp_path):
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"year_start": 2015, "year_end": 2016}))
    loc = tmp_path / "bad.json"
    loc.write_text(json.dumps({"location_name": "X"}))  # no center -> bbox cannot resolve

    result = runner.invoke(app, ["location-run-json", str(loc), "--base-json", str(base)])
    assert result.exit_code == 2
