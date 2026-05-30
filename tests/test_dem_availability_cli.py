import json
from pathlib import Path

from typer.testing import CliRunner

from satmap_dataset.cli import app, _build_dem_availability_config_from_base_and_location

runner = CliRunner()


def test_dem_availability_json_command(tmp_path, monkeypatch):
    captured = {}

    def _fake_run(config):
        captured["config"] = config
        out = tmp_path / "av.json"
        out.write_text(json.dumps({
            "kind": "dem_availability", "aoi_bbox": config.bbox, "srs": "EPSG:2180",
            "entries": [], "errors": {}, "full_coverage_options": [],
        }))
        return (0, out)

    monkeypatch.setattr("satmap_dataset.pipeline.dem_availability.run", _fake_run)
    params = tmp_path / "p.json"
    params.write_text(json.dumps({
        "bbox": "210300,521900,210500,522100",
        "output_json": str(tmp_path / "av.json"),
    }))
    result = runner.invoke(app, ["dem-availability-json", str(params)])
    assert result.exit_code == 0
    assert captured["config"].products == ["nmt", "nmpt"]
    assert result.stdout.strip().splitlines()[-1].endswith("av.json")


def test_dem_availability_location_builder(tmp_path):
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"products": ["nmpt"]}))
    loc = tmp_path / "location_x.json"
    loc.write_text(json.dumps({"location_name": "Test", "center_lat": 52.4, "center_lon": 16.9, "square_km": 1.0}))
    cfg = _build_dem_availability_config_from_base_and_location(base_json=base, location_json=loc)
    assert cfg.products == ["nmpt"]
    assert cfg.bbox
    assert str(cfg.output_json).endswith("dem_availability.json")
    assert "artifacts_test" in str(cfg.output_json)


def test_dem_availability_json_bad_config_exit_2(tmp_path):
    params = tmp_path / "p.json"
    params.write_text(json.dumps({"bbox": "10,10,0,0"}))
    result = runner.invoke(app, ["dem-availability-json", str(params)])
    assert result.exit_code == 2


def test_dem_availability_builder_ignores_base_year_range(tmp_path):
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"year_start": 2014, "year_end": 2025}))
    loc = tmp_path / "location_x.json"
    loc.write_text(json.dumps({"location_name": "Test", "center_lat": 52.4, "center_lon": 16.9, "square_km": 1.0}))
    cfg = _build_dem_availability_config_from_base_and_location(base_json=base, location_json=loc)
    assert cfg.year_start is None and cfg.year_end is None  # base range ignored for discovery
    assert cfg.requested_years is None


def test_dem_availability_builder_honors_location_year_range(tmp_path):
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"year_start": 2014, "year_end": 2025}))
    loc = tmp_path / "location_x.json"
    loc.write_text(json.dumps({
        "location_name": "Test", "center_lat": 52.4, "center_lon": 16.9, "square_km": 1.0,
        "year_start": 2020, "year_end": 2022,
    }))
    cfg = _build_dem_availability_config_from_base_and_location(base_json=base, location_json=loc)
    assert cfg.year_start == 2020 and cfg.year_end == 2022
    assert cfg.requested_years == [2020, 2021, 2022]
