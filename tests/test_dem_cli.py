import json
from pathlib import Path

from typer.testing import CliRunner

from satmap_dataset.cli import (
    app,
    _apply_location_paths_policy,
    _build_dem_config_from_base_and_location,
)

runner = CliRunner()


def test_apply_location_paths_policy_adds_dem_root(tmp_path):
    out = _apply_location_paths_policy({"location_name": "Poznań"}, tmp_path)
    assert out["dem_root"].endswith("dem_poznan")


def test_build_dem_config_from_base_and_location(tmp_path):
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"vertical_datum": "kron86", "max_request_px": 1024}))
    loc = tmp_path / "location_poznan.json"
    loc.write_text(json.dumps({
        "location_name": "Poznań", "center_lat": 52.4, "center_lon": 16.9, "square_km": 4.0,
    }))
    cfg = _build_dem_config_from_base_and_location(base_json=base, location_json=loc)
    assert cfg.vertical_datum == "kron86"
    assert cfg.max_request_px == 1024
    assert str(cfg.dem_root).endswith("dem_poznan")
    assert str(cfg.output_json).endswith("dem_manifest.json")
    assert cfg.bbox  # center resolved to a concrete bbox
    # align_to_render defaults True -> render_manifest auto-points at the render manifest
    assert cfg.render_manifest is not None
    assert str(cfg.render_manifest).endswith("dataset_manifest_render.json")
    assert "artifacts_poznan" in str(cfg.render_manifest)


def test_dem_json_command_invokes_pipeline(tmp_path, monkeypatch):
    captured = {}

    def _fake_run(config):
        captured["config"] = config
        out = tmp_path / "dem_manifest.json"
        out.write_text("{}")
        return (0, out)

    monkeypatch.setattr("satmap_dataset.pipeline.dem.run", _fake_run)
    params = tmp_path / "params.json"
    params.write_text(json.dumps({
        "bbox": "210300,521900,210500,522100", "products": ["nmt"], "align_to_render": False,
        "dem_root": str(tmp_path / "dem_x"), "output_json": str(tmp_path / "dem_manifest.json"),
    }))
    result = runner.invoke(app, ["dem-json", str(params)])
    assert result.exit_code == 0
    assert captured["config"].products == ["nmt"]
    assert result.stdout.strip().splitlines()[-1].endswith("dem_manifest.json")


def test_dem_json_command_bad_config_exit_2(tmp_path):
    params = tmp_path / "params.json"
    params.write_text(json.dumps({"bbox": "10,10,0,0"}))  # invalid order
    result = runner.invoke(app, ["dem-json", str(params)])
    assert result.exit_code == 2
