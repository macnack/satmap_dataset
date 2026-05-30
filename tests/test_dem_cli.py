import json
import sys
from pathlib import Path

from typer.testing import CliRunner

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

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


def test_manage_roots_knows_dem_kind(tmp_path):
    import scripts.manage_location_roots as mlr

    assert "dem" in mlr.KINDS
    payload = {"location_name": "Poznań"}
    path = mlr._path_for_kind(payload, "dem", tmp_path)
    assert str(path).endswith("dem_poznan")


def test_dem_json_skorowidz_dispatches(tmp_path, monkeypatch):
    captured = {}

    def _fake_run(config):
        captured["config"] = config
        out = tmp_path / "m.json"
        out.write_text("{}")
        return (0, out)

    monkeypatch.setattr("satmap_dataset.pipeline.dem_skorowidz.run", _fake_run)
    params = tmp_path / "p.json"
    params.write_text(json.dumps({
        "bbox": "210300,521900,210500,522100", "transport": "skorowidz",
        "year_start": 2012, "year_end": 2019, "products": ["nmt"], "align_to_render": False,
        "dem_root": str(tmp_path / "dem_x"), "output_json": str(tmp_path / "m.json"),
    }))
    result = runner.invoke(app, ["dem-json", str(params)])
    assert result.exit_code == 0
    assert captured["config"].transport == "skorowidz"
    assert captured["config"].requested_years == list(range(2012, 2020))


def test_dem_flag_transport_year_options(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr("satmap_dataset.pipeline.dem_skorowidz.run",
                        lambda config: (captured.setdefault("c", config), (0, tmp_path / "m.json"))[1])
    result = runner.invoke(app, [
        "dem", "--bbox", "0,0,100,100", "--transport", "skorowidz",
        "--year-start", "2012", "--year-end", "2014", "--products", "nmt", "--no-align",
        "--dem-root", str(tmp_path / "dem_x"), "--output-json", str(tmp_path / "m.json"),
    ])
    assert result.exit_code == 0
    assert captured["c"].transport == "skorowidz"
    assert captured["c"].requested_years == [2012, 2013, 2014]


def test_dem_location_builder_inherits_years_from_base(tmp_path):
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"transport": "skorowidz", "year_start": 2011, "year_end": 2019}))
    loc = tmp_path / "location_x.json"
    loc.write_text(json.dumps({"location_name": "Test", "center_lat": 52.4, "center_lon": 16.9, "square_km": 1.0}))
    cfg = _build_dem_config_from_base_and_location(base_json=base, location_json=loc)
    assert cfg.transport == "skorowidz"
    assert cfg.requested_years == list(range(2011, 2020))
