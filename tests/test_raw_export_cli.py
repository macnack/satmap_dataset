import json
from pathlib import Path

from typer.testing import CliRunner

from satmap_dataset import cli
from satmap_dataset.config import RawExportConfig

runner = CliRunner()


def _write_base_and_location(tmp_path):
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"year_start": 2015, "year_end": 2021, "provider": "geoportal"}))
    loc = tmp_path / "location_poznan.json"
    loc.write_text(json.dumps({"location_name": "Poznań", "center_lat": 52.4, "center_lon": 16.9}))
    return base, loc


def test_builder_derives_area_and_shared_raw_root(tmp_path, monkeypatch):
    monkeypatch.setenv("SATMAP_RAW_ROOT", str(tmp_path / "sat_data_raw"))
    base, loc = _write_base_and_location(tmp_path)
    cfg = cli._build_raw_export_config_from_base_and_location(base_json=base, location_json=loc)
    assert isinstance(cfg, RawExportConfig)
    assert cfg.provider == "geoportal"
    assert cfg.area == "poznan"
    assert "downloads_poznan" in str(cfg.download_root)
    assert str(cfg.raw_root) == str(tmp_path / "sat_data_raw")  # shared, not per-location


def test_raw_export_json_invokes_run(tmp_path, monkeypatch):
    captured = {}

    def _fake_run(config):
        captured["config"] = config
        out = Path(config.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{}")
        return 0, out

    monkeypatch.setattr(cli.raw_export, "run", _fake_run)
    params = {
        "provider": "geoportal", "area": "poznan",
        "download_root": str(tmp_path / "downloads_poznan"),
        "raw_root": str(tmp_path / "sat_data_raw"),
        "artifacts_dir": str(tmp_path / "artifacts_poznan"),
        "output_json": str(tmp_path / "artifacts_poznan" / "raw_export_manifest.json"),
    }
    p = tmp_path / "params.json"
    p.write_text(json.dumps(params))
    result = runner.invoke(cli.app, ["raw-export-json", str(p)])
    assert result.exit_code == 0, result.stdout
    assert isinstance(captured["config"], RawExportConfig)
    assert captured["config"].area == "poznan"


def test_raw_export_json_rejects_sentinel2(tmp_path):
    params = {"provider": "sentinel2", "area": "x", "download_root": str(tmp_path)}
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(params))
    result = runner.invoke(cli.app, ["raw-export-json", str(p)])
    assert result.exit_code == 2


def test_raw_test_manifest_invokes_builder(tmp_path, monkeypatch):
    captured = {}

    def _fake_build(root, out, min_years=2):
        captured["root"] = Path(root)
        captured["out"] = Path(out)
        captured["min_years"] = min_years
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text("roots: {}\n")
        return {"roots": {}}

    monkeypatch.setattr(cli, "build_test_manifest", _fake_build)
    result = runner.invoke(
        cli.app,
        ["raw-test-manifest", "--raw-root", str(tmp_path / "sat_data_raw"), "--min-years", "3"],
    )
    assert result.exit_code == 0, result.stdout
    assert captured["min_years"] == 3
    assert captured["out"] == tmp_path / "sat_data_raw" / "test_manifest.yaml"


def test_raw_export_help_smoke():
    for cmd in ("raw-export", "raw-export-json", "raw-export-location-json",
                "raw-export-all-location-json", "raw-test-manifest"):
        result = runner.invoke(cli.app, [cmd, "--help"])
        assert result.exit_code == 0, result.stdout
        assert "Usage" in result.stdout
