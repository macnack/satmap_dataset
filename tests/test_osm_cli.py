import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from satmap_dataset.cli import (
    app,
    _apply_location_paths_policy,
    _build_osm_config_from_base_and_location,
)
from satmap_dataset.config import OsmConfig

runner = CliRunner()


def test_osm_config_defaults():
    cfg = OsmConfig(bbox="210300,521900,210500,522100")
    assert cfg.categories == ["buildings", "highways", "landuse", "water"]
    assert cfg.srs == "EPSG:2180"
    assert cfg.retries == 3
    assert cfg.sleep_min == 1.0
    assert cfg.sleep_max == 3.0
    assert cfg.overwrite is False


def test_osm_config_invalid_category():
    with pytest.raises(ValidationError):
        OsmConfig(bbox="0,0,10,10", categories=["buildings", "spaceships"])


def test_osm_config_empty_categories():
    with pytest.raises(ValidationError):
        OsmConfig(bbox="0,0,10,10", categories=[])


def test_osm_config_bbox_order_validated():
    with pytest.raises(ValidationError):
        OsmConfig(bbox="10,10,0,0")


def test_osm_config_sleep_order():
    with pytest.raises(ValidationError):
        OsmConfig(bbox="0,0,10,10", sleep_min=3.0, sleep_max=1.0)


def test_osm_config_target_dims_paired():
    with pytest.raises(ValidationError):
        OsmConfig(bbox="0,0,10,10", target_width=100)  # height missing
    ok = OsmConfig(bbox="0,0,10,10", target_width=100, target_height=200)
    assert ok.target_width == 100 and ok.target_height == 200


def test_osm_config_year_date_map():
    cfg = OsmConfig(bbox="0,0,10,10", year_date_map={2022: "2022-04-29", 2023: "2023-05-21"})
    assert cfg.year_date_map[2022] == "2022-04-29"


def test_osm_config_categories_normalized():
    cfg = OsmConfig(bbox="0,0,10,10", categories=["BUILDINGS", "Highways"])
    assert cfg.categories == ["buildings", "highways"]


def test_apply_location_paths_policy_adds_osm_root(tmp_path):
    out = _apply_location_paths_policy({"location_name": "Poznań"}, tmp_path)
    assert out["osm_root"].endswith("osm_poznan")


def test_build_osm_config_from_base_and_location(tmp_path):
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"categories": ["buildings", "highways"]}))
    loc = tmp_path / "loc_poznan.json"
    loc.write_text(json.dumps({
        "location_name": "Poznań",
        "center_lat": 52.4, "center_lon": 16.9, "square_km": 1.0,
    }))
    cfg = _build_osm_config_from_base_and_location(base_json=base, location_json=loc)
    assert cfg.categories == ["buildings", "highways"]
    assert str(cfg.osm_root).endswith("osm_poznan")
    assert cfg.bbox


def test_osm_json_command_invokes_pipeline(tmp_path, monkeypatch):
    captured = {}

    def _fake_run(config):
        captured["config"] = config
        out = tmp_path / "osm_manifest.json"
        out.write_text("{}")
        return (0, out)

    monkeypatch.setattr("satmap_dataset.pipeline.osm.run", _fake_run)
    params = tmp_path / "params.json"
    params.write_text(json.dumps({
        "bbox": "210300,521900,210500,522100",
        "year_date_map": {"2022": "2022-04-29"},
        "categories": ["buildings"],
        "target_width": 100, "target_height": 100,
        "osm_root": str(tmp_path / "osm_x"),
        "output_json": str(tmp_path / "osm_manifest.json"),
    }))
    result = runner.invoke(app, ["osm-json", str(params)])
    assert result.exit_code == 0
    assert captured["config"].categories == ["buildings"]
    assert result.stdout.strip().splitlines()[-1].endswith("osm_manifest.json")


def test_osm_json_command_bad_config_exit_2(tmp_path):
    params = tmp_path / "params.json"
    params.write_text(json.dumps({"bbox": "10,10,0,0"}))
    result = runner.invoke(app, ["osm-json", str(params)])
    assert result.exit_code == 2


def test_manage_roots_knows_osm_kind(tmp_path):
    import scripts.manage_location_roots as mlr

    assert "osm" in mlr.KINDS
    payload = {"location_name": "Poznań"}
    path = mlr._path_for_kind(payload, "osm", tmp_path)
    assert str(path).endswith("osm_poznan")
