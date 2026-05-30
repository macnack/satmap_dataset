import pytest
from pydantic import ValidationError

from satmap_dataset.config import OsmConfig


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
