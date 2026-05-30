import pytest
from pydantic import ValidationError

from satmap_dataset.config import DemConfig


def test_defaults_are_both_products_evrf2007():
    cfg = DemConfig(bbox="210300,521900,210500,522100")
    assert cfg.products == ["nmt", "nmpt"]
    assert cfg.vertical_datum == "evrf2007"
    assert cfg.srs == "EPSG:2180"
    assert cfg.align_to_render is True
    assert cfg.max_request_px == 2048


def test_products_normalized_and_validated():
    cfg = DemConfig(bbox="0,0,10,10", products=["NMT"])
    assert cfg.products == ["nmt"]
    with pytest.raises(ValidationError):
        DemConfig(bbox="0,0,10,10", products=["foo"])
    with pytest.raises(ValidationError):
        DemConfig(bbox="0,0,10,10", products=[])


def test_vertical_datum_enum():
    assert DemConfig(bbox="0,0,10,10", vertical_datum="kron86").vertical_datum == "kron86"
    with pytest.raises(ValidationError):
        DemConfig(bbox="0,0,10,10", vertical_datum="nonsense")


def test_bbox_order_validated():
    with pytest.raises(ValidationError):
        DemConfig(bbox="10,10,0,0")


def test_sleep_and_paired_target_dims():
    with pytest.raises(ValidationError):
        DemConfig(bbox="0,0,10,10", sleep_min=2.0, sleep_max=1.0)
    with pytest.raises(ValidationError):
        DemConfig(bbox="0,0,10,10", target_width=100)  # height missing
    ok = DemConfig(bbox="0,0,10,10", target_width=100, target_height=200)
    assert ok.target_width == 100 and ok.target_height == 200
