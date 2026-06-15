import pytest

from satmap_dataset.config import TrajectoryConfig


def test_defaults():
    c = TrajectoryConfig(track_path="gps_001", output_dir="out")
    assert c.cell_km == 1.0
    assert c.year_start == 2020 and c.year_end == 2025
    assert c.srs == "EPSG:2180"
    assert c.download is False and c.preview is True
    assert c.mode == "hybrid" and c.profile == "train"


def test_year_order_validated():
    with pytest.raises(ValueError):
        TrajectoryConfig(track_path="t", output_dir="o", year_start=2025, year_end=2020)


def test_cell_km_positive():
    with pytest.raises(ValueError):
        TrajectoryConfig(track_path="t", output_dir="o", cell_km=0.0)


def test_sleep_range_validated():
    with pytest.raises(ValueError):
        TrajectoryConfig(track_path="t", output_dir="o", sleep_min=2.0, sleep_max=1.0)
