import math

import pytest

from satmap_dataset.trajectory import (
    Cell,
    TrackPoint,
    _cells_from_projected,
    select_cells,
)


def test_single_point_one_cell():
    assert _cells_from_projected([(500.0, 500.0)], cell_m=1000.0, origin=(0.0, 0.0)) == [(0, 0)]


def test_segment_crosses_multiple_cells():
    # densified line from x=500 to x=2500 at y=500 spans cells 0,1,2 in x
    cells = _cells_from_projected(
        [(500.0, 500.0), (2500.0, 500.0)], cell_m=1000.0, origin=(0.0, 0.0)
    )
    assert cells == [(0, 0), (1, 0), (2, 0)]


def test_dedup_and_sorted():
    cells = _cells_from_projected(
        [(100.0, 100.0), (200.0, 200.0), (100.0, 100.0)],
        cell_m=1000.0,
        origin=(0.0, 0.0),
    )
    assert cells == [(0, 0)]


def test_origin_offset_and_negative_index():
    cells = _cells_from_projected([(-1.0, -1.0)], cell_m=1000.0, origin=(0.0, 0.0))
    assert cells == [(-1, -1)]


def test_invalid_cell_size():
    with pytest.raises(ValueError):
        _cells_from_projected([(0.0, 0.0)], cell_m=0.0, origin=(0.0, 0.0))


def test_select_cells_builds_aligned_bbox():
    # Two points near Kepno, PL -> at least one cell, bbox aligned to 1000 m grid.
    pts = [TrackPoint(51.70227, 17.83960), TrackPoint(51.70250, 17.84050)]
    cells = select_cells(pts, cell_m=1000.0, srs="EPSG:2180", name_stem="t")
    assert len(cells) >= 1
    c = cells[0]
    assert isinstance(c, Cell)
    xmin, ymin, xmax, ymax = c.bbox_2180
    assert math.isclose(xmax - xmin, 1000.0)
    assert math.isclose(ymax - ymin, 1000.0)
    assert math.isclose(xmin % 1000.0, 0.0, abs_tol=1e-6)
    assert c.name == f"t_x{c.ix}_y{c.iy}"
    # wgs84 bbox brackets the cell; center lat/lon inside the wgs84 bbox
    wlon0, wlat0, wlon1, wlat1 = c.bbox_wgs84
    assert wlon0 < c.center_lon < wlon1
    assert wlat0 < c.center_lat < wlat1
