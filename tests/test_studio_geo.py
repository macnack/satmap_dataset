"""Tests for studio geo helpers."""

from __future__ import annotations

import pytest

from satmap_dataset.studio.geo import (
    bbox_from_center,
    bbox_corners_wgs84,
    estimate_output_pixels,
    format_bbox,
    square_side_meters,
    utm_epsg_for_lon_lat,
)

POZNAN_LAT = 52.4012627
POZNAN_LON = 16.9517999


def test_utm_epsg_for_poznan():
    epsg = utm_epsg_for_lon_lat(POZNAN_LON, POZNAN_LAT)
    assert epsg == "EPSG:32633"


def test_utm_epsg_southern_hemisphere():
    epsg = utm_epsg_for_lon_lat(18.0, -33.0)
    assert epsg == "EPSG:32734"


def test_bbox_from_center_epsg2180_ordering():
    bbox_str, bbox_tuple = bbox_from_center(POZNAN_LAT, POZNAN_LON, 4.0, "EPSG:2180")
    minx, miny, maxx, maxy = bbox_tuple
    assert minx < maxx
    assert miny < maxy
    assert bbox_str == format_bbox(bbox_tuple)


def test_bbox_from_center_epsg3006():
    # Kisa, Sweden
    bbox_str, bbox_tuple = bbox_from_center(57.985, 15.629, 4.0, "EPSG:3006")
    minx, miny, maxx, maxy = bbox_tuple
    assert minx < maxx and miny < maxy
    assert "EPSG:3006" not in bbox_str  # numeric only
    assert minx > 100000  # SWEREF easting scale


def test_bbox_from_center_epsg3067():
    # Helsinki area
    bbox_str, bbox_tuple = bbox_from_center(60.17, 24.94, 4.0, "EPSG:3067")
    minx, miny, maxx, maxy = bbox_tuple
    assert minx < maxx and miny < maxy
    assert minx > 100000


def test_bbox_corners_wgs84_closed_ring():
    _, bbox_tuple = bbox_from_center(POZNAN_LAT, POZNAN_LON, 4.0, "EPSG:2180")
    corners = bbox_corners_wgs84(bbox_tuple, "EPSG:2180")
    assert len(corners) == 5
    assert corners[0] == corners[-1]
    for lat, lon in corners:
        assert -90 <= lat <= 90
        assert -180 <= lon <= 180


def test_estimate_output_pixels():
    side = square_side_meters(4.0)
    w, h = estimate_output_pixels(side, 15.0)
    assert w == h
    assert w == int(round(side * 15.0))


def test_square_side_meters():
    assert square_side_meters(4.0) == 2000.0
    assert square_side_meters(9.0) == 3000.0


def test_square_side_meters_invalid():
    with pytest.raises(ValueError):
        square_side_meters(0)
