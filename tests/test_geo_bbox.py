from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.geo import bbox as geo_bbox


def test_project_bbox_parses_xy_order() -> None:
    parsed = geo_bbox.parse("359700,504900,361700,506900")
    assert parsed.as_tuple() == (359700.0, 504900.0, 361700.0, 506900.0)


def test_wfs_query_bbox_swaps_epsg2180_to_authority_order() -> None:
    query = geo_bbox.wfs_query_bbox_str("359700,504900,361700,506900", "EPSG:2180")
    assert query == geo_bbox.swap_axes_str("359700,504900,361700,506900")


def test_wfs_query_bbox_leaves_non_2180_unchanged() -> None:
    project = "500000,6500000,502000,6502000"
    assert geo_bbox.wfs_query_bbox_str(project, "EPSG:3006") == project


def test_tile_bboxes_look_swapped_vs_project() -> None:
    project = (348760.243, 508296.603, 350174.457, 509710.817)
    swapped_sample = (507954.8, 347030.43, 510336.71, 349225.86)
    assert geo_bbox.tile_bboxes_look_swapped_vs_project([swapped_sample], project) is True
    normal_sample = (348800.0, 508300.0, 350100.0, 509700.0)
    assert geo_bbox.tile_bboxes_look_swapped_vs_project([normal_sample], project) is False


@pytest.mark.parametrize(
    ("project_bbox", "srs", "expected_swapped"),
    [
        ("359700,504900,361700,506900", "EPSG:2180", True),
        ("500000,6500000,502000,6502000", "EPSG:3006", False),
    ],
)
def test_wfs_query_axes_swapped(project_bbox: str, srs: str, expected_swapped: bool) -> None:
    assert geo_bbox.wfs_query_axes_swapped(srs) is expected_swapped
    if expected_swapped:
        assert geo_bbox.wfs_query_bbox_str(project_bbox, srs) != project_bbox
    else:
        assert geo_bbox.wfs_query_bbox_str(project_bbox, srs) == project_bbox
