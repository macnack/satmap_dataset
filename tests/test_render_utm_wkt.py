from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.pipeline import render
from satmap_dataset.pipeline.render import (
    _utm_wkt_for_epsg,
    _wkt_for_epsg,
    _write_worldfile_and_prj,
    BBox,
)


def test_utm_wkt_zone_33_n_central_meridian() -> None:
    wkt = _utm_wkt_for_epsg(32633)
    assert wkt is not None
    assert 'PROJCS["WGS 84 / UTM zone 33N"' in wkt
    assert 'PARAMETER["central_meridian",15]' in wkt
    assert 'PARAMETER["false_northing",0]' in wkt
    assert 'AUTHORITY["EPSG","32633"]' in wkt


def test_utm_wkt_zone_60_n_central_meridian() -> None:
    wkt = _utm_wkt_for_epsg(32660)
    assert 'PARAMETER["central_meridian",177]' in wkt


def test_utm_wkt_southern_hemisphere_false_northing() -> None:
    wkt = _utm_wkt_for_epsg(32733)
    assert "UTM zone 33S" in wkt
    assert 'PARAMETER["false_northing",10000000]' in wkt


def test_utm_wkt_returns_none_outside_range() -> None:
    assert _utm_wkt_for_epsg(4326) is None
    assert _utm_wkt_for_epsg(32600) is None
    assert _utm_wkt_for_epsg(32661) is None


def test_wkt_for_epsg_prefers_static_table_for_known_codes() -> None:
    wkt_3006 = _wkt_for_epsg(3006)
    assert wkt_3006 is not None
    assert "SWEREF99 TM" in wkt_3006
    wkt_2180 = _wkt_for_epsg(2180)
    assert wkt_2180 is not None
    assert "CS92" in wkt_2180
    wkt_3067 = _wkt_for_epsg(3067)
    assert wkt_3067 is not None
    assert "TM35FIN" in wkt_3067
    assert 'AUTHORITY["EPSG","3067"]' in wkt_3067


def test_write_prj_emits_utm_wkt(tmp_path: Path) -> None:
    out = tmp_path / "scene.tiff"
    out.write_bytes(b"x")
    _write_worldfile_and_prj(
        out,
        target_bbox=BBox(min_x=536000, min_y=6426000, max_x=538000, max_y=6428000),
        target_width=200,
        target_height=200,
        srs="EPSG:32633",
    )
    prj_text = (tmp_path / "scene.prj").read_text(encoding="ascii")
    assert "UTM zone 33N" in prj_text
    assert 'AUTHORITY["EPSG","32633"]' in prj_text
    assert (tmp_path / "scene.tfw").exists()


def test_write_prj_skipped_for_unknown_epsg(tmp_path: Path) -> None:
    out = tmp_path / "scene.tiff"
    out.write_bytes(b"x")
    _write_worldfile_and_prj(
        out,
        target_bbox=BBox(min_x=0, min_y=0, max_x=100, max_y=100),
        target_width=10,
        target_height=10,
        srs="EPSG:9999999",  # nonsense; helper should leave .prj absent
    )
    assert not (tmp_path / "scene.prj").exists()
    assert (tmp_path / "scene.tfw").exists()
