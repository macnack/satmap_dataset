import pytest

from satmap_dataset.geoportal import wcs_client


def test_coverage_id_all_combinations():
    assert wcs_client.coverage_id("nmt", "evrf2007") == "DTM_PL-EVRF2007-NH_TIFF"
    assert wcs_client.coverage_id("nmt", "kron86") == "DTM_PL-KRON86-NH_TIFF"
    assert wcs_client.coverage_id("nmpt", "evrf2007") == "DSM_PL-EVRF2007-NH_TIFF"
    assert wcs_client.coverage_id("nmpt", "kron86") == "DSM_PL-KRON86-NH_TIFF"


def test_coverage_id_rejects_unknown():
    with pytest.raises(ValueError):
        wcs_client.coverage_id("foo", "evrf2007")
    with pytest.raises(ValueError):
        wcs_client.coverage_id("nmt", "wgs84")


def test_endpoint_url_default_and_override():
    assert "NMT/GRID1/WCS" in wcs_client.endpoint_url("nmt")
    assert "NMPT/GRID1/WCS" in wcs_client.endpoint_url("nmpt")
    custom = {"endpoints": {"nmt": "https://example/custom"}}
    assert wcs_client.endpoint_url("nmt", custom) == "https://example/custom"


def test_split_bbox_single_tile_when_within_cap():
    tiles = wcs_client.split_bbox((0.0, 0.0, 100.0, 100.0), max_request_px=2048, gsd_m=1.0)
    assert tiles == [(0.0, 0.0, 100.0, 100.0)]


def test_split_bbox_tiles_and_covers_exactly():
    bbox = (0.0, 0.0, 250.0, 150.0)
    tiles = wcs_client.split_bbox(bbox, max_request_px=100, gsd_m=1.0)
    # 250m/100m -> 3 cols, 150m/100m -> 2 rows
    assert len(tiles) == 6
    assert min(t[0] for t in tiles) == 0.0
    assert min(t[1] for t in tiles) == 0.0
    assert max(t[2] for t in tiles) == 250.0
    assert max(t[3] for t in tiles) == 150.0
    for x0, y0, x1, y1 in tiles:
        assert x1 - x0 <= 100.0 + 1e-9
        assert y1 - y0 <= 100.0 + 1e-9


def test_split_bbox_rejects_bad_cap():
    with pytest.raises(ValueError):
        wcs_client.split_bbox((0, 0, 10, 10), max_request_px=0)
