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
