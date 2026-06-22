from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import parse_qsl, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.providers.nls.wcs import (
    build_describe_coverage_url,
    build_get_coverage_url,
)


WCS_BASE = "https://avoin-karttakuva.maanmittauslaitos.fi/ortokuvat-ja-korkeusmallit/wcs/v2"


def _qs(url: str) -> dict[str, str]:
    return dict(parse_qsl(urlparse(url).query, keep_blank_values=True))


def test_describe_coverage_url_shape():
    url = build_describe_coverage_url(WCS_BASE, coverage_id="ortokuva_vari")
    assert urlparse(url).path.endswith("/wcs/v2")
    qs = _qs(url)
    assert qs["service"] == "WCS"
    assert qs["version"] == "2.0.1"
    assert qs["request"] == "DescribeCoverage"
    assert qs["coverageID"] == "ortokuva_vari"


def test_get_coverage_url_includes_subsets_and_geotiff_options():
    url = build_get_coverage_url(
        WCS_BASE,
        coverage_id="ortokuva_vari",
        bbox=(393450, 7495450, 393650, 7495650),
        year=2010,
    )
    qs = _qs(url)
    assert qs["service"] == "WCS"
    assert qs["version"] == "2.0.1"
    assert qs["request"] == "GetCoverage"
    assert qs["CoverageID"] == "ortokuva_vari"
    assert qs["format"] == "image/tiff"
    assert qs["geotiff:compression"] == "LZW"
    assert qs["geotiff:tiling"] == "true"
    assert qs["geotiff:tilewidth"] == "256"
    assert qs["geotiff:tileheight"] == "256"
    assert "EPSG/0/3067" in qs["SubsettingCRS"]
    assert "EPSG/0/3067" in qs["OutputCRS"]
    pairs = parse_qsl(urlparse(url).query, keep_blank_values=True)
    subsets = [v for k, v in pairs if k == "SUBSET"]
    assert any(s.startswith("E(393450") for s in subsets)
    assert any(s.startswith("N(7495450") for s in subsets)
    assert any('time("2010-12-31' in s for s in subsets)


def test_get_coverage_url_uses_provided_base():
    url = build_get_coverage_url(
        "https://example.test/wcs/v2",
        coverage_id="ortokuva_vari",
        bbox=(0, 0, 1000, 1000),
        year=2020,
    )
    assert url.startswith("https://example.test/wcs/v2?")
