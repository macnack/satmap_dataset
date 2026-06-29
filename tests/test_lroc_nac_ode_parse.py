from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.providers.lroc_nac.ode import (
    OdeProduct,
    build_query_url,
    group_products_by_year,
    parse_products,
)

FIXTURES = ROOT / "tests" / "fixtures" / "lroc_nac"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_build_query_url_has_required_params() -> None:
    url = build_query_url(
        "https://oderest.rsl.wustl.edu/live2",
        product_type="CDRNAC4",
        westlon=30.6, eastlon=30.9, minlat=20.0, maxlat=20.35,
        min_obtime="2009-01-01", max_obtime="2026-12-31",
    )
    for fragment in (
        "query=product", "target=moon", "ihid=LRO", "iid=LROC",
        "pt=CDRNAC4", "westernlon=30.6", "easternlon=30.9",
        "minlat=20", "maxlat=20.35", "loc=f",
        "minobtime=2009-01-01", "maxobtime=2026-12-31",
        "results=opmf", "output=JSON",
    ):
        assert fragment in url


def test_parse_products_two_years() -> None:
    products = parse_products(_load("ode_search_two_years.json"))
    assert len(products) == 2
    first = products[0]
    assert isinstance(first, OdeProduct)
    assert first.pdsid == "M101013931LC"
    assert first.acquisition_year == 2009
    assert first.incidence_angle == 42.5
    assert first.map_resolution == 0.5
    assert first.file_url == "https://pds.example/M101013931LC.IMG"
    assert first.file_bytes == 51200 * 1024
    assert first.footprint_bbox == (30.60, 20.05, 30.72, 20.30)
    # Single-object Product_file (not a list) still resolves a URL:
    assert products[1].file_url == "https://pds.example/M198273648LC.IMG"


def test_group_products_by_year() -> None:
    grouped = group_products_by_year(parse_products(_load("ode_search_two_years.json")))
    assert sorted(grouped.keys()) == [2009, 2012]
    assert len(grouped[2009]) == 1 and len(grouped[2012]) == 1
