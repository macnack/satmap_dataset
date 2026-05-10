from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.parse import parse_qsl, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.providers.nls.oapif import build_items_url, parse_aoi_years


BASE = "https://avoin-karttakuva.maanmittauslaitos.fi/ortokuvat-ja-korkeusmallit/features/v2"


def test_build_items_url_uses_epsg_3067_bbox_crs():
    url = build_items_url(
        BASE, collection="ortokuva_vari", bbox=(351089, 6671973, 353089, 6673973)
    )
    qs = dict(parse_qsl(urlparse(url).query, keep_blank_values=True))
    assert qs["bbox"] == "351089,6671973,353089,6673973"
    assert "EPSG/0/3067" in qs["bbox-crs"]
    assert qs["f"] == "json"
    assert "/collections/ortokuva_vari/items" in url


def test_parse_aoi_years_collects_kuvausvuosi_set():
    payload = json.dumps(
        {
            "type": "FeatureCollection",
            "features": [
                {"properties": {"kuvausvuosi": "2013"}},
                {"properties": {"kuvausvuosi": "2015"}},
                {"properties": {"kuvausvuosi": "2015"}},  # duplicate, deduped
                {"properties": {"kuvausvuosi": "not-a-year"}},  # ignored
                {"properties": {}},  # ignored
            ],
        }
    ).encode("utf-8")
    assert parse_aoi_years(payload) == {2013, 2015}


def test_parse_aoi_years_handles_empty_collection():
    payload = b'{"type":"FeatureCollection","features":[]}'
    assert parse_aoi_years(payload) == set()


def test_parse_aoi_years_handles_invalid_json():
    assert parse_aoi_years(b"not json at all") == set()
