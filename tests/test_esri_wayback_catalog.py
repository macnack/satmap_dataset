from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.providers.esri_wayback import catalog

FIXTURES = ROOT / "tests" / "fixtures" / "esri_wayback"
CAPS = FIXTURES / "wmts_capabilities_excerpt.xml"
CONFIG = FIXTURES / "waybackconfig_excerpt.json"


def test_parse_capabilities_orders_by_release_date_not_number() -> None:
    releases = catalog.parse_capabilities(CAPS.read_bytes())
    assert [r.release_num for r in releases] == [26334, 22869, 60013, 64776, 4230, 10]
    newest = releases[0]
    assert newest.identifier == "WB_2026_R07"
    assert newest.release_date == "2026-08-05"
    assert releases[-1].release_date == "2014-02-20"


def test_parse_capabilities_normalises_tile_template() -> None:
    release = catalog.parse_capabilities(CAPS.read_bytes())[0]
    assert release.tile_url(17, 73186, 43158) == (
        "https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/"
        "default028mm/MapServer/tile/26334/17/43158/73186"
    )


def test_metadata_url_derived_from_identifier() -> None:
    by_num = {r.release_num: r for r in catalog.parse_capabilities(CAPS.read_bytes())}
    assert by_num[26334].metadata_layer_url == (
        "https://metadata.maptiles.arcgis.com/arcgis/rest/services/World_Imagery_Metadata_2026_r07/MapServer"
    )
    assert catalog.derive_metadata_layer_url("WB_2014_R1") is not None
    assert catalog.derive_metadata_layer_url("not-a-release") is None


def test_apply_config_overlays_metadata_urls() -> None:
    releases = catalog.apply_config(catalog.parse_capabilities(CAPS.read_bytes()), CONFIG.read_text())
    by_num = {r.release_num: r for r in releases}
    expected = json.loads(CONFIG.read_text())["64776"]["metadataLayerUrl"]
    assert by_num[64776].metadata_layer_url == expected


def test_parse_capabilities_rejects_empty_document() -> None:
    with pytest.raises(catalog.CatalogParseError):
        catalog.parse_capabilities(b"<Capabilities><Contents/></Capabilities>")
    with pytest.raises(catalog.CatalogParseError):
        catalog.parse_capabilities(b"not xml")


def test_release_filter_dates_and_numbers() -> None:
    releases = catalog.parse_capabilities(CAPS.read_bytes())
    window = catalog.ReleaseFilter.from_options({"release_date_start": "2023-09", "release_date_end": "2026-03"})
    assert [r.release_num for r in releases if window.accepts(r)] == [22869, 60013]
    pick = catalog.ReleaseFilter.from_options({"release_numbers": [10, 26334], "exclude_release_numbers": 10})
    assert [r.release_num for r in releases if pick.accepts(r)] == [26334]
    with pytest.raises(ValueError):
        catalog.ReleaseFilter.from_options({"release_date_start": "March 2020"})
