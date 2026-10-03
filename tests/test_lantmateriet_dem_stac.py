from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.providers.lantmateriet import dem as lm_dem
from satmap_dataset.providers.lantmateriet import stac

FIXTURES = ROOT / "tests" / "fixtures" / "lantmateriet"


def test_resolve_dem_search_options_uses_stac_hojd_not_bild():
    opts = lm_dem.resolve_dem_search_options({})
    assert "stac-hojd" in opts.url
    assert "stac-bild" not in opts.url
    assert opts.collections == ("dtm-cog",)


def test_resolve_dem_search_options_honors_overrides(monkeypatch):
    monkeypatch.setenv(
        "SATMAP_LANTMATERIET_STAC_HOJD_URL",
        "https://example.test/stac-hojd/v1/search",
    )
    monkeypatch.setenv("SATMAP_LANTMATERIET_STAC_HOJD_COLLECTION", "mhm-65_6")
    opts = lm_dem.resolve_dem_search_options({"page_limit": 10})
    assert opts.url == "https://example.test/stac-hojd/v1/search"
    assert opts.collections == ("mhm-65_6",)
    assert opts.page_limit == 10


def test_dem_credentials_prefer_dem_specific_env(monkeypatch):
    monkeypatch.setenv("SATMAP_LANTMATERIET_USERNAME", "orto_user")
    monkeypatch.setenv("SATMAP_LANTMATERIET_PASSWORD", "orto_pass")
    monkeypatch.setenv("SATMAP_LANTMATERIET_DEM_USERNAME", "dem_user")
    monkeypatch.setenv("SATMAP_LANTMATERIET_DEM_PASSWORD", "dem_pass")
    headers = lm_dem.auth_headers({})
    assert headers["Authorization"].startswith("Basic ")
    # Decode without printing secrets: dem_user must be in the token.
    import base64

    token = headers["Authorization"].split(" ", 1)[1]
    decoded = base64.b64decode(token).decode("utf-8")
    assert decoded.startswith("dem_user:")


def test_parse_dtm_cog_fixture_selects_cog_asset():
    payload = json.loads((FIXTURES / "stac_hojd_dtm_cog_search.json").read_text())
    items = stac.parse_stac_features(payload)
    assert items
    pairs = lm_dem.items_with_raster_assets(items)
    assert pairs
    item, asset = pairs[0]
    assert item.collection == "dtm-cog"
    assert asset.key == "data"
    assert "grid/mhm" in asset.href or asset.href.endswith(".tif")
    assert "tiff" in (asset.media_type or "").lower()
