from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.providers.lantmateriet.stac import (
    StacItem,
    group_items_by_year,
    parse_stac_features,
    parse_stac_item,
    pick_item_for_bbox,
    select_asset,
)


FIXTURES = ROOT / "tests" / "fixtures" / "lantmateriet"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_parse_stac_features_single_year() -> None:
    payload = _load("stac_search_response_single_year.json")
    items = parse_stac_features(payload)
    assert len(items) == 1
    item = items[0]
    assert isinstance(item, StacItem)
    assert item.item_id == "ortofoto_2022_kisa_1"
    assert item.year == 2022
    assert item.epsg == 3006
    assert item.gsd == 0.16
    assert item.license == "CC-BY-4.0"
    assert item.attribution.startswith("©")
    assert item.bbox is not None and len(item.bbox) == 4
    assert any(asset.key == "data" for asset in item.assets)


def test_group_items_by_year_buckets_correctly() -> None:
    payload = _load("stac_search_response_multi_year.json")
    items = parse_stac_features(payload)
    grouped = group_items_by_year(items)
    assert sorted(grouped.keys()) == [2018, 2020, 2024]
    assert len(grouped[2020]) == 2


def test_pick_item_for_bbox_prefers_higher_overlap() -> None:
    payload = _load("stac_search_response_multi_year.json")
    items = parse_stac_features(payload)
    grouped = group_items_by_year(items)
    target_bbox = (536194.91, 6426213.28, 538194.91, 6428213.28)
    chosen = pick_item_for_bbox(grouped[2020], target_bbox)
    assert chosen is not None
    assert chosen.item_id == "ortofoto_2020_kisa_better"


def test_select_asset_prefers_cog_then_geotiff_then_tiff() -> None:
    payload = _load("stac_search_response_multi_year.json")
    items = parse_stac_features(payload)
    item_2024 = next(i for i in items if i.year == 2024)
    item_2018 = next(i for i in items if i.year == 2018)
    item_2020_better = next(i for i in items if i.item_id == "ortofoto_2020_kisa_better")
    item_2020_low = next(i for i in items if i.item_id == "ortofoto_2020_kisa")

    assert select_asset(item_2024).key == "cog"
    assert select_asset(item_2018).key == "geotiff"
    assert select_asset(item_2020_better).key == "data"
    chosen = select_asset(item_2020_low)
    assert chosen is not None
    assert chosen.href.endswith("kisa_low_overlap.tif")


def test_select_asset_returns_none_when_no_raster_assets() -> None:
    payload = _load("stac_item_missing_asset.json")
    item = parse_stac_item(payload)
    chosen = select_asset(item)
    # All assets are non-raster (jpeg / json) so we must not pick them as data.
    if chosen is not None:
        assert not chosen.href.endswith((".tif", ".tiff"))
        assert chosen.media_type not in {
            "image/tiff",
            "image/tiff; application=geotiff",
            "image/tiff; application=geotiff; profile=cloud-optimized",
        }


def test_parse_stac_item_handles_missing_properties_gracefully() -> None:
    item = parse_stac_item({"id": "x", "assets": {}, "bbox": None})
    assert item.item_id == "x"
    assert item.assets == ()
    assert item.bbox is None
    assert item.year is None
