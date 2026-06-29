import asyncio
import pytest
from satmap_dataset.osm import ohsome_client


def test_bbox_epsg2180_to_wgs84_known_values():
    result = ohsome_client.bbox_epsg2180_to_wgs84(
        "348967.353,508503.706,349967.353,509503.706"
    )
    parts = [float(x) for x in result.split(",")]
    lon_min, lat_min, lon_max, lat_max = parts
    assert abs(lon_min - 16.778248) < 0.001
    assert abs(lat_min - 52.421547) < 0.001
    assert abs(lon_max - 16.792497) < 0.001
    assert abs(lat_max - 52.430809) < 0.001


def test_bbox_to_wgs84_epsg3006_sweden():
    # Kiruna AOI in EPSG:3006 (SWEREF99 TM) -> WGS84 lon/lat.
    result = ohsome_client.bbox_to_wgs84(
        "717646.631,7534133.478,721519.615,7538006.462", "EPSG:3006"
    )
    lon_min, lat_min, lon_max, lat_max = (float(x) for x in result.split(","))
    assert abs(lon_min - 20.175606) < 0.001
    assert abs(lat_min - 67.839960) < 0.001
    assert abs(lon_max - 20.275062) < 0.001
    assert abs(lat_max - 67.871624) < 0.001


def test_bbox_to_wgs84_epsg2180_matches_legacy_helper():
    bbox = "348967.353,508503.706,349967.353,509503.706"
    assert ohsome_client.bbox_to_wgs84(bbox, "EPSG:2180") == (
        ohsome_client.bbox_epsg2180_to_wgs84(bbox)
    )


def test_category_filters_all_present():
    # ohsome_client.CATEGORY_FILTERS is legacy; current pipeline uses overpass_client.CATEGORY_QUERIES
    from satmap_dataset.osm.overpass_client import CATEGORY_QUERIES
    assert set(CATEGORY_QUERIES.keys()) == {"buildings", "roads", "paths", "green", "water"}


def test_build_query_uses_global_bbox_for_all_categories():
    """Union categories (green/water) must not get a bbox applied to the group.

    The valid form is a global [bbox:S,W,N,E] setting; `(union)(bbox)` is a 400.
    """
    from satmap_dataset.osm import overpass_client

    bbox = "67.83,20.17,67.87,20.27"  # S,W,N,E
    for category in ("buildings", "roads", "paths", "green", "water"):
        q = overpass_client._build_query(category, bbox, "2021-07-01T00:00:00Z")
        assert f"[bbox:{bbox}]" in q, q
        # The broken pattern applied a bbox to the parenthesised union/statement.
        assert f"({bbox})" not in q, q
        assert q.endswith("out geom;")

    # green is a union of multiple way[...] statements, all inside one (...).
    green = overpass_client._build_query("green", bbox, "2021-07-01T00:00:00Z")
    assert green.count("way[") >= 3, green


def test_overpass_request_sends_json_accept_header(monkeypatch):
    """overpass-api.de returns 406 unless Accept: application/json is sent."""
    from satmap_dataset.osm import overpass_client

    captured = {}

    class _FakeResponse:
        def json(self):
            return {"elements": []}

    async def _fake_request(method, url, *, data, timeout, retry_policy, headers=None, **kwargs):
        captured["headers"] = headers
        captured["data"] = data
        return _FakeResponse()

    monkeypatch.setattr(overpass_client, "request_with_retry", _fake_request)

    asyncio.run(
        overpass_client.get_elements_geometry(
            "20.17,67.83,20.27,67.87", "buildings", "2021-07-01"
        )
    )
    assert captured["headers"] is not None
    assert captured["headers"].get("Accept") == "application/json"


def test_get_elements_geometry_builds_correct_post(monkeypatch):
    captured = {}

    class _FakeResponse:
        def json(self):
            return {"type": "FeatureCollection", "features": []}

    async def _fake_request(method, url, *, data, timeout, retry_policy, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["data"] = data
        return _FakeResponse()

    monkeypatch.setattr(ohsome_client, "request_with_retry", _fake_request)

    result = asyncio.run(
        ohsome_client.get_elements_geometry(
            "16.778,52.421,16.792,52.430",
            "building=* and type:way",
            "2022-04-29",
        )
    )
    assert result == {"type": "FeatureCollection", "features": []}
    assert captured["method"] == "POST"
    assert captured["url"].endswith("/elements/geometry")
    assert captured["data"]["bboxes"] == "16.778,52.421,16.792,52.430"
    assert captured["data"]["filter"] == "building=* and type:way"
    assert captured["data"]["time"] == "2022-04-29T00:00:00Z"
    assert captured["data"]["clipGeometry"] == "false"


def test_get_elements_geometry_normalizes_time_with_z(monkeypatch):
    captured = {}

    class _FakeResponse:
        def json(self):
            return {"type": "FeatureCollection", "features": []}

    async def _fake_request(method, url, *, data, timeout, retry_policy, **kwargs):
        captured["time"] = data["time"]
        return _FakeResponse()

    monkeypatch.setattr(ohsome_client, "request_with_retry", _fake_request)

    asyncio.run(
        ohsome_client.get_elements_geometry(
            "16.778,52.421,16.792,52.430",
            "building=* and type:way",
            "2022-04-29T00:00:00Z",
        )
    )
    assert captured["time"] == "2022-04-29T00:00:00Z"
