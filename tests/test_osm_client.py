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


def test_category_filters_all_present():
    expected = {"buildings", "highways", "landuse", "water"}
    assert set(ohsome_client.CATEGORY_FILTERS.keys()) == expected


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
    assert captured["data"]["clipGeometry"] == "true"


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
