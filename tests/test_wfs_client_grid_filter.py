from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.geoportal import wfs_client


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


def _feature_xml(*, tile_id: str, url: str, uklad_xy: str, kolor: str, min_x: float, min_y: float, max_x: float, max_y: float) -> str:
    return f"""
  <wfs:member>
    <gugik:SkorowidzOrtofomapy2015>
      <gugik:msGeometry>
        <gml:Polygon srsName="urn:ogc:def:crs:EPSG::2180">
          <gml:exterior>
            <gml:LinearRing>
              <gml:posList>{min_x} {min_y} {max_x} {min_y} {max_x} {max_y} {min_x} {max_y} {min_x} {min_y}</gml:posList>
            </gml:LinearRing>
          </gml:exterior>
        </gml:Polygon>
      </gugik:msGeometry>
      <gugik:godlo>{tile_id}</gugik:godlo>
      <gugik:uklad_xy>{uklad_xy}</gugik:uklad_xy>
      <gugik:kolor>{kolor}</gugik:kolor>
      <gugik:url_do_pobrania>{url}</gugik:url_do_pobrania>
    </gugik:SkorowidzOrtofomapy2015>
  </wfs:member>
"""


def _collection_xml(*, feature_count: int, members: list[str]) -> str:
    return (
        f"""<?xml version="1.0" encoding="UTF-8"?>
<wfs:FeatureCollection xmlns:gugik="http://www.gugik.gov.pl" xmlns:gml="http://www.opengis.net/gml/3.2" xmlns:wfs="http://www.opengis.net/wfs/2.0" numberMatched="{feature_count}" numberReturned="{feature_count}">
"""
        + "".join(members)
        + "\n</wfs:FeatureCollection>\n"
    )


def test_wfs_prefers_srs_compatible_and_rgb(monkeypatch) -> None:
    xml = _collection_xml(
        feature_count=2,
        members=[
            _feature_xml(
                tile_id="M-34-27-C-a-4-1",
                url="https://opendata.geoportal.gov.pl/ortofotomapa/69801/69801_233894_M-34-27-C-a-4-1.tif",
                uklad_xy="PL-1992",
                kolor="CIR",
                min_x=359711.4,
                min_y=504372.49,
                max_x=362029.89,
                max_y=506561.69,
            ),
            _feature_xml(
                tile_id="M-34-27-C-a-4-1",
                url="https://opendata.geoportal.gov.pl/ortofotomapa/69802/69802_233893_M-34-27-C-a-4-1.tif",
                uklad_xy="PL-1992",
                kolor="RGB",
                min_x=359711.4,
                min_y=504372.49,
                max_x=362029.89,
                max_y=506561.69,
            ),
        ],
    )

    async def fake_request_with_retry(*args, **kwargs):
        return _FakeResponse(xml)

    monkeypatch.setattr(wfs_client, "request_with_retry", fake_request_with_retry)

    status, tiles, tile_bboxes = asyncio.run(
        wfs_client.get_year_tiles(
            2015,
            "359700,504900,361700,506900",
            "EPSG:2180",
            year_to_typename={2015: "gugik:SkorowidzOrtofomapy2015"},
        )
    )

    assert status.status == "has_features"
    assert tiles["M-34-27-C-a-4-1"].endswith("69802_233893_M-34-27-C-a-4-1.tif")
    assert tile_bboxes["M-34-27-C-a-4-1"] == [359711.4, 504372.49, 362029.89, 506561.69]


def test_wfs_rejects_incompatible_grid_for_epsg2180(monkeypatch) -> None:
    xml = _collection_xml(
        feature_count=1,
        members=[
            _feature_xml(
                tile_id="6.149.30.22",
                url="https://opendata.geoportal.gov.pl/ortofotomapa/66049/66049_510151_6.149.30.22.tif",
                uklad_xy="PL-2000:S6",
                kolor="RGB",
                min_x=360924.706554,
                min_y=503580.447838,
                max_x=361945.647633,
                max_y=505192.772630,
            ),
        ],
    )

    async def fake_request_with_retry(*args, **kwargs):
        return _FakeResponse(xml)

    monkeypatch.setattr(wfs_client, "request_with_retry", fake_request_with_retry)

    status, tiles, tile_bboxes = asyncio.run(
        wfs_client.get_year_tiles(
            2015,
            "359700,504900,361700,506900",
            "EPSG:2180",
            year_to_typename={2015: "gugik:SkorowidzOrtofomapy2015"},
        )
    )

    assert status.status == "zero_features"
    assert status.feature_count == 1
    assert status.reason is not None
    assert "compatible with requested SRS" in status.reason
    assert tiles == {}
    assert tile_bboxes == {}


def test_wfs_uses_startindex_paging(monkeypatch) -> None:
    page0 = _collection_xml(
        feature_count=3,
        members=[
            _feature_xml(
                tile_id="M-34-27-C-a-4-1",
                url="https://opendata.geoportal.gov.pl/ortofotomapa/69802/69802_233893_M-34-27-C-a-4-1.tif",
                uklad_xy="PL-1992",
                kolor="RGB",
                min_x=359711.4,
                min_y=504372.49,
                max_x=362029.89,
                max_y=506561.69,
            ),
            _feature_xml(
                tile_id="M-34-27-C-a-4-2",
                url="https://opendata.geoportal.gov.pl/ortofotomapa/69802/69802_233896_M-34-27-C-a-4-2.tif",
                uklad_xy="PL-1992",
                kolor="RGB",
                min_x=359713.72,
                min_y=506558.74,
                max_x=362033.14,
                max_y=508748.92,
            ),
        ],
    )
    page2 = _collection_xml(
        feature_count=3,
        members=[
            _feature_xml(
                tile_id="M-34-27-C-a-4-3",
                url="https://opendata.geoportal.gov.pl/ortofotomapa/69802/69802_233897_M-34-27-C-a-4-3.tif",
                uklad_xy="PL-1992",
                kolor="RGB",
                min_x=357395.46,
                min_y=504374.46,
                max_x=359713.72,
                max_y=506564.64,
            ),
        ],
    )
    requested_start_indexes: list[str] = []

    async def fake_request_with_retry(*args, **kwargs):
        params = kwargs.get("params", {})
        requested_start_indexes.append(str(params.get("STARTINDEX", "")))
        if str(params.get("STARTINDEX")) == "0":
            return _FakeResponse(page0)
        if str(params.get("STARTINDEX")) == "2":
            return _FakeResponse(page2)
        return _FakeResponse(_collection_xml(feature_count=3, members=[]))

    monkeypatch.setattr(wfs_client, "request_with_retry", fake_request_with_retry)
    monkeypatch.setattr(wfs_client, "DEFAULT_PAGE_SIZE", 2)

    status, tiles, _ = asyncio.run(
        wfs_client.get_year_tiles(
            2015,
            "359700,504900,361700,506900",
            "EPSG:2180",
            year_to_typename={2015: "gugik:SkorowidzOrtofomapy2015"},
        )
    )

    assert status.status == "has_features"
    assert requested_start_indexes[:2] == ["0", "2"]
    assert sorted(tiles.keys()) == ["M-34-27-C-a-4-1", "M-34-27-C-a-4-2", "M-34-27-C-a-4-3"]


def test_wfs_normalizes_swapped_feature_bbox_axes(monkeypatch) -> None:
    # Feature geometry is encoded as y,x pairs (swapped axis order),
    # but output tile_bboxes must be strict xmin,ymin,xmax,ymax.
    xml = _collection_xml(
        feature_count=1,
        members=[
            _feature_xml(
                tile_id="M-34-27-C-a-4-1",
                url="https://opendata.geoportal.gov.pl/ortofotomapa/69802/69802_233893_M-34-27-C-a-4-1.tif",
                uklad_xy="PL-1992",
                kolor="RGB",
                # Intentionally swapped in source geometry:
                # x-range 359711.4..362029.89, y-range 504372.49..506561.69
                min_x=504372.49,
                min_y=359711.4,
                max_x=506561.69,
                max_y=362029.89,
            ),
        ],
    )

    async def fake_request_with_retry(*args, **kwargs):
        return _FakeResponse(xml)

    monkeypatch.setattr(wfs_client, "request_with_retry", fake_request_with_retry)

    status, _, tile_bboxes = asyncio.run(
        wfs_client.get_year_tiles(
            2015,
            "359700,504900,361700,506900",
            "EPSG:2180",
            year_to_typename={2015: "gugik:SkorowidzOrtofomapy2015"},
        )
    )

    assert status.status == "has_features"
    assert tile_bboxes["M-34-27-C-a-4-1"] == [359711.4, 504372.49, 362029.89, 506561.69]
