"""Tests for Overpass relation (multipolygon) handling and area-category selectors."""
from satmap_dataset.osm import overpass_client as oc


def test_area_categories_query_relations_lines_do_not():
    # water and green can be multipolygon relations (lakes, large parks).
    assert 'relation["natural"="water"]' in oc.CATEGORY_QUERIES["water"]
    assert 'relation[' in oc.CATEGORY_QUERIES["green"]
    # line / point categories stay way-only.
    assert "relation" not in oc.CATEGORY_QUERIES["roads"]
    assert "relation" not in oc.CATEGORY_QUERIES["paths"]
    assert "relation" not in oc.CATEGORY_QUERIES["buildings"]


def test_way_polygon_and_line_still_convert():
    result = {
        "elements": [
            {"type": "way", "tags": {"building": "yes"},
             "geometry": [{"lon": 0, "lat": 0}, {"lon": 1, "lat": 0},
                          {"lon": 1, "lat": 1}, {"lon": 0, "lat": 0}]},
            {"type": "way", "tags": {"highway": "primary"},
             "geometry": [{"lon": 0, "lat": 0}, {"lon": 5, "lat": 5}]},
        ]
    }
    fc = oc._elements_to_geojson(result)
    geoms = [f["geometry"]["type"] for f in fc["features"]]
    assert geoms == ["Polygon", "LineString"]


def test_relation_multipolygon_assembled_from_segments_with_hole():
    # Outer ring split across two member ways; one inner ring (a hole).
    result = {
        "elements": [
            {
                "type": "relation",
                "tags": {"natural": "water", "type": "multipolygon"},
                "members": [
                    {"type": "way", "role": "outer",
                     "geometry": [{"lon": 0, "lat": 0}, {"lon": 4, "lat": 0}]},
                    {"type": "way", "role": "outer",
                     "geometry": [{"lon": 4, "lat": 0}, {"lon": 4, "lat": 4},
                                  {"lon": 0, "lat": 4}, {"lon": 0, "lat": 0}]},
                    {"type": "way", "role": "inner",
                     "geometry": [{"lon": 1, "lat": 1}, {"lon": 2, "lat": 1},
                                  {"lon": 2, "lat": 2}, {"lon": 1, "lat": 2},
                                  {"lon": 1, "lat": 1}]},
                ],
            }
        ]
    }
    fc = oc._elements_to_geojson(result)
    assert len(fc["features"]) == 1
    geom = fc["features"][0]["geometry"]
    assert geom["type"] == "Polygon"
    rings = geom["coordinates"]
    assert len(rings) == 2  # outer + 1 hole
    outer = rings[0]
    assert outer[0] == outer[-1]  # closed
    assert len(outer) >= 5  # the 4-corner square assembled from 2 segments
    # the inner ring is the small square
    assert rings[1][0] == rings[1][-1]


def test_relation_without_geometry_is_skipped():
    result = {"elements": [{"type": "relation", "tags": {"natural": "water"}, "members": []}]}
    fc = oc._elements_to_geojson(result)
    assert fc["features"] == []
