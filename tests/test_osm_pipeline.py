from satmap_dataset.models import OsmCategoryAsset, OsmManifest, OsmYearAsset


def test_osm_manifest_round_trip_with_null_rasters():
    manifest = OsmManifest(
        bbox="348967,508503,349967,509503",
        bbox_wgs84="16.778,52.421,16.792,52.430",
        srs="EPSG:2180",
        target_width=15000,
        target_height=15000,
        categories=["buildings", "highways"],
        years=[
            OsmYearAsset(
                year=2022,
                snapshot_date="2022-04-29T00:00:00Z",
                categories={
                    "buildings": OsmCategoryAsset(
                        feature_count=1326,
                        raster_path="osm_x/year_2022_buildings.tif",
                    ),
                    "highways": OsmCategoryAsset(
                        feature_count=0,
                        raster_path=None,
                    ),
                },
                passed=True,
            ),
        ],
        passed=True,
    )
    blob = manifest.model_dump_json()
    restored = OsmManifest.model_validate_json(blob)
    assert restored.kind == "osm_manifest"
    assert restored.stage == "osm"
    assert restored.years[0].year == 2022
    assert restored.years[0].categories["buildings"].feature_count == 1326
    assert restored.years[0].categories["buildings"].raster_path == "osm_x/year_2022_buildings.tif"
    assert restored.years[0].categories["highways"].raster_path is None
    assert restored.passed is True
