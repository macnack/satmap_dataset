from satmap_dataset.models import DemManifest, DemProductAsset


def test_dem_manifest_round_trip():
    manifest = DemManifest(
        bbox="0,0,10,10",
        srs="EPSG:2180",
        vertical_datum="evrf2007",
        products=[
            DemProductAsset(
                product="nmt",
                coverage_id="DTM_PL-EVRF2007-NH_TIFF",
                endpoint="https://example/wcs",
                native_path="dem_x/native/nmt_evrf2007.tif",
                native_width=10,
                native_height=10,
                tile_count=1,
                passed=True,
            )
        ],
        passed=True,
    )
    blob = manifest.model_dump_json()
    restored = DemManifest.model_validate_json(blob)
    assert restored.kind == "dem_manifest"
    assert restored.stage == "dem"
    assert restored.products[0].product == "nmt"
    assert restored.products[0].coverage_id == "DTM_PL-EVRF2007-NH_TIFF"
    assert restored.passed is True
