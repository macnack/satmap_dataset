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


def test_dem_manifest_skorowidz_round_trip():
    from satmap_dataset.models import DemYearAsset

    manifest = DemManifest(
        bbox="0,0,10,10", srs="EPSG:2180", vertical_datum="kron86",
        transport="skorowidz", years_requested=[2012, 2019], years_skipped={2015: "no tiles in AOI"},
        products=[
            DemProductAsset(
                product="nmt", coverage_id="skorowidz:nmt:kron86", endpoint="https://example/wfs",
                years=[
                    DemYearAsset(
                        year=2012, native_path="dem_x/skorowidz/nmt_kron86/native/year_2012.tif",
                        native_width=10, native_height=10, tile_count=2,
                        godla=["N-33-141-C-a-3-4"], passed=True,
                    )
                ],
                passed=True,
            )
        ],
        passed=True,
    )
    restored = DemManifest.model_validate_json(manifest.model_dump_json())
    assert restored.transport == "skorowidz"
    assert restored.years_requested == [2012, 2019]
    assert restored.years_skipped == {2015: "no tiles in AOI"}
    assert restored.products[0].years[0].year == 2012
    assert restored.products[0].years[0].godla == ["N-33-141-C-a-3-4"]
    assert restored.products[0].years[0].mean_height_error is None
