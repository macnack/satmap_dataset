from satmap_dataset.config import DemConfig
from satmap_dataset.models import DemProductAsset, DemYearAsset, LayerManifest
from satmap_dataset.pipeline.dem import build_dem_layer_manifest


def test_dem_sub_assets_round_trip():
    asset = DemProductAsset(
        product="nmt",
        coverage_id="DTM_PL-EVRF2007-NH_TIFF",
        endpoint="https://example/wcs",
        native_path="dem_x/native/nmt_evrf2007.tif",
        native_width=10,
        native_height=10,
        tile_count=1,
        passed=True,
    )
    restored = DemProductAsset.model_validate_json(asset.model_dump_json())
    assert restored.product == "nmt"
    assert restored.coverage_id == "DTM_PL-EVRF2007-NH_TIFF"


def test_build_dem_layer_manifest_wcs():
    cfg = DemConfig(
        bbox="0,0,10,10",
        srs="EPSG:2180",
        vertical_datum="evrf2007",
        products=["nmt", "nmpt"],
        align_to_render=False,
        dem_root="dem_x",
        output_json="dem_x/dem_manifest.json",
    )
    products = [
        DemProductAsset(
            product="nmt",
            coverage_id="c1",
            endpoint="e1",
            native_path="dem_x/native/nmt.tif",
            aligned_path="dem_x/aligned/nmt.tif",
            passed=True,
        ),
        DemProductAsset(
            product="nmpt",
            coverage_id="c2",
            endpoint="e2",
            native_path="dem_x/native/nmpt.tif",
            passed=True,
        ),
    ]
    manifest = build_dem_layer_manifest(
        cfg, products, transport="wcs", years_skipped={}, grid=None,
        passed=True, errors=[], notes="wcs",
    )
    restored = LayerManifest.model_validate_json(manifest.model_dump_json())
    assert restored.role == "dem"
    assert restored.layer == "dem"
    assert restored.bands == ["nmt", "nmpt"]
    # WCS is not year-aware -> no per-year entries.
    assert restored.years == []
    # aligned preferred, native fallback.
    assert "dem_x/aligned/nmt.tif" in restored.assets
    assert "dem_x/native/nmpt.tif" in restored.assets
    assert restored.provider_metadata["transport"] == "wcs"
    assert len(restored.provider_metadata["products"]) == 2


def test_build_dem_layer_manifest_skorowidz_year_aware():
    cfg = DemConfig(
        bbox="0,0,10,10",
        srs="EPSG:2180",
        vertical_datum="kron86",
        transport="skorowidz",
        year_start=2012,
        year_end=2019,
        products=["nmt"],
        dem_root="dem_x",
        output_json="dem_x/dem_manifest.json",
    )
    products = [
        DemProductAsset(
            product="nmt",
            coverage_id="skorowidz:nmt:kron86",
            endpoint="https://example/wfs",
            years=[
                DemYearAsset(
                    year=2012,
                    native_path="dem_x/skorowidz/nmt_kron86/native/year_2012.tif",
                    aligned_path="dem_x/skorowidz/nmt_kron86/aligned/year_2012.tif",
                    tile_count=2,
                    godla=["N-33-141-C-a-3-4"],
                    passed=True,
                )
            ],
            passed=True,
        )
    ]
    manifest = build_dem_layer_manifest(
        cfg, products, transport="skorowidz", years_skipped={2015: "no tiles in AOI"},
        grid=None, passed=True, errors=[], notes="skorowidz",
    )
    assert manifest.years_included == [2012]
    assert manifest.years_excluded_with_reason == {2015: "no tiles in AOI"}
    year = manifest.years[0]
    assert year.year == 2012
    assert year.native_paths["nmt"].endswith("native/year_2012.tif")
    assert year.assets["nmt"].endswith("aligned/year_2012.tif")
