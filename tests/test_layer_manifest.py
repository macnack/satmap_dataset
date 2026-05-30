from __future__ import annotations

from satmap_dataset.models import (
    LayerManifest,
    LayerYearAsset,
    ReferenceGrid,
    TileAcquisitionMetadata,
)


def test_layer_year_asset_round_trip():
    asset = LayerYearAsset(
        year=2024,
        snapshot_date="2024-06-01T00:00:00Z",
        source="wfs",
        assets={"rgb": "rendered_poznan/year_2024.tiff"},
        native_paths={"nmt": "dem_poznan/native/nmt.tif"},
        feature_counts={"buildings": 42},
        acquisition={"t1": TileAcquisitionMetadata(acquisition_date="2024-06-01")},
        passed=True,
    )
    restored = LayerYearAsset.model_validate_json(asset.model_dump_json())
    assert restored == asset
    assert restored.assets["rgb"].endswith("year_2024.tiff")


def test_layer_manifest_round_trip_and_discriminator():
    manifest = LayerManifest(
        layer="geoportal_rgb",
        role="rgb",
        stage="render",
        provider="geoportal",
        grid=ReferenceGrid(
            bbox="210300,521900,210500,522100",
            width=3000,
            height=3000,
            srs="EPSG:2180",
            px_per_meter=15.0,
        ),
        bands=["red", "green", "blue"],
        years_requested=[2023, 2024],
        years_included=[2023, 2024],
        years_source_map={2023: "wfs", 2024: "wms"},
        years=[
            LayerYearAsset(year=2023, passed=True),
            LayerYearAsset(year=2024, passed=True),
        ],
        assets=["rendered_poznan/year_2023.tiff", "rendered_poznan/year_2024.tiff"],
        passed=True,
    )
    restored = LayerManifest.model_validate_json(manifest.model_dump_json())
    assert restored.kind == "layer_manifest"
    assert restored.layer == "geoportal_rgb"
    assert restored.role == "rgb"
    assert restored.grid is not None and restored.grid.width == 3000
    assert restored.years_source_map == {2023: "wfs", 2024: "wms"}
    assert len(restored.years) == 2


def test_layer_manifest_required_fields():
    # layer + role are required; everything else defaults.
    assert LayerManifest.model_fields["layer"].is_required()
    assert LayerManifest.model_fields["role"].is_required()
    assert not LayerManifest.model_fields["grid"].is_required()
    assert not LayerManifest.model_fields["passed"].is_required()


def test_layer_manifest_defaults_minimal():
    manifest = LayerManifest(layer="osm", role="labels")
    assert manifest.kind == "layer_manifest"
    assert manifest.provider is None
    assert manifest.grid is None
    assert manifest.passed is False
    assert manifest.assets == []
