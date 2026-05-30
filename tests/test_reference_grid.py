from satmap_dataset.models import (
    LayerManifest,
    ReferenceGrid,
    TileAcquisitionMetadata,
)


def test_reference_grid_round_trip():
    grid = ReferenceGrid(
        bbox="210300,521900,210500,522100",
        width=3000,
        height=3000,
        srs="EPSG:2180",
        px_per_meter=15.0,
        year_date_map={2024: "2024-06-01T00:00:00Z"},
    )
    restored = ReferenceGrid.model_validate_json(grid.model_dump_json())
    assert restored.bbox == "210300,521900,210500,522100"
    assert restored.width == 3000
    assert restored.height == 3000
    assert restored.srs == "EPSG:2180"
    assert restored.px_per_meter == 15.0
    assert restored.year_date_map == {2024: "2024-06-01T00:00:00Z"}


def test_reference_grid_requires_positive_dims():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ReferenceGrid(bbox="0,0,1,1", width=0, height=10, srs="EPSG:2180")


def test_from_render_manifest_extracts_grid_and_year_dates():
    manifest = LayerManifest(
        layer="geoportal_rgb",
        role="rgb",
        stage="render",
        provider="geoportal",
        years_included=[2023, 2024],
        target_bbox="210300,521900,210500,522100",
        target_width=3000,
        target_height=3000,
        target_srs="EPSG:2180",
        px_per_meter=15.0,
        tile_acquisition_by_year={
            2023: {
                "t1": TileAcquisitionMetadata(acquisition_date="2023-05-01"),
                "t2": TileAcquisitionMetadata(acquisition_date="2023-07-15"),
            },
            2024: {
                "t3": TileAcquisitionMetadata(acquisition_date="2024-06-01"),
                "t4": TileAcquisitionMetadata(acquisition_date=None),
            },
        },
    )

    grid = ReferenceGrid.from_render_manifest(manifest)
    assert grid.bbox == "210300,521900,210500,522100"
    assert grid.width == 3000
    assert grid.height == 3000
    assert grid.srs == "EPSG:2180"
    assert grid.px_per_meter == 15.0
    # max acquisition date per year, skipping None
    assert grid.year_date_map == {2023: "2023-07-15", 2024: "2024-06-01"}
