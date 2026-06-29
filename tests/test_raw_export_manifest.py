from satmap_dataset.models import RawExportManifest


def test_raw_export_manifest_defaults_and_roundtrip():
    m = RawExportManifest(
        provider="geoportal",
        area="poznan",
        raw_root="/data/sat_data_raw",
        epsg=2180,
        cell_size_m=[2500.0, 2500.0],
        exported_tile_counts_by_year={2015: 4, 2018: 4},
        cells_produced=2,
        seasons_kept=3,
        seasons_dropped=1,
    )
    assert m.kind == "raw_export_manifest"
    assert m.stage == "raw_export"
    assert m.epsg_provider_mismatch is False
    assert m.link_mode == "symlink"
    assert m.passed is False  # explicit pass set by the stage
    dumped = m.model_dump(mode="json")
    assert dumped["exported_tile_counts_by_year"]["2015"] == 4
    assert RawExportManifest.model_validate(dumped).cells_produced == 2
