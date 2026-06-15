from satmap_dataset.models import CellEntry, TrajectoryManifest


def test_manifest_round_trips():
    m = TrajectoryManifest(
        track_path="gps_001",
        point_count=3204,
        srs="EPSG:2180",
        cell_m=1000.0,
        year_start=2020,
        year_end=2025,
        union_bbox_2180="410000.000,395000.000,443000.000,427000.000",
        cell_count=1,
        cells=[
            CellEntry(
                name="gps_001_x440_y430",
                ix=440,
                iy=430,
                bbox="440000.000,430000.000,441000.000,431000.000",
                bbox_wgs84="17.83,51.70,17.85,51.71",
                center_lat=51.705,
                center_lon=17.84,
            )
        ],
    )
    restored = TrajectoryManifest.model_validate_json(m.model_dump_json())
    assert restored.cell_count == 1
    assert restored.cells[0].name == "gps_001_x440_y430"
    assert restored.cells[0].download_status is None
