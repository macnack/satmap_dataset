from satmap_dataset.models import DemAvailabilityEntry, DemAvailabilityReport


def test_availability_report_round_trip():
    report = DemAvailabilityReport(
        aoi_bbox="0,0,10,10", srs="EPSG:2180",
        entries=[
            DemAvailabilityEntry(
                product="nmpt", datum="evrf2007", year=2024,
                godla=["N-33-130-D-a-3-3", "N-33-130-D-a-3-4"], tile_count=2,
                formats=["asc", "xyz.zip"], coverage="full", coverage_pct=100.0,
                acquisition_dates=["2024-03-01"],
            ),
            DemAvailabilityEntry(
                product="nmpt", datum="evrf2007", year=2020,
                godla=[], tile_count=0, formats=[], coverage="none", coverage_pct=0.0,
                acquisition_dates=[],
            ),
        ],
        errors={"nmt|kron86": "capabilities timeout"},
        full_coverage_options=[{"product": "nmpt", "datum": "evrf2007", "year": 2024}],
    )
    restored = DemAvailabilityReport.model_validate_json(report.model_dump_json())
    assert restored.kind == "dem_availability"
    assert restored.entries[0].coverage == "full"
    assert restored.entries[0].coverage_pct == 100.0
    assert restored.entries[1].coverage == "none"
    assert restored.errors == {"nmt|kron86": "capabilities timeout"}
    assert restored.full_coverage_options[0]["year"] == 2024
