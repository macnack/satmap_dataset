from pathlib import Path

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


import pytest
from pydantic import ValidationError
from satmap_dataset.config import DemAvailabilityConfig


def test_availability_config_defaults_and_validation():
    cfg = DemAvailabilityConfig(bbox="0,0,10,10")
    assert cfg.products == ["nmt", "nmpt"]
    assert cfg.datums == ["evrf2007", "kron86"]
    assert cfg.year_start is None and cfg.year_end is None
    with pytest.raises(ValidationError):
        DemAvailabilityConfig(bbox="10,10,0,0")  # bad bbox order
    with pytest.raises(ValidationError):
        DemAvailabilityConfig(bbox="0,0,10,10", products=["lidar"])
    with pytest.raises(ValidationError):
        DemAvailabilityConfig(bbox="0,0,10,10", datums=["pl1965"])
    assert DemAvailabilityConfig(bbox="0,0,10,10", products=["NMT"]).products == ["nmt"]
    assert DemAvailabilityConfig(bbox="0,0,10,10", datums=["KRON86"]).datums == ["kron86"]


from satmap_dataset.pipeline import dem_availability as dav


def test_coverage_full_partial_none():
    aoi = (0.0, 0.0, 100.0, 100.0)
    assert dav._coverage_pct(aoi, [(0.0, 0.0, 100.0, 100.0)]) == 100.0
    half = dav._coverage_pct(aoi, [(0.0, 0.0, 50.0, 100.0)])
    assert 45.0 <= half <= 55.0
    assert dav._coverage_pct(aoi, []) == 0.0
    two = dav._coverage_pct(aoi, [(0.0, 0.0, 50.0, 100.0), (50.0, 0.0, 100.0, 100.0)])
    assert two == 100.0


def test_classify_coverage():
    assert dav._classify(100.0) == "full"
    assert dav._classify(99.95) == "full"
    assert dav._classify(60.0) == "partial"
    assert dav._classify(0.0) == "none"


def test_formats_from_urls():
    urls = [
        "https://x/a_M-1-1.asc",
        "https://x/b_M-1-2.xyz.zip",
        "https://x/c_M-1-3.zip",
        "https://x/d_M-1-4.xyz",
        "https://x/e_M-1-5.tif",
    ]
    assert dav._formats_from_urls(urls) == ["asc", "tif", "xyz", "xyz.zip", "zip"]


import json
from satmap_dataset.models import YearStatus


def _patch_av(monkeypatch, *, years_by_combo, tiles_for):
    async def _yt(product, datum, options=None, *, timeout=45.0, retry_policy=None):
        key = f"{product}|{datum}"
        if isinstance(years_by_combo.get(key), Exception):
            raise years_by_combo[key]
        return {y: f"gugik:Skorowidz{product.upper()}{y}" for y in years_by_combo.get(key, [])}

    async def _tf(product, datum, year, bbox, srs, *, year_to_typename, options=None, timeout=45.0, retry_policy=None):
        tiles, bboxes = tiles_for(product, datum, year)
        status = YearStatus(year=year, typename_exists=True, feature_count=len(tiles),
                            status="has_features" if tiles else "zero_features")
        acq = {tid: {"acquisition_date": f"{year}-03-01"} for tid in tiles}
        return status, dict(tiles), dict(bboxes), acq

    monkeypatch.setattr(dav.dem_skorowidz_client, "year_typenames", _yt)
    monkeypatch.setattr(dav.dem_skorowidz_client, "tiles_for_year", _tf)


def test_run_builds_report(tmp_path, monkeypatch):
    years = {
        "nmt|evrf2007": [2019],
        "nmpt|evrf2007": [2019, 2024],
        "nmt|kron86": RuntimeError("caps down"),
        "nmpt|kron86": [],
    }

    def tiles_for(product, datum, year):
        if product == "nmpt" and datum == "evrf2007" and year == 2024:
            return {"g1": "https://x/a_g1.asc"}, {"g1": [0.0, 0.0, 100.0, 100.0]}
        if product == "nmpt" and datum == "evrf2007" and year == 2019:
            return {"g2": "https://x/a_g2.asc"}, {"g2": [0.0, 0.0, 50.0, 100.0]}
        if product == "nmt" and datum == "evrf2007" and year == 2019:
            return {"g3": "https://x/a_g3.xyz.zip"}, {"g3": [0.0, 0.0, 100.0, 100.0]}
        return {}, {}

    _patch_av(monkeypatch, years_by_combo=years, tiles_for=tiles_for)
    from satmap_dataset.config import DemAvailabilityConfig
    cfg = DemAvailabilityConfig(bbox="0,0,100,100", output_json=tmp_path / "av.json")
    code, path = dav.run(cfg)
    assert code == 0
    report = DemAvailabilityReport.model_validate_json(Path(path).read_text())
    by = {(e.product, e.datum, e.year): e for e in report.entries}
    assert by[("nmpt", "evrf2007", 2024)].coverage == "full"
    assert by[("nmpt", "evrf2007", 2024)].formats == ["asc"]
    assert by[("nmpt", "evrf2007", 2019)].coverage == "partial"
    assert by[("nmt", "evrf2007", 2019)].formats == ["xyz.zip"]
    assert report.errors["nmt|kron86"].startswith("caps")
    assert {"product": "nmpt", "datum": "evrf2007", "year": 2024} in report.full_coverage_options
    assert by[("nmpt", "evrf2007", 2024)].acquisition_dates == ["2024-03-01"]
