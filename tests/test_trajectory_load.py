from pathlib import Path

import pytest

from satmap_dataset.trajectory import TrackPoint, load_track

SAMPLE_IGC = (
    "ANAV9A1\n"
    "HFDTEDATE:190426,01\n"
    "B1039295142136N01750376EA0011800175\n"  # 51.70227, 17.83960
    "B1039305142200N01750400EA0011800175\n"
    "Lsomethingelse\n"
)


def test_load_igc_parses_b_records(tmp_path: Path):
    p = tmp_path / "track.igc"
    p.write_text(SAMPLE_IGC, encoding="latin-1")
    pts = load_track(p)
    assert len(pts) == 2
    assert pts[0].lat == pytest.approx(51.70227, abs=1e-4)
    assert pts[0].lon == pytest.approx(17.83960, abs=1e-4)


def test_load_igc_southwest_hemisphere(tmp_path: Path):
    p = tmp_path / "s.igc"
    p.write_text("B1039295142136S01750376WA000\n", encoding="latin-1")
    pts = load_track(p)
    assert pts[0].lat < 0 and pts[0].lon < 0


def test_load_dir_autodetects_single_igc(tmp_path: Path):
    (tmp_path / "track.igc").write_text(SAMPLE_IGC, encoding="latin-1")
    pts = load_track(tmp_path)
    assert len(pts) == 2


def test_load_dir_rejects_multiple_igc(tmp_path: Path):
    (tmp_path / "a.igc").write_text(SAMPLE_IGC, encoding="latin-1")
    (tmp_path / "b.igc").write_text(SAMPLE_IGC, encoding="latin-1")
    with pytest.raises(ValueError):
        load_track(tmp_path)


def test_load_csv_lat_lon(tmp_path: Path):
    p = tmp_path / "t.csv"
    p.write_text("lat,lon\n51.5,17.8\n51.6,17.9\n", encoding="utf-8")
    pts = load_track(p)
    assert pts == [TrackPoint(51.5, 17.8), TrackPoint(51.6, 17.9)]


def test_load_csv_latitude_longitude_aliases(tmp_path: Path):
    p = tmp_path / "t.csv"
    p.write_text("time,Latitude,Longitude\n1,51.5,17.8\n", encoding="utf-8")
    pts = load_track(p)
    assert pts == [TrackPoint(51.5, 17.8)]


def test_load_csv_missing_columns_raises(tmp_path: Path):
    p = tmp_path / "t.csv"
    p.write_text("x,y\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_track(p)


def test_load_empty_track_raises(tmp_path: Path):
    p = tmp_path / "t.csv"
    p.write_text("lat,lon\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_track(p)
