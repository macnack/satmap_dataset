import json
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


def test_load_mars_lvig_gps_json(tmp_path: Path):
    p = tmp_path / "gps.json"
    p.write_text(
        json.dumps(
            {
                "items": [
                    {"latitude": 22.416, "longitude": 114.043, "altitude": 30.0},
                    {"latitude": 22.417, "longitude": 114.044, "altitude": 31.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    pts = load_track(p)
    assert pts == [TrackPoint(22.416, 114.043), TrackPoint(22.417, 114.044)]


def test_load_dir_finds_receiver_gps_json(tmp_path: Path):
    recv = tmp_path / "HKairport01" / "receiver"
    recv.mkdir(parents=True)
    (recv / "gps.json").write_text(
        json.dumps({"items": [{"latitude": 22.4, "longitude": 114.0}]}),
        encoding="utf-8",
    )
    pts = load_track(tmp_path / "HKairport01")
    assert pts == [TrackPoint(22.4, 114.0)]


def test_load_gps_json_list_root(tmp_path: Path):
    p = tmp_path / "track.json"
    p.write_text(
        json.dumps([{"lat": 1.0, "lon": 2.0}, {"latitude": 3.0, "longitude": 4.0}]),
        encoding="utf-8",
    )
    pts = load_track(p)
    assert pts == [TrackPoint(1.0, 2.0), TrackPoint(3.0, 4.0)]
