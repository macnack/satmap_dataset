"""Unit tests for studio year-timeline status derivation."""

from __future__ import annotations

from pathlib import Path

from satmap_dataset.studio.timeline import (
    derive_year_timeline,
    load_json_mapping,
    timeline_html,
)


def test_derive_year_timeline_levels(tmp_path: Path) -> None:
    download_root = tmp_path / "downloads"
    (download_root / "2017").mkdir(parents=True)
    (download_root / "2017" / "tile.tif").write_bytes(b"x")
    render_root = tmp_path / "rendered"
    render_root.mkdir()
    (render_root / "year_2018.tif").write_bytes(b"x")

    cells = derive_year_timeline(
        year_start=2015,
        year_end=2018,
        index_payload={
            "years_available_wfs": [2016, 2017, 2018],
            "year_statuses": [
                {"year": 2015, "status": "zero_features", "feature_count": 0},
                {"year": 2016, "status": "has_features", "feature_count": 2},
            ],
        },
        download_payload={"years_included": [2017]},
        render_payload={"years_included": [2018], "assets": [str(render_root / "year_2018.tif")]},
        download_root=download_root,
        render_root=render_root,
    )
    by_year = {c.year: c for c in cells}
    assert by_year[2015].level == "missing"
    assert by_year[2016].level == "available"
    assert by_year[2017].level == "downloaded"
    assert by_year[2018].level == "rendered"
    assert by_year[2018].available is True


def test_derive_swaps_inverted_range() -> None:
    cells = derive_year_timeline(year_start=2020, year_end=2018)
    assert [c.year for c in cells] == [2018, 2019, 2020]
    assert all(c.level == "missing" for c in cells)


def test_timeline_html_contains_years() -> None:
    cells = derive_year_timeline(
        year_start=2019,
        year_end=2020,
        index_payload={"years_included": [2019]},
    )
    html = timeline_html(cells)
    assert "2019" in html
    assert "2020" in html
    assert "available" in html
    assert "missing" in html


def test_load_json_mapping(tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    path.write_text('{"years_included": [2021]}', encoding="utf-8")
    assert load_json_mapping(path) == {"years_included": [2021]}
    assert load_json_mapping(tmp_path / "nope.json") is None
