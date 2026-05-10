from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.providers.sentinel2.year_policy import (
    CandidateItem,
    select_year,
    select_years,
)


def _c(item_id: str, datetime_iso: str, cloud_cover: float | None) -> CandidateItem:
    return CandidateItem(item_id=item_id, datetime_iso=datetime_iso, cloud_cover=cloud_cover)


def test_picks_closest_to_target_doy_in_year() -> None:
    candidates = [
        _c("late_jan", "2024-01-30T10:00:00Z", 5.0),
        _c("mid_feb", "2024-02-14T10:00:00Z", 4.0),  # 1 day away
        _c("early_mar", "2024-03-05T10:00:00Z", 3.0),
    ]
    pick = select_year(2024, candidates)
    assert pick.chosen.item_id == "mid_feb"
    assert pick.delta_days == 1


def test_filters_high_cloud_cover() -> None:
    candidates = [
        _c("clear_jan", "2024-01-15T10:00:00Z", 30.0),  # >20%, dropped
        _c("foggy_feb15", "2024-02-15T10:00:00Z", 80.0),  # >20%, dropped
        _c("clear_mar", "2024-03-10T10:00:00Z", 10.0),
    ]
    pick = select_year(2024, candidates, max_cloud_cover_pct=20.0)
    assert pick.chosen.item_id == "clear_mar"


def test_returns_none_when_no_candidate_passes_filter() -> None:
    candidates = [
        _c("foggy_jan", "2024-01-10T10:00:00Z", 99.0),
    ]
    pick = select_year(2024, candidates, max_cloud_cover_pct=20.0)
    assert pick.chosen is None
    assert "no_item_within_cloud_cover" in pick.reason


def test_ignores_items_from_other_years() -> None:
    candidates = [
        _c("2023_winter", "2023-02-15T10:00:00Z", 1.0),
        _c("2025_winter", "2025-02-15T10:00:00Z", 1.0),
    ]
    pick = select_year(2024, candidates)
    assert pick.chosen is None


def test_select_years_picks_one_per_year() -> None:
    candidates = [
        _c("2022a", "2022-02-13T10:00:00Z", 4.5),
        _c("2022b", "2022-02-25T10:00:00Z", 60.0),  # cloudy, dropped
        _c("2023a", "2023-02-12T10:00:00Z", 8.0),
        _c("2024a", "2024-01-18T10:00:00Z", 1.0),
        _c("2024b", "2024-02-17T10:00:00Z", 12.5),
    ]
    picks = select_years([2022, 2023, 2024], candidates)
    assert picks[2022].chosen.item_id == "2022a"
    assert picks[2023].chosen.item_id == "2023a"
    # 2024-02-17 is 2 days from Feb 15; 2024-01-18 is 28 days. Closer wins.
    assert picks[2024].chosen.item_id == "2024b"


def test_target_doy_override() -> None:
    candidates = [
        _c("july", "2024-07-15T10:00:00Z", 1.0),
        _c("january", "2024-01-15T10:00:00Z", 1.0),
    ]
    pick = select_year(2024, candidates, target_month=7, target_day=15)
    assert pick.chosen.item_id == "july"


def test_max_cloud_cover_none_disables_filter() -> None:
    candidates = [
        _c("foggy_feb15", "2024-02-15T10:00:00Z", 95.0),
    ]
    pick = select_year(2024, candidates, max_cloud_cover_pct=None)
    assert pick.chosen.item_id == "foggy_feb15"
