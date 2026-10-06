from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.providers.naip.year_policy import CandidateItem, select_year, select_years


def test_select_year_picks_closest_to_june_15() -> None:
    candidates = [
        CandidateItem("late", "2023-09-10T16:00:00Z", 2023),
        CandidateItem("near", "2023-05-25T16:00:00Z", 2023),
    ]
    pick = select_year(2023, candidates, target_month=6, target_day=15)
    assert pick.chosen is not None
    assert pick.chosen.item_id == "near"


def test_select_year_buckets_by_naip_year_not_datetime_year() -> None:
    # Collection year 2021 with a datetime that slips into 2022.
    candidates = [
        CandidateItem("slip", "2022-01-05T00:00:00Z", 2021),
    ]
    pick = select_year(2021, candidates)
    assert pick.chosen is not None
    assert pick.chosen.item_id == "slip"
    assert select_year(2022, candidates).chosen is None


def test_select_years_covers_requested_range() -> None:
    candidates = [
        CandidateItem("a", "2018-06-20T00:00:00Z", 2018),
        CandidateItem("b", "2021-06-17T00:00:00Z", 2021),
    ]
    picks = select_years([2018, 2019, 2021], candidates)
    assert picks[2018].chosen is not None
    assert picks[2019].chosen is None
    assert picks[2021].chosen is not None
