from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.providers.lantmateriet.year_policy import select_year


def test_exact_only_returns_match_when_year_present() -> None:
    match = select_year(2018, [2015, 2018, 2020], policy="exact_only")
    assert match.matched_year == 2018
    assert match.delta == 0


def test_exact_only_returns_none_when_year_absent() -> None:
    match = select_year(2018, [2015, 2017, 2020], policy="exact_only")
    assert match.matched_year is None
    assert match.delta is None
    assert "exact_only" in match.reason


def test_nearest_before_picks_closest_earlier_year() -> None:
    match = select_year(2020, [2015, 2017, 2022], policy="nearest_before")
    assert match.matched_year == 2017
    assert match.delta == 3


def test_nearest_before_returns_none_when_no_earlier_year() -> None:
    match = select_year(2010, [2015, 2017], policy="nearest_before")
    assert match.matched_year is None


def test_nearest_after_picks_closest_later_year() -> None:
    match = select_year(2017, [2015, 2020, 2024], policy="nearest_after")
    assert match.matched_year == 2020
    assert match.delta == 3


def test_nearest_any_with_max_delta_respects_window() -> None:
    match = select_year(
        2020,
        [2015, 2018, 2022],
        policy="nearest_any_with_max_delta",
        max_delta=2,
    )
    assert match.matched_year == 2018
    assert match.delta == 2


def test_nearest_any_with_max_delta_returns_none_outside_window() -> None:
    match = select_year(
        2020,
        [2015, 2018, 2025],
        policy="nearest_any_with_max_delta",
        max_delta=1,
    )
    assert match.matched_year is None
    assert "max_delta" in match.reason


def test_nearest_any_with_max_delta_requires_max_delta() -> None:
    with pytest.raises(ValueError):
        select_year(2020, [2018], policy="nearest_any_with_max_delta", max_delta=None)


def test_unknown_policy_raises_value_error() -> None:
    with pytest.raises(ValueError):
        select_year(2020, [2018], policy="latest")


def test_no_available_years_returns_none() -> None:
    match = select_year(2020, [], policy="exact_only")
    assert match.matched_year is None
    assert match.reason == "no_available_years"
