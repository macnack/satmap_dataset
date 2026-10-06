"""Per-year scene selection for the NAIP provider.

NAIP revisits CONUS on a ~2–3 year cycle (some states yearly). Items carry
``naip:year`` which is the collection year and can differ from the datetime
year when flights slip into the following calendar year. We bucket by
``acquisition_year`` (preferring ``naip:year``) and pick the candidate closest
to a target day-of-year (default June 15 — leaf-on).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, Sequence


@dataclass(frozen=True)
class CandidateItem:
    item_id: str
    datetime_iso: str
    acquisition_year: int


@dataclass(frozen=True)
class YearPick:
    requested_year: int
    chosen: CandidateItem | None
    delta_days: int | None
    reason: str


def _parse_dt(value: str) -> datetime | None:
    text = (value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _delta_days(item_dt: datetime, target: date) -> int:
    return abs((item_dt.date() - target).days)


def select_year(
    requested_year: int,
    candidates: Iterable[CandidateItem],
    *,
    target_month: int = 6,
    target_day: int = 15,
) -> YearPick:
    target = date(requested_year, target_month, target_day)
    in_year: list[tuple[int, CandidateItem]] = []
    for item in candidates:
        if item.acquisition_year != requested_year:
            continue
        dt = _parse_dt(item.datetime_iso)
        if dt is None:
            continue
        in_year.append((_delta_days(dt, target), item))

    if not in_year:
        return YearPick(requested_year, None, None, "no_item_for_year")
    in_year.sort(key=lambda pair: (pair[0], pair[1].item_id))
    delta, chosen = in_year[0]
    return YearPick(requested_year, chosen, delta, "closest_to_target_doy")


def select_years(
    requested_years: Sequence[int],
    candidates: Iterable[CandidateItem],
    *,
    target_month: int = 6,
    target_day: int = 15,
) -> dict[int, YearPick]:
    cached = list(candidates)
    return {
        year: select_year(
            year,
            cached,
            target_month=target_month,
            target_day=target_day,
        )
        for year in requested_years
    }
