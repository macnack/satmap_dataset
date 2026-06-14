"""Per-year scene selection for the Sentinel-2 provider.

Sentinel-2 revisits every ~5 days, so each requested year has many candidate
items rather than a single annual capture. We pick one representative per
year by minimising the distance from a target day-of-year (default Feb 15)
while filtering out items above a cloud-cover threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, Sequence


@dataclass(frozen=True)
class CandidateItem:
    """Lightweight view of a STAC item used by the year policy."""

    item_id: str
    datetime_iso: str
    cloud_cover: float | None  # eo:cloud_cover, percent 0..100


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
    target_month: int = 2,
    target_day: int = 15,
    max_cloud_cover_pct: float | None = 20.0,
) -> YearPick:
    """Pick the candidate closest to (year, target_month, target_day) that
    passes the cloud-cover threshold.

    `max_cloud_cover_pct=None` disables the filter.
    """
    target = date(requested_year, target_month, target_day)
    in_year: list[tuple[int, CandidateItem]] = []
    for item in candidates:
        dt = _parse_dt(item.datetime_iso)
        if dt is None or dt.year != requested_year:
            continue
        if max_cloud_cover_pct is not None and item.cloud_cover is not None:
            if item.cloud_cover > max_cloud_cover_pct:
                continue
        in_year.append((_delta_days(dt, target), item))

    if not in_year:
        return YearPick(
            requested_year,
            None,
            None,
            f"no_item_within_cloud_cover_{max_cloud_cover_pct}",
        )
    in_year.sort(key=lambda pair: (pair[0], pair[1].cloud_cover or 100.0, pair[1].item_id))
    delta, chosen = in_year[0]
    return YearPick(requested_year, chosen, delta, "closest_to_target_doy")


def select_years(
    requested_years: Sequence[int],
    candidates: Iterable[CandidateItem],
    *,
    target_month: int = 2,
    target_day: int = 15,
    max_cloud_cover_pct: float | None = 20.0,
) -> dict[int, YearPick]:
    cached: list[CandidateItem] = list(candidates)
    return {
        year: select_year(
            year,
            cached,
            target_month=target_month,
            target_day=target_day,
            max_cloud_cover_pct=max_cloud_cover_pct,
        )
        for year in requested_years
    }
