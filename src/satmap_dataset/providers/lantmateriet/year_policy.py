from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

YearPolicyName = Literal[
    "exact_only",
    "nearest_before",
    "nearest_after",
    "nearest_any_with_max_delta",
]

ALLOWED_POLICIES: frozenset[str] = frozenset(
    {"exact_only", "nearest_before", "nearest_after", "nearest_any_with_max_delta"}
)


@dataclass(frozen=True)
class YearMatch:
    requested_year: int
    matched_year: int | None
    delta: int | None
    reason: str


def _validate_policy(policy: str) -> None:
    if policy not in ALLOWED_POLICIES:
        raise ValueError(
            f"Unknown year policy {policy!r}. Expected one of {sorted(ALLOWED_POLICIES)}."
        )


def select_year(
    requested_year: int,
    available_years: Iterable[int],
    *,
    policy: str = "exact_only",
    max_delta: int | None = None,
) -> YearMatch:
    """Pick a matched_year for `requested_year` from `available_years` per policy.

    `max_delta` is required for `nearest_any_with_max_delta` and ignored otherwise
    (use `nearest_before`/`nearest_after` for unbounded directional fallback).
    """
    _validate_policy(policy)
    available_sorted = sorted(set(int(y) for y in available_years))
    if not available_sorted:
        return YearMatch(requested_year, None, None, "no_available_years")

    if requested_year in available_sorted:
        return YearMatch(requested_year, requested_year, 0, "exact_match")

    if policy == "exact_only":
        return YearMatch(requested_year, None, None, "exact_only_policy_no_match")

    earlier = [y for y in available_sorted if y < requested_year]
    later = [y for y in available_sorted if y > requested_year]

    if policy == "nearest_before":
        if not earlier:
            return YearMatch(requested_year, None, None, "no_earlier_year")
        chosen = earlier[-1]
        return YearMatch(requested_year, chosen, requested_year - chosen, "nearest_before")

    if policy == "nearest_after":
        if not later:
            return YearMatch(requested_year, None, None, "no_later_year")
        chosen = later[0]
        return YearMatch(requested_year, chosen, chosen - requested_year, "nearest_after")

    # nearest_any_with_max_delta
    if max_delta is None or max_delta < 0:
        raise ValueError("nearest_any_with_max_delta requires max_delta >= 0")
    candidates = [
        (abs(y - requested_year), y) for y in available_sorted if abs(y - requested_year) <= max_delta
    ]
    if not candidates:
        return YearMatch(
            requested_year,
            None,
            None,
            f"no_year_within_max_delta_{max_delta}",
        )
    candidates.sort(key=lambda pair: (pair[0], abs(pair[1] - requested_year), pair[1]))
    delta, chosen = candidates[0]
    return YearMatch(requested_year, chosen, delta, "nearest_any_with_max_delta")
