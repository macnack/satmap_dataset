from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.pipeline.validator import evaluate_year_policy


def test_policy_non_strict_passes_when_min_years_met() -> None:
    result = evaluate_year_policy(
        requested_years=[2015, 2016, 2017],
        available_years=[2015, 2017],
        strict_years=False,
        min_years=2,
    )

    assert result.passed is True
    assert result.missing_years == [2016]
    assert result.errors == []
    assert result.warnings


def test_policy_strict_fails_on_missing_years() -> None:
    result = evaluate_year_policy(
        requested_years=[2015, 2016, 2017],
        available_years=[2015, 2017],
        strict_years=True,
        min_years=2,
    )

    assert result.passed is False
    assert result.missing_years == [2016]
    assert any("strict_years=True" in message for message in result.errors)


def test_policy_min_years_failure() -> None:
    result = evaluate_year_policy(
        requested_years=[2015, 2016, 2017],
        available_years=[2015],
        strict_years=False,
        min_years=2,
    )

    assert result.passed is False
    assert any("min_years" in message for message in result.errors)
