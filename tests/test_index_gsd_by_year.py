from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.models import (
    IndexManifest,
    YearAvailabilityReport,
    YearGsdSummary,
)


def test_index_manifest_accepts_gsd_by_year():
    m = IndexManifest(
        year_start=2024,
        year_end=2024,
        bbox="0,0,1,1",
        srs="EPSG:2180",
        years_requested=[2024],
        year_statuses=[],
        years_available_wfs=[2024],
        years_included=[2024],
        passed=True,
        gsd_by_year={2024: YearGsdSummary(histogram={"0.05": 3}, finest=0.05, coarsest=0.05)},
    )
    assert m.gsd_by_year[2024].finest == 0.05


def test_year_report_defaults_empty_gsd_by_year():
    r = YearAvailabilityReport(
        year_start=2024,
        year_end=2024,
        bbox="0,0,1,1",
        srs="EPSG:2180",
        years_requested=[2024],
        year_statuses=[],
        years_available_wfs=[2024],
        years_included=[2024],
        passed=True,
    )
    assert r.gsd_by_year == {}
