from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from satmap_dataset.config import IndexConfig
from satmap_dataset.models import IndexManifest
from satmap_dataset.pipeline import index_builder
from _wfs_test_helpers import mock_statuses_2015_2026, mock_tiles_by_year_2015_2026


def test_manifest_contains_exclusion_reasons(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        index_builder,
        "probe_years_wfs_with_tiles",
        lambda *args, **kwargs: (mock_statuses_2015_2026(), mock_tiles_by_year_2015_2026()),
    )
    config = IndexConfig(
        year_start=2015,
        year_end=2026,
        bbox="210300,521900,210500,522100",
        srs="EPSG:2180",
        strict_years=False,
        min_years=2,
        output_json=tmp_path / "index_manifest.json",
        year_availability_output_json=tmp_path / "year_availability_report.json",
    )

    _, output_path = index_builder.run(config)
    manifest = IndexManifest.model_validate_json(output_path.read_text(encoding="utf-8"))

    for year in [2016, 2020, 2022, 2024, 2025, 2026]:
        assert year in manifest.years_excluded_with_reason
        assert manifest.years_excluded_with_reason[year]
