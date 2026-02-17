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


def test_strict_years_mode(monkeypatch, tmp_path: Path) -> None:
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
        strict_years=True,
        min_years=2,
        output_json=tmp_path / "index_manifest.json",
        year_availability_output_json=tmp_path / "year_availability_report.json",
    )

    exit_code, output_path = index_builder.run(config)
    manifest = IndexManifest.model_validate_json(output_path.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert manifest.passed is False
    assert any("strict_years=True" in error for error in manifest.errors)
