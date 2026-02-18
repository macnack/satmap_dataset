from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.config import IndexConfig
from satmap_dataset.models import IndexManifest, YearAvailabilityReport, YearStatus
from satmap_dataset.pipeline import index_builder


def test_index_writes_aoi_preview_html(monkeypatch, tmp_path: Path) -> None:
    def fake_probe(aoi: str, year_start: int, year_end: int, srs: str):
        statuses = [
            YearStatus(
                year=2023,
                typename_exists=True,
                feature_count=1,
                status="has_features",
                reason=None,
            )
        ]
        tiles = {2023: {"TILE_1": "https://example.com/TILE_1.tif"}}
        tile_bboxes = {2023: {"TILE_1": [359700.0, 504900.0, 361700.0, 506900.0]}}
        return statuses, tiles, tile_bboxes

    monkeypatch.setattr(index_builder, "probe_years_wfs_with_tiles", fake_probe)

    config = IndexConfig(
        year_start=2023,
        year_end=2023,
        bbox="359700,504900,361700,506900",
        srs="EPSG:2180",
        strict_years=False,
        min_years=1,
        output_json=tmp_path / "index_manifest.json",
        year_availability_output_json=tmp_path / "year_availability_report.json",
    )
    exit_code, output_path = index_builder.run(config)

    assert exit_code == 0
    preview_path = tmp_path / "aoi_preview.html"
    assert preview_path.exists()
    html = preview_path.read_text(encoding="utf-8")
    assert "OpenStreetMap contributors" in html
    assert "359700.0,504900.0,361700.0,506900.0" in html

    manifest = IndexManifest.model_validate_json(output_path.read_text(encoding="utf-8"))
    assert manifest.aoi_preview_html == str(preview_path)

    year_report = YearAvailabilityReport.model_validate_json(
        (tmp_path / "year_availability_report.json").read_text(encoding="utf-8")
    )
    assert year_report.aoi_preview_html == str(preview_path)
