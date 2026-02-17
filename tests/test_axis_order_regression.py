from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.config import IndexConfig
from satmap_dataset.models import IndexManifest, YearStatus
from satmap_dataset.pipeline import index_builder


def test_index_keeps_bbox_axis_order_when_legacy_swap_flag_is_set(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, str] = {}

    def fake_probe(aoi: str, year_start: int, year_end: int, srs: str):
        captured["aoi"] = aoi
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
        return statuses, tiles

    monkeypatch.setattr(index_builder, "probe_years_wfs_with_tiles", fake_probe)

    bbox = "359700,504900,361700,506900"
    config = IndexConfig(
        year_start=2023,
        year_end=2023,
        bbox=bbox,
        srs="EPSG:2180",
        strict_years=False,
        experimental_wfs_swap_bbox_axes=True,
        min_years=1,
        output_json=tmp_path / "index_manifest.json",
        year_availability_output_json=tmp_path / "year_availability_report.json",
    )

    exit_code, output_path = index_builder.run(config)
    manifest = IndexManifest.model_validate_json(output_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert captured["aoi"] == bbox
    assert any("strictly as xmin,ymin,xmax,ymax" in warning for warning in manifest.warnings)
