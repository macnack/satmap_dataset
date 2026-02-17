from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.config import RenderConfig
from satmap_dataset.models import DatasetManifest
from satmap_dataset.pipeline import render


def _write_geotiff_with_georef(path: Path, *, tie_x: float, tie_y: float, px: float, py: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((100, 100, 3), 128, dtype=np.uint8)
    tifffile.imwrite(
        path,
        data,
        photometric="rgb",
        compression="deflate",
        metadata=None,
        extratags=[
            (33550, "d", 3, (px, py, 0.0), False),
            (33922, "d", 6, (0.0, 0.0, 0.0, tie_x, tie_y, 0.0), False),
        ],
    )


def test_infer_global_wfs_axis_mode_uses_all_wfs_years(tmp_path: Path) -> None:
    # Target grid is EPSG:2180 bbox [x=359700..361700, y=504900..506900].
    target_bbox = render.BBox(min_x=359700.0, min_y=504900.0, max_x=361700.0, max_y=506900.0)
    years_source_map = {2022: "wfs", 2023: "wfs"}

    # Year 2022 has no assets; year 2023 has swapped-axis georef.
    swapped_tile = tmp_path / "downloads" / "2023" / "tile_2023.tif"
    _write_geotiff_with_georef(
        swapped_tile,
        tie_x=504900.0,  # normal X carries target Y range start
        tie_y=361700.0,  # normal Y carries target X range end
        px=20.0,
        py=20.0,
    )
    grouped_assets = {2022: [], 2023: [swapped_tile]}

    mode = render._infer_global_wfs_axis_mode(
        grouped_assets=grouped_assets,
        years_source_map=years_source_map,
        target_bbox=target_bbox,
        reference_year=2022,
    )
    assert mode == "swapped"


def test_render_run_applies_inferred_swapped_axis_mode_without_calibration(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source_tile = tmp_path / "downloads" / "2023" / "tile_2023.tif"
    _write_geotiff_with_georef(
        source_tile,
        tie_x=504900.0,
        tie_y=361700.0,
        px=20.0,
        py=20.0,
    )

    source_manifest = DatasetManifest(
        stage="download",
        years_requested=[2023],
        years_available_wfs=[2023],
        years_included=[2023],
        years_excluded_with_reason={},
        common_tile_ids=[],
        tile_sources_by_year={},
        assets=[str(source_tile)],
        mode="wfs_render",
        target_bbox="359700,504900,361700,506900",
        target_srs="EPSG:2180",
        profile="reference",
        px_per_meter=0.001,
        years_source_map={2023: "wfs"},
        passed=True,
    )
    dataset_manifest_path = tmp_path / "dataset_manifest_download.json"
    dataset_manifest_path.write_text(source_manifest.model_dump_json(indent=2), encoding="utf-8")

    captured: dict[str, str] = {}

    def fake_render_year(*, source_axis_mode, out_path, target_bbox, target_width, target_height, target_srs, **kwargs):
        captured["source_axis_mode"] = source_axis_mode
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(out_path, np.full((target_height, target_width, 3), 64, dtype=np.uint8), photometric="rgb")
        render._ensure_georeferenced_output(
            out_path=out_path,
            target_bbox=target_bbox,
            target_width=target_width,
            target_height=target_height,
            srs=target_srs,
        )
        return 1.0, [64.0, 64.0, 64.0]

    monkeypatch.setattr(render, "_render_year", fake_render_year)

    config = RenderConfig(
        dataset_manifest=dataset_manifest_path,
        render_root=tmp_path / "rendered",
        mode="wfs_render",
        profile="reference",
        px_per_meter=0.001,  # 2000m AOI -> 2px output, tiny/faster test artifact.
        target_srs="EPSG:2180",
        wfs_global_calibration=False,
        output_json=tmp_path / "dataset_manifest_render.json",
    )

    exit_code, out_manifest_path = render.run(config)
    assert exit_code == 0
    assert captured["source_axis_mode"] == "swapped"

    out_manifest = DatasetManifest.model_validate_json(out_manifest_path.read_text(encoding="utf-8"))
    assert out_manifest.calibration_source_axis_mode == "swapped"
    assert out_manifest.passed is True
