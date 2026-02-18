from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.config import DownloadConfig
from satmap_dataset.models import DatasetManifest, IndexManifest, YearStatus
from satmap_dataset.pipeline import downloader


def _write_index_manifest(
    path: Path,
    *,
    bbox: str = "0,0,1,1",
    years_requested: list[int],
    years_included: list[int],
    tile_sources_by_year: dict[int, dict[str, str]],
    tile_bboxes_by_year: dict[int, dict[str, list[float]]] | None = None,
) -> None:
    statuses: list[YearStatus] = []
    included = set(years_included)
    for year in years_requested:
        if year in included:
            statuses.append(
                YearStatus(
                    year=year,
                    typename_exists=True,
                    feature_count=max(1, len(tile_sources_by_year.get(year, {}))),
                    status="has_features",
                    reason=None,
                )
            )
        else:
            statuses.append(
                YearStatus(
                    year=year,
                    typename_exists=True,
                    feature_count=0,
                    status="zero_features",
                    reason="Typename exists but no features found for AOI.",
                )
            )

    manifest = IndexManifest(
        year_start=min(years_requested),
        year_end=max(years_requested),
        bbox=bbox,
        srs="EPSG:2180",
        strict_years=False,
        min_years=1,
        years_requested=years_requested,
        year_statuses=statuses,
        years_available_wfs=years_included,
        years_included=years_included,
        years_excluded_with_reason={year: "Typename exists but no features found for AOI." for year in years_requested if year not in included},
        common_tile_ids=[],
        tile_sources_by_year=tile_sources_by_year,
        tile_bboxes_by_year=tile_bboxes_by_year or {},
        passed=True,
        errors=[],
        warnings=[],
    )
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def _write_rgb_tiff(path: Path, value: int) -> None:
    arr = np.full((8, 8, 3), value, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, arr, photometric="rgb", compression="deflate")


def _write_geotiff(path: Path, *, value: int, bbox: tuple[float, float, float, float]) -> None:
    xmin, ymin, xmax, ymax = bbox
    arr = np.full((8, 8, 3), value, dtype=np.uint8)
    px_x = (xmax - xmin) / 8.0
    px_y = (ymax - ymin) / 8.0
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        path,
        arr,
        photometric="rgb",
        compression="deflate",
        metadata=None,
        extratags=[
            (33550, "d", 3, (px_x, px_y, 0.0), False),
            (33922, "d", 6, (0.0, 0.0, 0.0, xmin, ymax, 0.0), False),
        ],
    )


def test_download_cache_reuse_and_corruption_redownload(monkeypatch, tmp_path: Path) -> None:
    index_path = tmp_path / "index_manifest.json"
    _write_index_manifest(
        index_path,
        years_requested=[2021],
        years_included=[2021],
        tile_sources_by_year={2021: {"tileA": "https://example.com/tileA.tif"}},
    )

    calls: list[Path] = []

    async def fake_download_with_retry(client, url, output_path, **kwargs):
        calls.append(output_path)
        _write_rgb_tiff(output_path, value=10)
        return True

    monkeypatch.setattr(downloader, "_download_with_retry", fake_download_with_retry)

    config = DownloadConfig(
        index_manifest=index_path,
        download_root=tmp_path / "downloads",
        mode="wfs_render",
        profile="train",
        output_json=tmp_path / "dataset_manifest_download.json",
    )

    code_1, out_1 = downloader.run(config)
    assert code_1 == 0
    assert len(calls) == 1

    code_2, out_2 = downloader.run(config)
    assert code_2 == 0
    assert out_1 == out_2
    assert len(calls) == 1

    cached_tile = tmp_path / "downloads" / "2021" / "tileA.tif"
    cached_tile.write_bytes(b"corrupt")

    code_3, _ = downloader.run(config)
    assert code_3 == 0
    assert len(calls) == 2


def test_hybrid_mode_uses_wms_fallback_for_missing_years(monkeypatch, tmp_path: Path) -> None:
    index_path = tmp_path / "index_manifest.json"
    _write_index_manifest(
        index_path,
        years_requested=[2020, 2021],
        years_included=[2021],
        tile_sources_by_year={2021: {"tileA": "https://example.com/tileA.tif"}},
    )

    async def fake_download_with_retry(client, url, output_path, **kwargs):
        _write_rgb_tiff(output_path, value=20)
        return True

    async def fake_download_wms_tile_with_retry(client, year, bbox, srs, width, height, output_path, **kwargs):
        _write_rgb_tiff(output_path, value=30)
        return True

    monkeypatch.setattr(downloader, "_download_with_retry", fake_download_with_retry)
    monkeypatch.setattr(downloader, "_download_wms_tile_with_retry", fake_download_wms_tile_with_retry)

    config = DownloadConfig(
        index_manifest=index_path,
        download_root=tmp_path / "downloads",
        mode="hybrid",
        profile="reference",
        bbox="0,0,1,1",
        px_per_meter=1.0,
        output_json=tmp_path / "dataset_manifest_download.json",
    )

    code, output = downloader.run(config)
    assert code == 0

    manifest = DatasetManifest.model_validate_json(output.read_text(encoding="utf-8"))
    assert manifest.passed is True
    assert manifest.years_included == [2020, 2021]
    assert manifest.years_source_map[2021] == "wfs"
    assert manifest.years_source_map[2020] == "wms_fallback"
    assert manifest.run_parameters["mode"] == "hybrid"
    assert manifest.run_parameters["bbox"] == "0,0,1,1"


def test_wfs_tile_georef_mismatch_is_rejected(monkeypatch, tmp_path: Path) -> None:
    index_path = tmp_path / "index_manifest.json"
    _write_index_manifest(
        index_path,
        bbox="359700,504900,361700,506900",
        years_requested=[2015, 2016],
        years_included=[2015],
        tile_sources_by_year={2015: {"tileA": "https://example.com/tileA.tif"}},
        tile_bboxes_by_year={2015: {"tileA": [359700.0, 504900.0, 361700.0, 506900.0]}},
    )

    async def fake_download_with_retry(client, url, output_path, **kwargs):
        # Write an unrelated georef footprint far away from expected bbox.
        _write_geotiff(output_path, value=10, bbox=(6500000.0, 5600000.0, 6500100.0, 5600100.0))
        return True

    async def fake_download_wms_tile_with_retry(client, year, bbox, srs, width, height, output_path, **kwargs):
        _write_rgb_tiff(output_path, value=30)
        return True

    monkeypatch.setattr(downloader, "_download_with_retry", fake_download_with_retry)
    monkeypatch.setattr(downloader, "_download_wms_tile_with_retry", fake_download_wms_tile_with_retry)

    config = DownloadConfig(
        index_manifest=index_path,
        download_root=tmp_path / "downloads",
        mode="hybrid",
        profile="reference",
        bbox="359700,504900,361700,506900",
        px_per_meter=1.0,
        output_json=tmp_path / "dataset_manifest_download.json",
    )

    code, output = downloader.run(config)
    manifest = DatasetManifest.model_validate_json(output.read_text(encoding="utf-8"))

    assert code == 1
    assert manifest.passed is False
    # WFS tile should be rejected due georef mismatch, so only fallback year remains.
    assert 2015 not in manifest.years_included
    assert manifest.years_source_map.get(2015) is None
    assert manifest.years_source_map.get(2016) == "wms_fallback"


def test_hybrid_force_wms_year_skips_wfs_for_that_year(monkeypatch, tmp_path: Path) -> None:
    index_path = tmp_path / "index_manifest.json"
    _write_index_manifest(
        index_path,
        bbox="359700,504900,361700,506900",
        years_requested=[2015, 2016],
        years_included=[2015],
        tile_sources_by_year={2015: {"tileA": "https://example.com/tileA.tif"}},
    )

    wfs_calls: list[Path] = []

    async def fake_download_with_retry(client, url, output_path, **kwargs):
        wfs_calls.append(output_path)
        _write_rgb_tiff(output_path, value=20)
        return True

    async def fake_download_wms_tile_with_retry(client, year, bbox, srs, width, height, output_path, **kwargs):
        _write_rgb_tiff(output_path, value=30)
        return True

    monkeypatch.setattr(downloader, "_download_with_retry", fake_download_with_retry)
    monkeypatch.setattr(downloader, "_download_wms_tile_with_retry", fake_download_wms_tile_with_retry)

    config = DownloadConfig(
        index_manifest=index_path,
        download_root=tmp_path / "downloads",
        mode="hybrid",
        profile="reference",
        bbox="359700,504900,361700,506900",
        px_per_meter=1.0,
        force_wms_years=[2015],
        output_json=tmp_path / "dataset_manifest_download.json",
    )

    code, output = downloader.run(config)
    manifest = DatasetManifest.model_validate_json(output.read_text(encoding="utf-8"))

    assert code == 0
    assert wfs_calls == []
    assert manifest.forced_wms_years == [2015]
    assert manifest.years_source_map[2015] == "wms_fallback"
    assert manifest.years_source_map[2016] == "wms_fallback"


def test_download_rejects_mismatched_wms_bbox_vs_index_bbox(tmp_path: Path) -> None:
    index_path = tmp_path / "index_manifest.json"
    _write_index_manifest(
        index_path,
        years_requested=[2021],
        years_included=[2021],
        tile_sources_by_year={2021: {"tileA": "https://example.com/tileA.tif"}},
    )

    config = DownloadConfig(
        index_manifest=index_path,
        download_root=tmp_path / "downloads",
        mode="hybrid",
        profile="reference",
        bbox="10,10,11,11",
        px_per_meter=1.0,
        output_json=tmp_path / "dataset_manifest_download.json",
    )

    with pytest.raises(ValueError, match="WMS bbox must match index bbox exactly"):
        downloader.run(config)
