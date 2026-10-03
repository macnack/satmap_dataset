from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.config import RunConfig
from satmap_dataset.fingerprint import fingerprint_provider_options
from satmap_dataset.models import IndexManifest, LayerManifest, YearStatus
from satmap_dataset.pipeline import rgb_pipeline, run_all

_EMPTY_OPTS_FP = fingerprint_provider_options({})


def _base_manifest(**overrides) -> IndexManifest:
    data = dict(
        year_start=2017,
        year_end=2017,
        bbox="348760.243,508296.603,350174.457,509710.817",
        srs="EPSG:2180",
        strict_years=False,
        min_years=1,
        wfs_bbox_axes_swapped=False,
        years_requested=[2017],
        year_statuses=[
            YearStatus(year=2017, typename_exists=True, feature_count=1, status="has_features")
        ],
        years_available_wfs=[2017],
        years_included=[2017],
        years_excluded_with_reason={},
        common_tile_ids=["N-33-130-D-a-3-3"],
        tile_sources_by_year={2017: {"N-33-130-D-a-3-3": "https://example.com/a.tif"}},
        tile_bboxes_by_year={
            2017: {"N-33-130-D-a-3-3": [348760.243, 508296.603, 350174.457, 509710.817]}
        },
        passed=True,
        errors=[],
        warnings=[],
        run_parameters={"mode": "hybrid"},
        provider="geoportal",
        provider_options_fingerprint=_EMPTY_OPTS_FP,
    )
    data.update(overrides)
    return IndexManifest(**data)


def test_can_reuse_index_rejects_manifest_with_swapped_tile_bboxes() -> None:
    config = RunConfig(
        year_start=2017,
        year_end=2017,
        bbox="348760.243,508296.603,350174.457,509710.817",
        srs="EPSG:2180",
        mode="hybrid",
    )
    manifest = _base_manifest(
        # Swapped order from legacy bug: ymin,xmin,ymax,xmax
        tile_bboxes_by_year={
            2017: {"N-33-130-D-a-3-3": [507954.8, 347030.43, 510336.71, 349225.86]}
        },
        run_parameters={},
    )

    assert run_all._index_manifest_has_swapped_tile_bboxes(manifest) is True
    assert run_all._can_reuse_index(manifest, config) is False


def test_can_reuse_index_rejects_wms_stub_for_hybrid() -> None:
    config = RunConfig(
        year_start=2017,
        year_end=2017,
        bbox="348760.243,508296.603,350174.457,509710.817",
        srs="EPSG:2180",
        mode="hybrid",
    )
    stub = _base_manifest(
        years_available_wfs=[],
        warnings=[rgb_pipeline._WMS_ONLY_INDEX_WARNING],
        run_parameters={"mode": "wms_tiled"},
    )
    assert rgb_pipeline._can_reuse_index(stub, config) is False


def test_can_reuse_index_accepts_wms_stub_for_wms_tiled() -> None:
    config = RunConfig(
        year_start=2017,
        year_end=2017,
        bbox="348760.243,508296.603,350174.457,509710.817",
        srs="EPSG:2180",
        mode="wms_tiled",
    )
    stub = _base_manifest(
        years_available_wfs=[],
        warnings=[rgb_pipeline._WMS_ONLY_INDEX_WARNING],
        run_parameters={"mode": "wms_tiled"},
    )
    assert rgb_pipeline._can_reuse_index(stub, config) is True


def test_can_reuse_index_rejects_missing_provider_options_fingerprint() -> None:
    config = RunConfig(
        year_start=2017,
        year_end=2017,
        bbox="348760.243,508296.603,350174.457,509710.817",
        srs="EPSG:2180",
        mode="hybrid",
    )
    manifest = _base_manifest(provider_options_fingerprint=None)
    assert rgb_pipeline._can_reuse_index(manifest, config) is False


def test_can_reuse_index_rejects_mismatched_provider_options_fingerprint() -> None:
    config = RunConfig(
        year_start=2017,
        year_end=2017,
        bbox="348760.243,508296.603,350174.457,509710.817",
        srs="EPSG:2180",
        mode="hybrid",
        provider_options={"year_policy": "nearest_before"},
    )
    manifest = _base_manifest(provider_options_fingerprint=_EMPTY_OPTS_FP)
    assert rgb_pipeline._can_reuse_index(manifest, config) is False


def test_can_reuse_index_accepts_matching_provider_options_fingerprint() -> None:
    options = {"stac_url": "https://example.org/stac/search", "year_policy": "exact_only"}
    config = RunConfig(
        year_start=2017,
        year_end=2017,
        bbox="348760.243,508296.603,350174.457,509710.817",
        srs="EPSG:2180",
        mode="hybrid",
        provider="lantmateriet",
        provider_options=options,
    )
    manifest = _base_manifest(
        provider="lantmateriet",
        provider_options_fingerprint=fingerprint_provider_options(options),
    )
    assert rgb_pipeline._can_reuse_index(manifest, config) is True


def _base_download_manifest(tmp_path: Path, **overrides) -> tuple[LayerManifest, Path, Path]:
    asset = tmp_path / "tile.tif"
    asset.write_bytes(b"tiff")
    index_output = tmp_path / "index_manifest.json"
    index_output.write_text("{}", encoding="utf-8")
    download_output = tmp_path / "dataset_manifest_download.json"
    data = dict(
        layer="geoportal_rgb",
        role="rgb",
        stage="download",
        provider="geoportal",
        passed=True,
        assets=[str(asset)],
        source_manifest=str(index_output),
        mode="hybrid",
        profile="train",
        forced_wms_years=[],
        provider_options_fingerprint=_EMPTY_OPTS_FP,
    )
    data.update(overrides)
    return LayerManifest(**data), index_output, download_output


def test_can_reuse_download_rejects_missing_provider_options_fingerprint(tmp_path: Path) -> None:
    config = RunConfig(
        year_start=2017,
        year_end=2017,
        bbox="348760.243,508296.603,350174.457,509710.817",
        srs="EPSG:2180",
        mode="hybrid",
    )
    manifest, index_output, download_output = _base_download_manifest(
        tmp_path, provider_options_fingerprint=None
    )
    assert (
        rgb_pipeline._can_reuse_download(manifest, config, index_output, download_output)
        is False
    )


def test_can_reuse_download_rejects_mismatched_provider_options_fingerprint(
    tmp_path: Path,
) -> None:
    config = RunConfig(
        year_start=2017,
        year_end=2017,
        bbox="348760.243,508296.603,350174.457,509710.817",
        srs="EPSG:2180",
        mode="hybrid",
        provider_options={"max_cloud_cover_pct": 1.0},
    )
    manifest, index_output, download_output = _base_download_manifest(tmp_path)
    assert (
        rgb_pipeline._can_reuse_download(manifest, config, index_output, download_output)
        is False
    )


def test_can_reuse_download_accepts_matching_provider_options_fingerprint(
    tmp_path: Path,
) -> None:
    options = {"max_cloud_cover_pct": 1.0}
    config = RunConfig(
        year_start=2017,
        year_end=2017,
        bbox="348760.243,508296.603,350174.457,509710.817",
        srs="EPSG:2180",
        mode="hybrid",
        provider_options=options,
    )
    manifest, index_output, download_output = _base_download_manifest(
        tmp_path,
        provider_options_fingerprint=fingerprint_provider_options(options),
    )
    assert (
        rgb_pipeline._can_reuse_download(manifest, config, index_output, download_output)
        is True
    )
