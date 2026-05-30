from __future__ import annotations

import json
from pathlib import Path

import pytest

from satmap_dataset.config import RunConfig
from satmap_dataset.layers import get_layer
from satmap_dataset.layers.rgb import RgbLayer
from satmap_dataset.models import (
    LayerManifest,
    TileAcquisitionMetadata,
)


def _run_config(tmp_path: Path) -> RunConfig:
    return RunConfig(
        year_start=2023,
        year_end=2024,
        bbox="210300,521900,210500,522100",
        srs="EPSG:2180",
        min_years=1,
        profile="reference",
        provider="geoportal",
        download_root=tmp_path / "downloads",
        render_root=tmp_path / "rendered",
        artifacts_dir=tmp_path / "artifacts",
    )


def _stub_rgb_pipeline(monkeypatch, tmp_path: Path) -> Path:
    """Patch the shared RGB core so produce() exercises mapping, not real IO."""
    render_path = tmp_path / "artifacts" / "dataset_manifest_render.json"

    def fake_run_rgb(config):
        render_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = LayerManifest(
            layer="geoportal_rgb",
            role="rgb",
            stage="render",
            provider="geoportal",
            years_requested=[2023, 2024],
            years_included=[2023, 2024],
            assets=[
                str(tmp_path / "rendered" / "year_2023.tiff"),
                str(tmp_path / "rendered" / "year_2024.tiff"),
            ],
            target_bbox="210300,521900,210500,522100",
            target_width=3000,
            target_height=3000,
            target_srs="EPSG:2180",
            px_per_meter=15.0,
            years_source_map={2023: "wfs", 2024: "wms"},
            tile_acquisition_by_year={
                2023: {"t1": TileAcquisitionMetadata(acquisition_date="2023-07-15")},
                2024: {"t2": TileAcquisitionMetadata(acquisition_date="2024-06-01")},
            },
            passed=True,
        )
        render_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return 0, render_path

    monkeypatch.setattr("satmap_dataset.layers.rgb._run_rgb_pipeline", fake_run_rgb)
    return render_path


def test_registry_resolves_rgb_layer():
    layer = get_layer("geoportal_rgb")
    assert isinstance(layer, RgbLayer)
    assert layer.role == "rgb"
    assert layer.defines_grid is True
    assert layer.provider_name == "geoportal"


def test_registry_unknown_raises():
    with pytest.raises(ValueError):
        get_layer("does_not_exist")


def test_rgb_layer_produce_builds_layer_manifest(monkeypatch, tmp_path: Path):
    _stub_rgb_pipeline(monkeypatch, tmp_path)
    layer = get_layer("geoportal_rgb")
    code, manifest = layer.produce(_run_config(tmp_path), grid=None)

    assert code == 0
    assert isinstance(manifest, LayerManifest)
    assert manifest.role == "rgb"
    assert manifest.layer == "geoportal_rgb"
    assert manifest.provider == "geoportal"
    assert manifest.bands == ["red", "green", "blue"]
    # Grid is defined by the RGB layer.
    assert manifest.grid is not None
    assert manifest.grid.width == 3000 and manifest.grid.height == 3000
    assert manifest.grid.bbox == "210300,521900,210500,522100"
    assert manifest.grid.year_date_map == {2023: "2023-07-15", 2024: "2024-06-01"}
    assert manifest.years_source_map == {2023: "wfs", 2024: "wms"}
    assert len(manifest.assets) == 2
    assert manifest.passed is True


def test_rgb_layer_produce_propagates_failure(monkeypatch, tmp_path: Path):
    def fake_fail(config):
        return 1, tmp_path / "artifacts" / "index_manifest.json"

    monkeypatch.setattr("satmap_dataset.layers.rgb._run_rgb_pipeline", fake_fail)
    layer = get_layer("geoportal_rgb")
    code, manifest = layer.produce(_run_config(tmp_path), grid=None)
    assert code == 1
    assert manifest.passed is False
