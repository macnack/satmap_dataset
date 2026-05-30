from __future__ import annotations

from pathlib import Path

from satmap_dataset.config import DemConfig
from satmap_dataset.layers import get_layer
from satmap_dataset.layers.dem import DemLayer
from satmap_dataset.models import LayerManifest, ReferenceGrid


def _dem_config(tmp_path: Path) -> DemConfig:
    return DemConfig(
        bbox="210300,521900,210500,522100",
        srs="EPSG:2180",
        transport="skorowidz",
        year_start=2023,
        year_end=2024,
        products=["nmt", "nmpt"],
        vertical_datum="evrf2007",
        dem_root=tmp_path / "dem",
        output_json=tmp_path / "dem" / "dem_manifest.json",
    )


def test_registry_resolves_dem_layer():
    layer = get_layer("dem")
    assert isinstance(layer, DemLayer)
    assert layer.role == "dem"
    assert layer.defines_grid is False


def test_dem_layer_injects_grid_and_returns_manifest(monkeypatch, tmp_path: Path):
    captured = {}

    def fake_dem_run(config: DemConfig):
        captured["config"] = config
        manifest = LayerManifest(
            layer="dem",
            role="dem",
            stage="dem",
            provider="geoportal",
            bands=["nmt", "nmpt"],
            years_included=[2023],
            assets=[str(tmp_path / "dem" / "aligned.tif")],
            passed=True,
        )
        config.output_json.parent.mkdir(parents=True, exist_ok=True)
        config.output_json.write_text(manifest.model_dump_json(), encoding="utf-8")
        return 0, config.output_json

    monkeypatch.setattr("satmap_dataset.layers.dem.dem_pipeline.run", fake_dem_run)

    grid = ReferenceGrid(
        bbox="210300,521900,210500,522100",
        width=3000,
        height=3000,
        srs="EPSG:2180",
        px_per_meter=15.0,
    )
    layer = get_layer("dem")
    code, manifest = layer.produce(_dem_config(tmp_path), grid)

    assert code == 0
    assert isinstance(manifest, LayerManifest)
    assert manifest.role == "dem"
    # Grid was injected into the DemConfig handed to the pipeline.
    cfg = captured["config"]
    assert cfg.align_to_render is True
    assert cfg.target_bbox == "210300,521900,210500,522100"
    assert cfg.target_width == 3000 and cfg.target_height == 3000


def test_dem_layer_bands_from_products(tmp_path: Path):
    layer = get_layer("dem")
    assert layer.bands(_dem_config(tmp_path)) == ["nmt", "nmpt"]
