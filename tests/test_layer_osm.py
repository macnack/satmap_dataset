from __future__ import annotations

from pathlib import Path

from satmap_dataset.config import OsmConfig
from satmap_dataset.layers import get_layer
from satmap_dataset.layers.osm import OsmLayer
from satmap_dataset.models import LayerManifest, ReferenceGrid


def _osm_config(tmp_path: Path) -> OsmConfig:
    return OsmConfig(
        bbox="210300,521900,210500,522100",
        srs="EPSG:2180",
        osm_root=tmp_path / "osm",
        output_json=tmp_path / "osm" / "osm_manifest.json",
        categories=["buildings", "water"],
    )


def test_registry_resolves_osm_layer():
    layer = get_layer("osm")
    assert isinstance(layer, OsmLayer)
    assert layer.role == "labels"
    assert layer.defines_grid is False
    assert layer.provider_name is None if hasattr(layer, "provider_name") else True


def test_osm_layer_injects_grid_and_returns_manifest(monkeypatch, tmp_path: Path):
    captured = {}

    def fake_osm_run(config: OsmConfig):
        captured["config"] = config
        manifest = LayerManifest(
            layer="osm",
            role="labels",
            stage="osm",
            bands=["buildings", "water"],
            years_included=[2024],
            assets=[str(tmp_path / "osm" / "year_2024_buildings.tif")],
            passed=True,
        )
        config.output_json.parent.mkdir(parents=True, exist_ok=True)
        config.output_json.write_text(manifest.model_dump_json(), encoding="utf-8")
        return 0, config.output_json

    monkeypatch.setattr("satmap_dataset.layers.osm.osm_pipeline.run", fake_osm_run)

    grid = ReferenceGrid(
        bbox="210300,521900,210500,522100",
        width=3000,
        height=3000,
        srs="EPSG:2180",
        year_date_map={2024: "2024-06-01T00:00:00Z"},
    )
    layer = get_layer("osm")
    code, manifest = layer.produce(_osm_config(tmp_path), grid)

    assert code == 0
    assert isinstance(manifest, LayerManifest)
    assert manifest.role == "labels"
    cfg = captured["config"]
    assert cfg.target_width == 3000 and cfg.target_height == 3000
    assert cfg.year_date_map == {2024: "2024-06-01T00:00:00Z"}


def test_osm_layer_bands_from_categories(tmp_path: Path):
    layer = get_layer("osm")
    assert layer.bands(_osm_config(tmp_path)) == ["buildings", "water"]
