"""Regression tests for the code-review fixes on the Layer refactor."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from satmap_dataset.config import DemConfig, OsmConfig
from satmap_dataset.layers import get_layer
from satmap_dataset.models import LayerManifest, ReferenceGrid


def test_layer_manifest_load_rejects_legacy_kind_with_clear_error(tmp_path: Path):
    legacy = tmp_path / "dataset_manifest_render.json"
    legacy.write_text(json.dumps({"kind": "dataset_manifest", "stage": "render"}))
    with pytest.raises(ValueError, match="pre-migration manifest"):
        LayerManifest.load(legacy)


def test_layer_manifest_load_parses_current(tmp_path: Path):
    m = LayerManifest(layer="geoportal_rgb", role="rgb", passed=True)
    p = tmp_path / "m.json"
    p.write_text(m.model_dump_json())
    assert LayerManifest.load(p).role == "rgb"


def test_rgb_layer_manifest_carries_flat_target_fields(monkeypatch, tmp_path: Path):
    """Fix #2: RGB layer manifest must carry flat target_* so the validator's
    size + georef-bbox checks stay active."""
    from satmap_dataset.config import RunConfig
    from satmap_dataset.models import TileAcquisitionMetadata

    render_path = tmp_path / "artifacts" / "dataset_manifest_render.json"

    def fake_run_rgb(config):
        render_path.parent.mkdir(parents=True, exist_ok=True)
        m = LayerManifest(
            layer="geoportal_rgb", role="rgb", stage="render", provider="geoportal",
            years_included=[2024],
            assets=[str(tmp_path / "rendered" / "year_2024.tiff")],
            target_bbox="210300,521900,210500,522100",
            target_width=3000, target_height=3000, target_srs="EPSG:2180",
            tile_acquisition_by_year={2024: {"t": TileAcquisitionMetadata(acquisition_date="2024-06-01")}},
            passed=True,
        )
        render_path.write_text(m.model_dump_json())
        return 0, render_path

    monkeypatch.setattr("satmap_dataset.layers.rgb._run_rgb_pipeline", fake_run_rgb)
    cfg = RunConfig(
        year_start=2024, year_end=2024, bbox="210300,521900,210500,522100",
        provider="geoportal", profile="reference",
        artifacts_dir=tmp_path / "artifacts", download_root=tmp_path / "d",
        render_root=tmp_path / "rendered",
    )
    _, manifest = get_layer("geoportal_rgb").produce(cfg, grid=None)
    assert manifest.grid is not None and manifest.grid.width == 3000
    # flat fields mirror the grid so the validator can size-check.
    assert manifest.target_width == 3000
    assert manifest.target_height == 3000
    assert manifest.target_bbox == "210300,521900,210500,522100"
    # per-year RGB asset resolved via render's canonical parser.
    assert manifest.years[0].assets["rgb"].endswith("year_2024.tiff")


def test_validator_skips_rgb_checks_for_non_rgb_role(tmp_path: Path):
    """Fix #10: a DEM/OSM manifest must not fail on the RGB-only pixel checks."""
    from satmap_dataset.config import ValidateConfig
    from satmap_dataset.pipeline import validator

    dem_manifest = LayerManifest(
        layer="dem", role="dem", stage="dem", pixel_profile="DEM_F32",
        years_included=[], assets=[], passed=True,
    )
    mpath = tmp_path / "dem_manifest.json"
    mpath.write_text(dem_manifest.model_dump_json())
    code, out = validator.run(
        ValidateConfig(dataset_manifest=mpath, requested_years=[], output_json=tmp_path / "vr.json")
    )
    report = json.loads(Path(out).read_text())
    assert not any("pixel profile" in e for e in report["errors"]), report["errors"]


def test_dem_layer_rejects_cross_crs_grid(tmp_path: Path):
    """Fix #5: aligning a DEM to a grid in a different CRS must error, not misalign."""
    cfg = DemConfig(
        bbox="0,0,100,100", srs="EPSG:2180", transport="skorowidz",
        year_start=2012, year_end=2012, dem_root=tmp_path / "dem",
        output_json=tmp_path / "dem" / "m.json",
    )
    grid = ReferenceGrid(bbox="0,0,100,100", width=10, height=10, srs="EPSG:32633")
    with pytest.raises(ValueError, match="different CRS"):
        get_layer("dem").produce(cfg, grid)


def test_osm_layer_injects_grid_bbox(monkeypatch, tmp_path: Path):
    """Fix #4: OSM layer must carry the RGB grid bbox (not just config.bbox)."""
    captured = {}

    def fake_osm_run(config: OsmConfig):
        captured["config"] = config
        m = LayerManifest(layer="osm", role="labels", stage="osm", passed=True)
        config.output_json.parent.mkdir(parents=True, exist_ok=True)
        config.output_json.write_text(m.model_dump_json())
        return 0, config.output_json

    monkeypatch.setattr("satmap_dataset.layers.osm.osm_pipeline.run", fake_osm_run)
    cfg = OsmConfig(
        bbox="0,0,100,100", osm_root=tmp_path / "osm",
        output_json=tmp_path / "osm" / "m.json",
    )
    grid = ReferenceGrid(bbox="5,5,95,95", width=90, height=90, srs="EPSG:2180")
    get_layer("osm").produce(cfg, grid)
    assert captured["config"].target_bbox == "5,5,95,95"


def test_osm_run_rejects_unresolvable_srs(tmp_path: Path):
    """A CRS pyproj cannot resolve must fail clearly rather than query a wrong AOI.

    Valid projected CRSes (EPSG:2180/3006/3067) are supported now; see
    test_osm_pipeline.test_run_accepts_epsg3006_and_converts_bbox.
    """
    from satmap_dataset.pipeline import osm as osm_pipeline

    cfg = OsmConfig(
        bbox="0,0,100,100", srs="EPSG:999999",
        osm_root=tmp_path / "osm", output_json=tmp_path / "osm" / "m.json",
        year_date_map={2024: "2024-06-01"},
        target_width=10, target_height=10,
    )
    code, out = osm_pipeline.run(cfg)
    assert code == 1
    manifest = LayerManifest.load(out)
    assert manifest.passed is False
    assert any("cannot project" in e for e in manifest.errors)


def test_location_run_returns_most_severe_exit_code(monkeypatch, tmp_path: Path):
    """Fix #6: a later code-1 failure must not be masked by an earlier code-2."""
    from satmap_dataset.pipeline import location_run

    grid = ReferenceGrid(bbox="0,0,1,1", width=10, height=10, srs="EPSG:2180")

    class _L:
        def __init__(self, code):
            self.code = code

        def produce(self, config, grid=None):
            return self.code, LayerManifest(layer="x", role="dem")

    rgb = LayerManifest(layer="geoportal_rgb", role="rgb", grid=grid, passed=True)

    class _Rgb:
        def produce(self, config, grid=None):
            return 0, rgb

    layers = {"geoportal_rgb": _Rgb(), "dem": _L(2), "osm": _L(1)}
    monkeypatch.setattr(location_run, "get_layer", lambda n: layers[n])

    from satmap_dataset.config import RunConfig

    code, _ = location_run.run_location(
        rgb_config=RunConfig(
            year_start=2024, year_end=2024, bbox="0,0,100,100", provider="geoportal",
            artifacts_dir=tmp_path / "a", download_root=tmp_path / "d", render_root=tmp_path / "r",
        ),
        dem_config=DemConfig(bbox="0,0,100,100", transport="skorowidz", year_start=2024,
                             year_end=2024, dem_root=tmp_path / "dem",
                             output_json=tmp_path / "dem" / "m.json"),
        osm_config=OsmConfig(bbox="0,0,100,100", osm_root=tmp_path / "osm",
                             output_json=tmp_path / "osm" / "m.json"),
        artifacts_dir=tmp_path / "a",
        validate=False,
    )
    assert code == 2  # max(0, 2, 1), and OSM's failure is not masked away
