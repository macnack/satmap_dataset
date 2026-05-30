from __future__ import annotations

from pathlib import Path

from satmap_dataset.config import DemConfig, OsmConfig, RunConfig
from satmap_dataset.models import LayerManifest, ReferenceGrid
from satmap_dataset.pipeline import location_run


def _rgb_config(tmp_path: Path) -> RunConfig:
    return RunConfig(
        year_start=2023,
        year_end=2024,
        bbox="210300,521900,210500,522100",
        provider="geoportal",
        profile="reference",
        artifacts_dir=tmp_path / "artifacts",
        download_root=tmp_path / "downloads",
        render_root=tmp_path / "rendered",
    )


def _dem_config(tmp_path: Path) -> DemConfig:
    return DemConfig(
        bbox="210300,521900,210500,522100",
        transport="skorowidz",
        year_start=2023,
        year_end=2024,
        dem_root=tmp_path / "dem",
        output_json=tmp_path / "dem" / "dem_manifest.json",
    )


def _osm_config(tmp_path: Path) -> OsmConfig:
    return OsmConfig(
        bbox="210300,521900,210500,522100",
        osm_root=tmp_path / "osm",
        output_json=tmp_path / "osm" / "osm_manifest.json",
    )


class _FakeLayer:
    def __init__(self, role, manifest, recorder=None):
        self.role = role
        self._manifest = manifest
        self._recorder = recorder

    def bands(self, config):
        return []

    def produce(self, config, grid):
        if self._recorder is not None:
            self._recorder["grid"] = grid
        return 0, self._manifest


def _install_fake_layers(monkeypatch, grid, recorder):
    rgb_manifest = LayerManifest(layer="geoportal_rgb", role="rgb", grid=grid, passed=True)
    dem_manifest = LayerManifest(layer="dem", role="dem", passed=True)
    osm_manifest = LayerManifest(layer="osm", role="labels", passed=True)
    layers = {
        "geoportal_rgb": _FakeLayer("rgb", rgb_manifest),
        "dem": _FakeLayer("dem", dem_manifest, recorder["dem"]),
        "osm": _FakeLayer("labels", osm_manifest, recorder["osm"]),
    }
    monkeypatch.setattr(location_run, "get_layer", lambda name: layers[name])


def test_run_location_computes_grid_once_and_passes_to_dem_osm(monkeypatch, tmp_path: Path):
    grid = ReferenceGrid(
        bbox="210300,521900,210500,522100",
        width=3000,
        height=3000,
        srs="EPSG:2180",
        year_date_map={2024: "2024-06-01"},
    )
    recorder = {"dem": {}, "osm": {}}
    _install_fake_layers(monkeypatch, grid, recorder)

    code, path = location_run.run_location(
        rgb_config=_rgb_config(tmp_path),
        dem_config=_dem_config(tmp_path),
        osm_config=_osm_config(tmp_path),
        artifacts_dir=tmp_path / "artifacts",
        validate=False,
    )

    assert code == 0
    # The RGB grid is computed once and handed to both downstream layers.
    assert recorder["dem"]["grid"] is grid
    assert recorder["osm"]["grid"] is grid
    # RGB layer manifest is written to artifacts.
    rgb_out = Path(path)
    assert rgb_out.exists()
    assert LayerManifest.model_validate_json(rgb_out.read_text()).role == "rgb"
    # DEM + OSM manifests written to their configured outputs.
    assert (tmp_path / "dem" / "dem_manifest.json").exists()
    assert (tmp_path / "osm" / "osm_manifest.json").exists()


def test_run_location_skips_dem_osm_when_not_requested(monkeypatch, tmp_path: Path):
    grid = ReferenceGrid(bbox="0,0,1,1", width=10, height=10, srs="EPSG:2180")
    recorder = {"dem": {}, "osm": {}}
    _install_fake_layers(monkeypatch, grid, recorder)

    code, _ = location_run.run_location(
        rgb_config=_rgb_config(tmp_path),
        dem_config=_dem_config(tmp_path),
        osm_config=_osm_config(tmp_path),
        artifacts_dir=tmp_path / "artifacts",
        run_dem=False,
        run_osm=False,
        validate=False,
    )
    assert code == 0
    assert recorder["dem"] == {}  # produce never called
    assert not (tmp_path / "dem" / "dem_manifest.json").exists()


def test_run_location_rgb_failure_short_circuits(monkeypatch, tmp_path: Path):
    rgb_manifest = LayerManifest(layer="geoportal_rgb", role="rgb", passed=False)

    class _FailRgb(_FakeLayer):
        def produce(self, config, grid):
            return 1, rgb_manifest

    monkeypatch.setattr(
        location_run, "get_layer", lambda name: _FailRgb("rgb", rgb_manifest)
    )
    code, _ = location_run.run_location(
        rgb_config=_rgb_config(tmp_path),
        dem_config=_dem_config(tmp_path),
        artifacts_dir=tmp_path / "artifacts",
        validate=False,
    )
    assert code == 1
    assert not (tmp_path / "dem" / "dem_manifest.json").exists()
