from __future__ import annotations

from pathlib import Path

from satmap_dataset.config import DemConfig, OsmConfig, PipelineConfig, RunConfig
from satmap_dataset.models import LayerManifest, PipelineManifest, ReferenceGrid
from satmap_dataset.pipeline import orchestrator


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
    def __init__(self, role, manifest, recorder=None, code=0):
        self.role = role
        self._manifest = manifest
        self._recorder = recorder
        self._code = code

    def bands(self, config):
        return []

    def produce(self, config, grid):
        if self._recorder is not None:
            self._recorder["grid"] = grid
            self._recorder["called"] = True
        return self._code, self._manifest


def _install_fake_layers(monkeypatch, grid, recorder, *, rgb_code=0):
    rgb_manifest = LayerManifest(
        layer="geoportal_rgb",
        role="rgb",
        grid=grid,
        passed=rgb_code == 0,
        source_manifest=str(recorder.get("fail_path", "")),
    )
    dem_manifest = LayerManifest(layer="dem", role="dem", passed=True)
    osm_manifest = LayerManifest(layer="osm", role="labels", passed=True)
    layers = {
        "geoportal_rgb": _FakeLayer("rgb", rgb_manifest, code=rgb_code),
        "dem": _FakeLayer("dem", dem_manifest, recorder["dem"]),
        "osm": _FakeLayer("labels", osm_manifest, recorder["osm"]),
    }
    monkeypatch.setattr(orchestrator, "get_layer", lambda name: layers[name])


def test_orchestrator_passes_grid_and_writes_pipeline_manifest(monkeypatch, tmp_path: Path):
    grid = ReferenceGrid(
        bbox="210300,521900,210500,522100",
        width=3000,
        height=3000,
        srs="EPSG:2180",
        year_date_map={2024: "2024-06-01"},
    )
    recorder = {"dem": {}, "osm": {}}
    _install_fake_layers(monkeypatch, grid, recorder)
    monkeypatch.setattr(
        orchestrator.validator,
        "run",
        lambda cfg: (0, cfg.output_json),
    )

    code, path = orchestrator.run(
        PipelineConfig(
            rgb=_rgb_config(tmp_path),
            dem=_dem_config(tmp_path),
            osm=_osm_config(tmp_path),
            run_dem=True,
            run_osm=True,
            run_validate=True,
        )
    )

    assert code == 0
    assert path.name == "pipeline_manifest.json"
    pm = PipelineManifest.model_validate_json(path.read_text())
    assert pm.passed is True
    assert pm.layers_completed == ["geoportal_rgb", "dem", "osm"]
    assert pm.grid == grid
    assert recorder["dem"]["grid"] is grid
    assert recorder["osm"]["grid"] is grid
    assert (tmp_path / "artifacts" / "rgb_layer_manifest.json").exists()
    assert (tmp_path / "dem" / "dem_manifest.json").exists()
    assert pm.validation_report is not None


def test_orchestrator_rgb_failure_short_circuits(monkeypatch, tmp_path: Path):
    grid = ReferenceGrid(bbox="0,0,1,1", width=10, height=10, srs="EPSG:2180")
    fail_path = tmp_path / "artifacts" / "index_manifest.json"
    fail_path.parent.mkdir(parents=True, exist_ok=True)
    fail_path.write_text("{}", encoding="utf-8")
    recorder = {"dem": {}, "osm": {}, "fail_path": str(fail_path)}
    _install_fake_layers(monkeypatch, grid, recorder, rgb_code=1)

    code, path = orchestrator.run(
        PipelineConfig(
            rgb=_rgb_config(tmp_path),
            dem=_dem_config(tmp_path),
            run_dem=True,
            run_validate=False,
        )
    )
    assert code == 1
    pm = PipelineManifest.model_validate_json(path.read_text())
    assert pm.passed is False
    assert pm.failed_artifact == str(fail_path)
    assert recorder["dem"] == {}
    assert not (tmp_path / "dem" / "dem_manifest.json").exists()


def test_orchestrator_skips_optional_layers(monkeypatch, tmp_path: Path):
    grid = ReferenceGrid(bbox="0,0,1,1", width=10, height=10, srs="EPSG:2180")
    recorder = {"dem": {}, "osm": {}}
    _install_fake_layers(monkeypatch, grid, recorder)

    code, path = orchestrator.run(
        PipelineConfig(
            rgb=_rgb_config(tmp_path),
            dem=_dem_config(tmp_path),
            osm=_osm_config(tmp_path),
            run_dem=False,
            run_osm=False,
            run_validate=False,
        )
    )
    assert code == 0
    pm = PipelineManifest.model_validate_json(path.read_text())
    assert pm.layers_requested == ["geoportal_rgb"]
    assert pm.layers_completed == ["geoportal_rgb"]
    assert recorder["dem"] == {}
