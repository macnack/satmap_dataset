import json
from pathlib import Path

from satmap_dataset.config import DemConfig
from satmap_dataset.models import DemManifest
from satmap_dataset.pipeline import dem


def _patch_seams(monkeypatch, *, empty=False):
    """Replace network + GDAL + raster IO with deterministic fakes."""

    async def _fake_fetch(config, product, dest_dir, *, retry_policy):
        tile = Path(dest_dir) / f"{product}_0000.tif"
        tile.write_bytes(b"TILE")
        return [tile]

    def _fake_merge(tiles, out_path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"NATIVE")

    def _fake_align(native, out_path, **kwargs):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"ALIGNED")

    monkeypatch.setattr(dem, "_fetch_tiles_for_product", _fake_fetch)
    monkeypatch.setattr(dem, "_merge_tiles", _fake_merge)
    monkeypatch.setattr(dem, "_align_to_grid", _fake_align)
    monkeypatch.setattr(dem, "_coverage_is_empty", lambda path: empty)
    monkeypatch.setattr(dem, "_raster_dims", lambda path: (10, 10))


def test_run_writes_native_and_aligned_for_both_products(tmp_path, monkeypatch):
    _patch_seams(monkeypatch)
    cfg = DemConfig(
        bbox="0,0,100,100",
        dem_root=tmp_path / "dem_x",
        output_json=tmp_path / "dem_x" / "dem_manifest.json",
        target_bbox="0,0,100,100",
        target_width=100,
        target_height=100,
    )
    code, path = dem.run(cfg)
    assert code == 0
    manifest = DemManifest.model_validate_json(Path(path).read_text())
    assert manifest.passed is True
    assert {p.product for p in manifest.products} == {"nmt", "nmpt"}
    for p in manifest.products:
        assert p.passed is True
        assert Path(p.native_path).exists()
        assert Path(p.aligned_path).exists()
        assert "native" in p.native_path and "aligned" in p.aligned_path


def test_run_no_align_when_disabled(tmp_path, monkeypatch):
    _patch_seams(monkeypatch)
    cfg = DemConfig(
        bbox="0,0,100,100",
        products=["nmt"],
        align_to_render=False,
        dem_root=tmp_path / "dem_x",
        output_json=tmp_path / "dem_x" / "dem_manifest.json",
    )
    code, path = dem.run(cfg)
    assert code == 0
    manifest = DemManifest.model_validate_json(Path(path).read_text())
    assert manifest.products[0].aligned_path is None


def test_run_fails_on_empty_coverage(tmp_path, monkeypatch):
    _patch_seams(monkeypatch, empty=True)
    cfg = DemConfig(
        bbox="0,0,100,100",
        products=["nmt"],
        align_to_render=False,
        dem_root=tmp_path / "dem_x",
        output_json=tmp_path / "dem_x" / "dem_manifest.json",
    )
    code, path = dem.run(cfg)
    assert code == 1
    manifest = DemManifest.model_validate_json(Path(path).read_text())
    assert manifest.passed is False
    assert manifest.products[0].passed is False


def test_resolve_align_grid_prefers_render_manifest(tmp_path):
    render_manifest = tmp_path / "dataset_manifest_render.json"
    render_manifest.write_text(json.dumps({
        "kind": "dataset_manifest", "stage": "render",
        "target_bbox": "5,5,55,55", "target_width": 500, "target_height": 500,
    }))
    cfg = DemConfig(
        bbox="0,0,100,100",
        render_manifest=render_manifest,
        target_bbox="0,0,100,100", target_width=100, target_height=100,
    )
    grid = dem._resolve_align_grid(cfg)
    assert grid == ((5.0, 5.0, 55.0, 55.0), 500, 500)


def test_run_reuses_existing_native_without_fetch(tmp_path, monkeypatch):
    calls = {"fetch": 0, "merge": 0, "align": 0}

    async def _fake_fetch(config, product, dest_dir, *, retry_policy):
        calls["fetch"] += 1
        tile = Path(dest_dir) / f"{product}_0000.tif"
        tile.write_bytes(b"TILE")
        return [tile]

    def _fake_merge(tiles, out_path):
        calls["merge"] += 1
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"NATIVE")

    def _fake_align(native, out_path, **kwargs):
        calls["align"] += 1
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"ALIGNED")

    monkeypatch.setattr(dem, "_fetch_tiles_for_product", _fake_fetch)
    monkeypatch.setattr(dem, "_merge_tiles", _fake_merge)
    monkeypatch.setattr(dem, "_align_to_grid", _fake_align)
    monkeypatch.setattr(dem, "_coverage_is_empty", lambda path: False)
    monkeypatch.setattr(dem, "_raster_dims", lambda path: (10, 10))

    dem_root = tmp_path / "dem_x"
    native = dem_root / "native" / "nmt_evrf2007.tif"
    native.parent.mkdir(parents=True, exist_ok=True)
    native.write_bytes(b"PREEXISTING")

    cfg = DemConfig(
        bbox="0,0,100,100", products=["nmt"], overwrite=False,
        dem_root=dem_root, output_json=dem_root / "dem_manifest.json",
        target_bbox="0,0,100,100", target_width=100, target_height=100,
    )
    code, _path = dem.run(cfg)
    assert code == 0
    assert calls["fetch"] == 0
    assert calls["merge"] == 0
    assert calls["align"] == 1


def test_run_partial_failure_marks_not_passed(tmp_path, monkeypatch):
    async def _fake_fetch(config, product, dest_dir, *, retry_policy):
        tile = Path(dest_dir) / f"{product}_0000.tif"
        tile.write_bytes(b"TILE")
        return [tile]

    def _fake_merge(tiles, out_path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"NATIVE")

    def _fake_empty(path):
        return "nmpt" in str(path)  # nmpt empty, nmt fine

    monkeypatch.setattr(dem, "_fetch_tiles_for_product", _fake_fetch)
    monkeypatch.setattr(dem, "_merge_tiles", _fake_merge)
    monkeypatch.setattr(dem, "_align_to_grid", lambda *a, **k: None)
    monkeypatch.setattr(dem, "_coverage_is_empty", _fake_empty)
    monkeypatch.setattr(dem, "_raster_dims", lambda path: (10, 10))

    cfg = DemConfig(
        bbox="0,0,100,100", products=["nmt", "nmpt"], align_to_render=False,
        dem_root=tmp_path / "dem_x", output_json=tmp_path / "dem_x" / "dem_manifest.json",
    )
    code, path = dem.run(cfg)
    assert code == 1
    manifest = DemManifest.model_validate_json(Path(path).read_text())
    assert manifest.passed is False
    by_product = {p.product: p for p in manifest.products}
    assert by_product["nmt"].passed is True
    assert by_product["nmpt"].passed is False


def test_geoportal_provider_dem_delegates(tmp_path, monkeypatch):
    from satmap_dataset.providers.geoportal import GeoportalProvider

    called = {}

    def _fake_run(config):
        called["config"] = config
        return (0, tmp_path / "dem_manifest.json")

    monkeypatch.setattr("satmap_dataset.pipeline.dem.run", _fake_run)
    cfg = DemConfig(bbox="0,0,10,10", dem_root=tmp_path / "dem_x")
    code, path = GeoportalProvider().dem(cfg)
    assert code == 0
    assert called["config"] is cfg
