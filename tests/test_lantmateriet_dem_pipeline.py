from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.config import DemConfig
from satmap_dataset.models import DemProductAsset, LayerManifest
from satmap_dataset.pipeline import dem as dem_common
from satmap_dataset.pipeline import dem_lantmateriet
from satmap_dataset.providers.lantmateriet import stac

FIXTURES = ROOT / "tests" / "fixtures" / "lantmateriet"


def _fixture_items() -> list[stac.StacItem]:
    payload = json.loads((FIXTURES / "stac_hojd_dtm_cog_search.json").read_text())
    return stac.parse_stac_features(payload)


def test_run_writes_manifest_with_mocked_stac_and_gdal(tmp_path, monkeypatch):
    items = _fixture_items()

    async def _fake_search(*_args, **_kwargs):
        return items

    async def _fake_download(client, url, output_path, **_kwargs):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"TILE")
        return True

    def _fake_merge(tiles, out_path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"MOSAIC")

    def _fake_clip(mosaic, out_path, **_kwargs):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"NATIVE")

    def _fake_align(native, out_path, **_kwargs):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"ALIGNED")

    monkeypatch.setenv("SATMAP_LANTMATERIET_DEM_USERNAME", "dem_user")
    monkeypatch.setenv("SATMAP_LANTMATERIET_DEM_PASSWORD", "dem_pass")
    monkeypatch.setattr(dem_lantmateriet.stac, "search_features", _fake_search)
    monkeypatch.setattr(dem_lantmateriet, "_download_asset_with_retry", _fake_download)
    monkeypatch.setattr(dem_common, "_merge_tiles", _fake_merge)
    monkeypatch.setattr(dem_lantmateriet, "_clip_to_bbox", _fake_clip)
    monkeypatch.setattr(dem_common, "_align_to_grid", _fake_align)
    monkeypatch.setattr(dem_common, "_coverage_is_empty", lambda path: False)
    monkeypatch.setattr(dem_common, "_raster_dims", lambda path: (200, 200))
    monkeypatch.setattr(dem_common, "_normalise_elevation_raster", lambda path: None)
    monkeypatch.setattr(dem_common, "_read_nodata", lambda path: -9999.0)

    cfg = DemConfig(
        bbox="675000,6587500,675200,6587700",
        provider="lantmateriet",
        dem_root=tmp_path / "dem_se",
        output_json=tmp_path / "dem_se" / "dem_manifest.json",
        target_bbox="675000,6587500,675200,6587700",
        target_width=200,
        target_height=200,
        sleep_min=0.0,
        sleep_max=0.0,
    )
    code, path = dem_common.run(cfg)
    assert code == 0
    manifest = LayerManifest.model_validate_json(Path(path).read_text())
    assert manifest.provider == "lantmateriet"
    assert manifest.passed is True
    assert manifest.pixel_profile == "DEM_F32"
    products = [DemProductAsset.model_validate(p) for p in manifest.provider_metadata["products"]]
    assert products[0].product == "nmt"
    assert products[0].passed is True
    assert Path(products[0].native_path).exists()
    assert Path(products[0].aligned_path).exists()
    assert manifest.provider_metadata["stac_hojd_collections"] == ["dtm-cog"]
    assert "Markhöjdmodell" in (manifest.notes or "")


def test_run_fails_without_credentials_when_download_needed(tmp_path, monkeypatch):
    items = _fixture_items()

    async def _fake_search(*_args, **_kwargs):
        return items

    monkeypatch.delenv("SATMAP_LANTMATERIET_USERNAME", raising=False)
    monkeypatch.delenv("SATMAP_LANTMATERIET_PASSWORD", raising=False)
    monkeypatch.delenv("SATMAP_LANTMATERIET_DEM_USERNAME", raising=False)
    monkeypatch.delenv("SATMAP_LANTMATERIET_DEM_PASSWORD", raising=False)
    monkeypatch.delenv("SATMAP_LANTMATERIET_API_KEY", raising=False)
    monkeypatch.setattr(dem_lantmateriet.stac, "search_features", _fake_search)

    cfg = DemConfig(
        bbox="675000,6587500,675200,6587700",
        provider="lantmateriet",
        align_to_render=False,
        dem_root=tmp_path / "dem_se",
        output_json=tmp_path / "dem_se" / "dem_manifest.json",
        sleep_min=0.0,
        sleep_max=0.0,
    )
    code, path = dem_common.run(cfg)
    assert code == 1
    manifest = LayerManifest.model_validate_json(Path(path).read_text())
    assert manifest.passed is False
    assert any("Geotorget" in e or "credentials" in e.lower() for e in manifest.errors)


def test_provider_dem_method_dispatches(tmp_path, monkeypatch):
    from satmap_dataset.providers.lantmateriet.provider import LantmaterietProvider

    called = {}

    def _fake_run(config):
        called["ok"] = True
        out = tmp_path / "dem_manifest.json"
        out.write_text(
            LayerManifest(
                layer="dem",
                role="dem",
                stage="dem",
                provider="lantmateriet",
                passed=True,
                assets=[],
            ).model_dump_json()
        )
        return 0, out

    monkeypatch.setattr("satmap_dataset.pipeline.dem_lantmateriet.run", _fake_run)
    cfg = DemConfig(
        bbox="675000,6587500,675200,6587700",
        provider="lantmateriet",
        align_to_render=False,
        dem_root=tmp_path / "dem_se",
        output_json=tmp_path / "dem_se" / "dem_manifest.json",
    )
    code, path = LantmaterietProvider().dem(cfg)
    assert code == 0
    assert called["ok"] is True
    assert path.exists()
