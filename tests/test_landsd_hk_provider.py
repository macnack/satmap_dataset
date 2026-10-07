"""Offline end-to-end tests for the landsd_hk provider (HTTP mocked)."""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.config import DownloadConfig, IndexConfig, RunConfig
from satmap_dataset.models import IndexManifest, LayerManifest
from satmap_dataset.pipeline import rgb_pipeline
from satmap_dataset.providers import get_provider

# ~300 m square near HKairport in EPSG:3857.
BBOX_3857 = "12694850,2561700,12695150,2562000"


def _png(rgb: tuple[int, int, int] = (40, 120, 200)) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.full((256, 256, 3), rgb, dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


class FakeLandsd:
    def __init__(self) -> None:
        self.requests: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(url)
        if re.search(r"/xyz/imagery/WGS84/\d+/\d+/\d+\.png$", url):
            return httpx.Response(200, content=_png(), headers={"Content-Type": "image/png"})
        return httpx.Response(404)


_RealAsyncClient = httpx.AsyncClient


@pytest.fixture()
def fake(monkeypatch) -> FakeLandsd:
    server = FakeLandsd()

    def fake_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(server)
        kwargs.setdefault("follow_redirects", True)
        return _RealAsyncClient(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", fake_client)
    return server


def _index_config(tmp_path: Path, **opts) -> IndexConfig:
    return IndexConfig(
        year_start=2020,
        year_end=2025,
        bbox=BBOX_3857,
        srs="EPSG:3857",
        provider="landsd_hk",
        min_years=1,
        output_json=tmp_path / "index_manifest.json",
        year_availability_output_json=tmp_path / "year_availability_report.json",
        provider_options={"imagery_year": 2025, "gsd_m": 0.3, **opts},
    )


def _run_config(tmp_path: Path, **opts) -> RunConfig:
    provider_options = {"imagery_year": 2025, "zoom": 17, "max_tiles": 64}
    provider_options.update(opts.pop("provider_options", {}))
    return RunConfig(
        year_start=2024,
        year_end=2025,
        bbox=BBOX_3857,
        srs="EPSG:3857",
        mode="wms_tiled",
        provider="landsd_hk",
        min_years=1,
        artifacts_dir=tmp_path / "artifacts",
        download_root=tmp_path / "downloads",
        render_root=tmp_path / "rendered",
        sleep_min=0.0,
        sleep_max=0.0,
        overview_levels=[2],
        provider_options=provider_options,
        **opts,
    )


def test_index_reports_single_synthetic_year(tmp_path: Path) -> None:
    code, path = get_provider("landsd_hk").index(_index_config(tmp_path))
    assert code == 0
    manifest = IndexManifest.model_validate_json(path.read_text())
    assert manifest.years_included == [2025]
    assert manifest.years_excluded_with_reason[2020].startswith("landsd_hk XYZ")
    assert manifest.provider_metadata["experimental"] is True
    assert "Lands Department" in manifest.provider_metadata["attribution"]
    assert "landsd_xyz_2025" in manifest.tile_sources_by_year[2025]


def test_index_fails_when_imagery_year_outside_range(tmp_path: Path) -> None:
    code, path = get_provider("landsd_hk").index(_index_config(tmp_path, imagery_year=2019))
    assert code == 1
    manifest = IndexManifest.model_validate_json(path.read_text())
    assert not manifest.passed
    assert any("imagery_year=2019" in e for e in manifest.errors)


def test_download_stitches_geotiff(fake: FakeLandsd, tmp_path: Path) -> None:
    provider = get_provider("landsd_hk")
    _, index_path = provider.index(_index_config(tmp_path))
    cfg = DownloadConfig(
        index_manifest=index_path,
        download_root=tmp_path / "downloads",
        bbox=BBOX_3857,
        srs="EPSG:3857",
        mode="wms_tiled",
        provider="landsd_hk",
        concurrency=4,
        sleep_min=0.0,
        sleep_max=0.0,
        output_json=tmp_path / "dataset_manifest_download.json",
        provider_options={"imagery_year": 2025, "zoom": 17, "max_tiles": 64},
    )
    code, path = provider.download(cfg)
    assert code == 0, path.read_text()
    manifest = LayerManifest.model_validate_json(path.read_text())
    assert manifest.passed and len(manifest.assets) == 1
    asset = Path(manifest.assets[0])
    assert asset.exists() and asset.stat().st_size > 0
    assert fake.requests
    assert all("/xyz/imagery/WGS84/17/" in u for u in fake.requests)


def test_config_validation_for_landsd() -> None:
    with pytest.raises(ValidationError, match="projected EPSG"):
        IndexConfig(
            year_start=2024,
            year_end=2025,
            bbox="114.04,22.41,114.05,22.42",
            srs="EPSG:4326",
            provider="landsd_hk",
        )
    with pytest.raises(ValidationError, match="mode must be one of"):
        RunConfig(
            year_start=2024,
            year_end=2025,
            bbox=BBOX_3857,
            srs="EPSG:3857",
            provider="landsd_hk",
            mode="stac",
        )
    cfg = RunConfig(
        year_start=2024,
        year_end=2025,
        bbox=BBOX_3857,
        srs="EPSG:3857",
        mode="hybrid",
        provider="landsd_hk",
    )
    assert cfg.mode == "wms_tiled"


def test_reuse_predicates_track_landsd_inputs(tmp_path: Path) -> None:
    cfg = _run_config(tmp_path)
    index = IndexManifest(
        provider="landsd_hk",
        year_start=cfg.year_start,
        year_end=cfg.year_end,
        bbox=cfg.bbox,
        srs=cfg.srs,
        min_years=cfg.min_years,
        years_requested=cfg.requested_years,
        year_statuses=[],
        years_available_wfs=[2025],
        years_included=[2025],
        passed=True,
        run_parameters={"provider_options": dict(cfg.provider_options)},
    )
    assert rgb_pipeline._can_reuse_index(index, cfg)
    changed = cfg.model_copy(
        update={"provider_options": {**cfg.provider_options, "zoom": 18}}
    )
    assert not rgb_pipeline._can_reuse_index(index, changed)

    asset = tmp_path / "downloads" / "2025" / "landsd_xyz_z17.tif"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"x")
    index_out = tmp_path / "artifacts" / "index_manifest.json"
    download_out = tmp_path / "artifacts" / "dataset_manifest_download.json"
    download = LayerManifest(
        layer="landsd_hk_rgb",
        role="rgb",
        stage="download",
        provider="landsd_hk",
        assets=[str(asset)],
        source_manifest=str(index_out),
        mode="wms_tiled",
        target_bbox=cfg.bbox,
        target_srs=cfg.srs,
        profile=cfg.profile,
        px_per_meter=cfg.px_per_meter,
        passed=True,
        run_parameters={"provider_options": dict(cfg.provider_options)},
    )
    assert rgb_pipeline._can_reuse_download(download, cfg, index_out, download_out)
    finer = cfg.model_copy(update={"px_per_meter": 4.0})
    assert not rgb_pipeline._can_reuse_download(download, finer, index_out, download_out)
    rezoomed = cfg.model_copy(
        update={"provider_options": {**cfg.provider_options, "zoom": 18}}
    )
    assert not rgb_pipeline._can_reuse_download(download, rezoomed, index_out, download_out)
