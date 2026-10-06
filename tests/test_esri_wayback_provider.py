"""Offline end-to-end tests for the esri_wayback provider (all HTTP mocked)."""

from __future__ import annotations

import io
import json
import re
import shutil
import sys
from pathlib import Path

import httpx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.config import DownloadConfig, IndexConfig, RunConfig, ValidateConfig
from satmap_dataset.models import IndexManifest, LayerManifest
from satmap_dataset.pipeline import rgb_pipeline, validator
from satmap_dataset.providers import get_provider
from satmap_dataset.providers.esri_wayback import provider as provider_module
from satmap_dataset.providers.esri_wayback import tiles

FIXTURES = ROOT / "tests" / "fixtures" / "esri_wayback"
CAPS = (FIXTURES / "wmts_capabilities_excerpt.xml").read_bytes()
CONFIG = (FIXTURES / "waybackconfig_excerpt.json").read_bytes()

# Fixture releases (newest first): 26334, 22869, 60013, 64776, 4230, 10.
TILEMAP_SELECT = {26334: 22869, 64776: 4230}
CAPTURES = {
    "2026_r03": (20230905, "Poland Orthos 2023", 0.05),  # release 22869
    "2024_r02": (20210428, "Poland Orthos", 0.05),  # release 60013
    "2014_r03": (20110804, "WV02", 0.5),  # release 4230
    "2014_r01": (20110804, "WV02", 0.5),  # release 10: same capture as 4230
}
COLOR = {22869: (200, 40, 40), 60013: (40, 200, 40), 4230: (40, 40, 200), 10: (90, 90, 90)}

# ~300 m square in Warsaw (EPSG:3857 and EPSG:2180).
BBOX_3857 = "2335800,6841500,2336300,6842000"
BBOX_2180 = "635300,487000,635600,487300"


def _jpeg(rgb: tuple[int, int, int]) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.full((256, 256, 3), rgb, dtype=np.uint8)).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


class FakeWayback:
    def __init__(self) -> None:
        self.requests: list[str] = []
        self.tilemap_fails = False

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(url)
        if url.endswith("WMTSCapabilities.xml"):
            return httpx.Response(200, content=CAPS)
        if url.endswith("waybackconfig.json"):
            return httpx.Response(200, content=CONFIG)
        m = re.search(r"/tilemap/(\d+)/(\d+)/(\d+)/(\d+)$", url)
        if m:
            if self.tilemap_fails:
                return httpx.Response(503)
            release = int(m.group(1))
            body: dict = {"data": [1]}
            if release in TILEMAP_SELECT:
                body["select"] = [TILEMAP_SELECT[release]]
            return httpx.Response(200, json=body)
        m = re.search(r"World_Imagery_Metadata_(\d{4}_r\d+)/MapServer/(\d+)/query", url)
        if m:
            cap = CAPTURES.get(m.group(1))
            feats = (
                [{"attributes": {"SRC_DATE": cap[0], "SRC_DESC": cap[1], "SRC_RES": cap[2], "SRC_ACC": 99999, "NICE_DESC": "x"}}]
                if cap
                else []
            )
            return httpx.Response(200, json={"features": feats})
        m = re.search(r"/tile/(\d+)/(\d+)/(\d+)/(\d+)$", url)
        if m:
            release = int(m.group(1))
            origin = TILEMAP_SELECT.get(release, release)
            if origin != release:
                # Wayback redirects re-served tiles to the release owning the pixels.
                return httpx.Response(301, headers={"Location": url.replace(f"/tile/{release}/", f"/tile/{origin}/")})
            return httpx.Response(200, content=_jpeg(COLOR.get(origin, (0, 0, 0))), headers={"Content-Type": "image/jpeg"})
        return httpx.Response(404)


@pytest.fixture()
def fake(monkeypatch) -> FakeWayback:
    server = FakeWayback()

    def client(options, *, timeout, concurrency=2):
        return httpx.AsyncClient(transport=httpx.MockTransport(server), follow_redirects=True)

    monkeypatch.setattr(provider_module, "_http_client", client)
    return server


FAST = {"index_sleep_min": 0.0, "index_sleep_max": 0.0, "probe": "center"}


def _index_config(tmp_path: Path, **opts) -> IndexConfig:
    return IndexConfig(
        year_start=2010,
        year_end=2026,
        bbox=BBOX_3857,
        srs="EPSG:3857",
        provider="esri_wayback",
        min_years=1,
        output_json=tmp_path / "index_manifest.json",
        year_availability_output_json=tmp_path / "year_availability_report.json",
        provider_options={**FAST, **opts},
    )


def test_index_collapses_releases_into_capture_years(fake: FakeWayback, tmp_path: Path) -> None:
    code, path = get_provider("esri_wayback").index(_index_config(tmp_path))
    assert code == 0
    manifest = IndexManifest.model_validate_json(path.read_text())
    meta = manifest.provider_metadata
    assert meta["releases_total"] == 6
    assert meta["dedupe_mode_effective"] == "tilemap"
    assert meta["distinct_versions_tilemap"] == 4  # 22869, 60013, 4230, 10
    assert meta["distinct_versions"] == 3  # 10 collapses into 4230 (same capture)
    assert manifest.years_included == [2011, 2021, 2023]
    assert manifest.tile_sources_by_year[2023] == {
        "wayback_22869": "https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/"
        "default028mm/MapServer/tile/22869/{z}/{y}/{x}"
    }
    acq = manifest.tile_acquisition_by_year[2023]["wayback_22869"]
    assert (acq.acquisition_date, acq.publication_date, acq.gsd) == ("2023-09-05", "2026-03-25", 0.05)
    by_num = {v["release_num"]: v for v in meta["versions"]}
    assert by_num[22869]["represented_release_nums"] == [26334, 22869]
    assert by_num[22869]["deduped"] is True
    assert by_num[4230]["collapsed_release_nums"] == [10]
    assert by_num[4230]["source"] == "WV02"
    assert meta["selection_by_year"]["2011"]["selected_release_num"] == 4230
    assert "Esri Master Agreement" in meta["license_notice"]
    # Tilemap walk skipped 22869 and 4230 (re-served) rather than querying all 6.
    assert sum(1 for u in fake.requests if "/tilemap/" in u) == 4


def test_index_falls_back_to_content_hash_when_tilemap_down(fake: FakeWayback, tmp_path: Path) -> None:
    fake.tilemap_fails = True
    code, path = get_provider("esri_wayback").index(_index_config(tmp_path, retry_max_attempts=1))
    assert code == 0
    meta = json.loads(path.read_text())["provider_metadata"]
    assert meta["dedupe_mode_effective"] == "content_hash"
    assert meta["distinct_versions_tilemap"] == 4
    assert json.loads(path.read_text())["years_included"] == [2011, 2021, 2023]


def test_index_release_filter_and_policy_failure(fake: FakeWayback, tmp_path: Path) -> None:
    cfg = _index_config(tmp_path, release_numbers=[26334])
    cfg.min_years = 2
    code, path = get_provider("esri_wayback").index(cfg)
    payload = json.loads(path.read_text())
    assert code == 1
    assert payload["provider_metadata"]["releases_considered"] == 1
    # 26334 re-serves 22869 (outside the filter) — still resolved as that version.
    assert payload["years_included"] == [2023]


def test_index_dedupe_none_keeps_every_release(fake: FakeWayback, tmp_path: Path) -> None:
    code, path = get_provider("esri_wayback").index(
        _index_config(tmp_path, dedupe_mode="none", collapse_same_capture=False)
    )
    assert code == 0
    assert json.loads(path.read_text())["provider_metadata"]["distinct_versions"] == 6


def test_download_stitches_georeferenced_3857_geotiff(fake: FakeWayback, tmp_path: Path) -> None:
    code, index_path = get_provider("esri_wayback").index(_index_config(tmp_path))
    assert code == 0
    cfg = DownloadConfig(
        index_manifest=index_path,
        download_root=tmp_path / "downloads",
        output_json=tmp_path / "dataset_manifest_download.json",
        provider="esri_wayback",
        mode="hybrid",
        bbox=BBOX_3857,
        srs="EPSG:3857",
        px_per_meter=1.0,
        concurrency=2,
        sleep_min=0.0,
        sleep_max=0.0,
        provider_options={**FAST, "zoom": 16},
    )
    assert cfg.mode == "wms_tiled"
    code, path = get_provider("esri_wayback").download(cfg)
    manifest = LayerManifest.model_validate_json(path.read_text())
    assert code == 0, manifest.notes
    assert manifest.years_included == [2011, 2021, 2023]
    assert manifest.years_source_map == {2011: "wmts", 2021: "wmts", 2023: "wmts"}
    asset = tmp_path / "downloads" / "2023" / "wayback_22869_z16.tif"
    assert str(asset) in manifest.assets
    import tifffile

    from satmap_dataset.pipeline import render

    assert render._read_source_epsg(asset) == 3857
    georef = render._read_georef(asset)
    aoi = tiles.MercatorBBox(*[float(v) for v in BBOX_3857.split(",")])
    assert georef.min_x <= aoi.min_x and georef.max_x >= aoi.max_x
    assert georef.min_y <= aoi.min_y and georef.max_y >= aoi.max_y
    assert georef.pixel_size_x == pytest.approx(tiles.resolution(16))
    pixel = tifffile.imread(asset)[5, 5]
    assert abs(int(pixel[0]) - 200) < 8 and int(pixel[1]) < 60  # release 22869 red
    detail = manifest.provider_metadata["downloads_by_year"]["2023"]
    assert detail["missing_tiles"] == 0 and detail["zoom"] == 16


def _run_config(tmp_path: Path, *, srs: str, bbox: str, **opts) -> RunConfig:
    return RunConfig(
        year_start=2010,
        year_end=2026,
        bbox=bbox,
        srs=srs,
        target_srs=srs,
        provider="esri_wayback",
        mode="hybrid",
        px_per_meter=0.5,
        min_years=1,
        artifacts_dir=tmp_path / "artifacts",
        download_root=tmp_path / "downloads",
        render_root=tmp_path / "rendered",
        sleep_min=0.0,
        sleep_max=0.0,
        overview_levels=[2],
        tile_size=256,
        provider_options={**FAST, "zoom": 16, **opts},
    )


def test_rgb_pipeline_render_validate_and_reuse(fake: FakeWayback, tmp_path: Path) -> None:
    pytest.importorskip("pyvips")
    cfg = _run_config(tmp_path, srs="EPSG:3857", bbox=BBOX_3857)
    code, render_path = rgb_pipeline.run_rgb_pipeline(cfg)
    rendered = LayerManifest.model_validate_json(render_path.read_text())
    assert code == 0, rendered.notes
    assert rendered.years_included == [2011, 2021, 2023]
    assert (tmp_path / "rendered" / "year_2023.tiff").exists()

    vcode, report_path = validator.run(
        ValidateConfig(dataset_manifest=render_path, output_json=tmp_path / "validation_report.json")
    )
    report = json.loads(report_path.read_text())
    assert vcode == 0, report

    # Second run reuses index and download: no new HTTP.
    before = len(fake.requests)
    code, _ = rgb_pipeline.run_rgb_pipeline(cfg)
    assert code == 0 and len(fake.requests) == before

    # Changing a provider option invalidates both index and download.
    cfg2 = _run_config(tmp_path, srs="EPSG:3857", bbox=BBOX_3857, max_versions=50)
    code, _ = rgb_pipeline.run_rgb_pipeline(cfg2)
    assert code == 0 and len(fake.requests) > before


@pytest.mark.skipif(shutil.which("gdalwarp") is None, reason="gdalwarp required for cross-CRS render")
def test_rgb_pipeline_reprojects_3857_tiles_to_epsg2180(fake: FakeWayback, tmp_path: Path) -> None:
    pytest.importorskip("pyvips")
    cfg = _run_config(tmp_path, srs="EPSG:2180", bbox=BBOX_2180)
    code, render_path = rgb_pipeline.run_rgb_pipeline(cfg)
    rendered = LayerManifest.model_validate_json(render_path.read_text())
    assert code == 0, rendered.notes
    from satmap_dataset.pipeline import render

    out = tmp_path / "rendered" / "year_2023.tiff"
    assert render._read_source_epsg(out) == 2180
    assert list((tmp_path / "downloads" / "2023" / "_reprojected").glob("*.tif"))
    vcode, report_path = validator.run(
        ValidateConfig(dataset_manifest=render_path, output_json=tmp_path / "validation_report.json")
    )
    assert vcode == 0, report_path.read_text()


def test_reuse_predicates_track_wayback_inputs(tmp_path: Path) -> None:
    cfg = _run_config(tmp_path, srs="EPSG:3857", bbox=BBOX_3857)
    index = IndexManifest(
        provider="esri_wayback",
        year_start=cfg.year_start,
        year_end=cfg.year_end,
        bbox=cfg.bbox,
        srs=cfg.srs,
        min_years=cfg.min_years,
        years_requested=cfg.requested_years,
        year_statuses=[],
        years_available_wfs=[2023],
        years_included=[2023],
        passed=True,
        run_parameters={"provider_options": dict(cfg.provider_options)},
    )
    assert rgb_pipeline._can_reuse_index(index, cfg)
    changed = cfg.model_copy(update={"provider_options": {**cfg.provider_options, "dedupe_mode": "content_hash"}})
    assert not rgb_pipeline._can_reuse_index(index, changed)

    asset = tmp_path / "downloads" / "2023" / "wayback_1_z16.tif"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"x")
    index_out = tmp_path / "artifacts" / "index_manifest.json"
    download_out = tmp_path / "artifacts" / "dataset_manifest_download.json"
    download = LayerManifest(
        layer="esri_wayback_rgb",
        role="rgb",
        stage="download",
        provider="esri_wayback",
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
    rezoomed = cfg.model_copy(update={"provider_options": {**cfg.provider_options, "zoom": 17}})
    assert not rgb_pipeline._can_reuse_download(download, rezoomed, index_out, download_out)


def test_config_validation_for_wayback() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="projected EPSG"):
        IndexConfig(year_start=2020, year_end=2021, bbox="20,52,21,53", srs="EPSG:4326", provider="esri_wayback")
    with pytest.raises(ValidationError, match="mode must be one of"):
        RunConfig(year_start=2020, year_end=2021, bbox=BBOX_3857, srs="EPSG:3857", provider="esri_wayback", mode="stac")
    with pytest.raises(ValidationError, match="bbox is required"):
        DownloadConfig(provider="esri_wayback", srs="EPSG:3857", mode="wms_tiled")
    run = RunConfig(year_start=2020, year_end=2021, bbox=BBOX_3857, srs="EPSG:3857", provider="esri_wayback")
    assert run.mode == "wms_tiled"


def test_download_rejects_tile_budget_overflow(fake: FakeWayback, tmp_path: Path) -> None:
    code, index_path = get_provider("esri_wayback").index(_index_config(tmp_path))
    cfg = DownloadConfig(
        index_manifest=index_path,
        download_root=tmp_path / "downloads",
        output_json=tmp_path / "dl.json",
        provider="esri_wayback",
        bbox=BBOX_3857,
        srs="EPSG:3857",
        sleep_min=0.0,
        sleep_max=0.0,
        provider_options={**FAST, "zoom": 18, "max_total_tiles": 5},
    )
    before = len(fake.requests)
    code, path = get_provider("esri_wayback").download(cfg)
    assert code == 1
    assert "max_total_tiles" in json.loads(path.read_text())["notes"]
    assert len(fake.requests) == before
