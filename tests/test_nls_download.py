from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import httpx

from satmap_dataset.config import DownloadConfig
from satmap_dataset.models import IndexManifest, YearStatus
from satmap_dataset.providers.nls.provider import NlsProvider


def _write_index_manifest(tmp_path: Path, years: list[int]) -> Path:
    manifest = IndexManifest(
        provider="nls",
        year_start=min(years),
        year_end=max(years),
        bbox="385000,6675000,387000,6677000",
        srs="EPSG:3067",
        years_requested=years,
        year_statuses=[
            YearStatus(year=y, typename_exists=True, feature_count=1, status="has_features")
            for y in years
        ],
        years_available_wfs=years,
        years_included=years,
        common_tile_ids=[f"nls_{y}" for y in years],
        tile_sources_by_year={y: {f"nls_{y}": f"https://example.test/wcs?year={y}"} for y in years},
        tile_bboxes_by_year={y: {f"nls_{y}": [0, 0, 1, 1]} for y in years},
        passed=True,
    )
    path = tmp_path / "index_manifest.json"
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return path


def test_download_writes_one_geotiff_per_year(monkeypatch, tmp_path):
    index_path = _write_index_manifest(tmp_path, [2018, 2020])
    cfg = DownloadConfig(
        index_manifest=index_path,
        download_root=tmp_path / "downloads",
        provider="nls",
        provider_options={"api_key": "test-key"},
        bbox="385000,6675000,387000,6677000",
        srs="EPSG:3067",
        output_json=tmp_path / "dataset_manifest_download.json",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"FAKE_TIFF_BYTES")

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "satmap_dataset.providers.nls.provider._make_async_client",
        lambda **kw: httpx.AsyncClient(transport=transport, **kw),
    )

    exit_code, manifest_path = NlsProvider().download(cfg)
    assert exit_code == 0
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["provider"] == "nls"
    assert data["mode"] == "wcs"
    assert sorted(data["years_included"]) == [2018, 2020]
    for year in [2018, 2020]:
        out = cfg.download_root / str(year) / f"nls_{year}.tif"
        assert out.is_file()
        assert out.read_bytes() == b"FAKE_TIFF_BYTES"


def test_download_marks_failed_on_http_error(monkeypatch, tmp_path):
    index_path = _write_index_manifest(tmp_path, [2018])
    cfg = DownloadConfig(
        index_manifest=index_path,
        download_root=tmp_path / "downloads",
        provider="nls",
        provider_options={"api_key": "test-key"},
        bbox="385000,6675000,387000,6677000",
        srs="EPSG:3067",
        retries=0,
        output_json=tmp_path / "dataset_manifest_download.json",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, content=b"unauthorized")

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "satmap_dataset.providers.nls.provider._make_async_client",
        lambda **kw: httpx.AsyncClient(transport=transport, **kw),
    )

    exit_code, manifest_path = NlsProvider().download(cfg)
    assert exit_code != 0
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["passed"] is False
    assert data["years_included"] == []
