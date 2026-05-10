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


def test_download_drops_empty_no_data_tiles(monkeypatch, tmp_path):
    """Empty WCS tiles (all zeros, ~196 KB) get dropped from the manifest and disk."""
    import numpy as np
    import tifffile

    index_path = _write_index_manifest(tmp_path, [2011, 2013])
    cfg = DownloadConfig(
        index_manifest=index_path,
        download_root=tmp_path / "downloads",
        provider="nls",
        provider_options={"api_key": "test-key"},
        bbox="385000,6675000,387000,6677000",
        srs="EPSG:3067",
        output_json=tmp_path / "dataset_manifest_download.json",
    )

    empty_tile = tmp_path / "empty.tif"
    real_tile = tmp_path / "real.tif"
    tifffile.imwrite(empty_tile, np.zeros((64, 64, 3), dtype=np.uint8), photometric="rgb")
    tifffile.imwrite(real_tile, np.full((64, 64, 3), 128, dtype=np.uint8), photometric="rgb")
    empty_bytes = empty_tile.read_bytes()
    real_bytes = real_tile.read_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        if "2011" in str(request.url):
            return httpx.Response(200, content=empty_bytes)
        return httpx.Response(200, content=real_bytes)

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "satmap_dataset.providers.nls.provider._make_async_client",
        lambda **kw: httpx.AsyncClient(transport=transport, **kw),
    )

    exit_code, manifest_path = NlsProvider().download(cfg)
    assert exit_code == 0
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["years_included"] == [2013]
    assert data["years_excluded_with_reason"]["2011"] == "wcs_returned_empty_tile"
    assert data["provider_metadata"]["dropped_years"]["2011"] == "wcs_returned_empty_tile"
    # The empty tile must not remain on disk.
    assert not (cfg.download_root / "2011" / "nls_2011.tif").exists()
    assert (cfg.download_root / "2013" / "nls_2013.tif").is_file()


def test_download_drops_partial_coverage_strips(monkeypatch, tmp_path):
    """Tiles with non-zero pixels but below the coverage threshold get dropped."""
    import numpy as np
    import tifffile

    index_path = _write_index_manifest(tmp_path, [2013, 2014])
    cfg = DownloadConfig(
        index_manifest=index_path,
        download_root=tmp_path / "downloads",
        provider="nls",
        provider_options={"api_key": "test-key", "min_coverage_ratio": 0.5},
        bbox="385000,6675000,387000,6677000",
        srs="EPSG:3067",
        output_json=tmp_path / "dataset_manifest_download.json",
    )

    # 2013: a thin strip of real data along one row, ~1.5% coverage
    partial = np.zeros((64, 64, 3), dtype=np.uint8)
    partial[0, :, :] = 200
    partial_path = tmp_path / "partial.tif"
    tifffile.imwrite(partial_path, partial, photometric="rgb")
    # 2014: full coverage
    full_path = tmp_path / "full.tif"
    tifffile.imwrite(
        full_path, np.full((64, 64, 3), 128, dtype=np.uint8), photometric="rgb"
    )
    partial_bytes = partial_path.read_bytes()
    full_bytes = full_path.read_bytes()

    def handler(request):
        if "2013" in str(request.url):
            return httpx.Response(200, content=partial_bytes)
        return httpx.Response(200, content=full_bytes)

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "satmap_dataset.providers.nls.provider._make_async_client",
        lambda **kw: httpx.AsyncClient(transport=transport, **kw),
    )

    exit_code, manifest_path = NlsProvider().download(cfg)
    assert exit_code == 0
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["years_included"] == [2014]
    assert (
        data["years_excluded_with_reason"]["2013"]
        == "wcs_partial_coverage_below_threshold"
    )
    assert not (cfg.download_root / "2013" / "nls_2013.tif").exists()


def test_download_re_checks_strict_years_after_drops(monkeypatch, tmp_path):
    """Empty/partial drops must re-trigger the year policy. With strict_years=True
    and a year dropped, passed must be False even if the survivors downloaded fine.
    """
    import numpy as np
    import tifffile

    # Index claims 2018 and 2019 are both available with strict_years=True.
    manifest = IndexManifest(
        provider="nls",
        year_start=2018,
        year_end=2019,
        bbox="385000,6675000,387000,6677000",
        srs="EPSG:3067",
        strict_years=True,
        years_requested=[2018, 2019],
        year_statuses=[
            YearStatus(year=2018, typename_exists=True, feature_count=1, status="has_features"),
            YearStatus(year=2019, typename_exists=True, feature_count=1, status="has_features"),
        ],
        years_available_wfs=[2018, 2019],
        years_included=[2018, 2019],
        common_tile_ids=["nls_2018", "nls_2019"],
        tile_sources_by_year={
            2018: {"nls_2018": "https://example.test/wcs?year=2018"},
            2019: {"nls_2019": "https://example.test/wcs?year=2019"},
        },
        tile_bboxes_by_year={
            2018: {"nls_2018": [0, 0, 1, 1]},
            2019: {"nls_2019": [0, 0, 1, 1]},
        },
        passed=True,
    )
    index_path = tmp_path / "index_manifest.json"
    index_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    cfg = DownloadConfig(
        index_manifest=index_path,
        download_root=tmp_path / "downloads",
        provider="nls",
        provider_options={"api_key": "test-key"},
        bbox="385000,6675000,387000,6677000",
        srs="EPSG:3067",
        output_json=tmp_path / "dataset_manifest_download.json",
    )

    full = tmp_path / "full.tif"
    empty = tmp_path / "empty.tif"
    tifffile.imwrite(full, np.full((64, 64, 3), 128, dtype=np.uint8), photometric="rgb")
    tifffile.imwrite(empty, np.zeros((64, 64, 3), dtype=np.uint8), photometric="rgb")
    full_bytes = full.read_bytes()
    empty_bytes = empty.read_bytes()

    def handler(request):
        if "2019" in str(request.url):
            return httpx.Response(200, content=empty_bytes)
        return httpx.Response(200, content=full_bytes)

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "satmap_dataset.providers.nls.provider._make_async_client",
        lambda **kw: httpx.AsyncClient(transport=transport, **kw),
    )

    exit_code, manifest_path = NlsProvider().download(cfg)
    assert exit_code != 0, "strict_years=True must fail when any requested year is missing"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["passed"] is False
    assert data["years_included"] == [2018]
    assert data["provider_metadata"]["post_download_policy_errors"]


def test_download_records_failed_year_in_excluded_with_reason(monkeypatch, tmp_path):
    """When a yearly download fails (e.g., 5xx exhausted retries), the year must
    appear in years_excluded_with_reason so downstream validation can diagnose
    the gap, not just disappear into provider_metadata.failed_urls.
    """
    index_path = _write_index_manifest(tmp_path, [2018, 2019])
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

    import numpy as np
    import tifffile
    full = tmp_path / "full.tif"
    tifffile.imwrite(full, np.full((64, 64, 3), 128, dtype=np.uint8), photometric="rgb")
    full_bytes = full.read_bytes()

    def handler(request):
        if "2019" in str(request.url):
            return httpx.Response(404, content=b"not found")
        return httpx.Response(200, content=full_bytes)

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "satmap_dataset.providers.nls.provider._make_async_client",
        lambda **kw: httpx.AsyncClient(transport=transport, **kw),
    )

    exit_code, manifest_path = NlsProvider().download(cfg)
    assert exit_code != 0  # one year failed
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["years_included"] == [2018]
    assert data["years_excluded_with_reason"]["2019"] == "wcs_download_failed"


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
