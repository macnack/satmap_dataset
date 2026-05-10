from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from typer.testing import CliRunner

from satmap_dataset.cli import app
from satmap_dataset.providers.nls import provider as nls_provider

runner = CliRunner()


def _fixture_xml() -> bytes:
    return (
        Path(__file__).parent / "fixtures" / "nls" / "describe_coverage_ortokuva_vari.xml"
    ).read_bytes()


def _fixture_oapif(years: list[int]) -> bytes:
    import json
    features = [
        {"type": "Feature", "id": str(i), "geometry": None,
         "properties": {"kuvausvuosi": str(year)}}
        for i, year in enumerate(years)
    ]
    return json.dumps({"type": "FeatureCollection", "features": features}).encode("utf-8")


def _patch_fetchers(monkeypatch, available_years: list[int]):
    monkeypatch.setattr(nls_provider, "_fetch_describe_coverage_xml", lambda **kw: _fixture_xml())
    monkeypatch.setattr(
        nls_provider, "_fetch_oapif_items_geojson", lambda **kw: _fixture_oapif(available_years)
    )


def test_nls_index_json_invokes_provider(monkeypatch, tmp_path):
    _patch_fetchers(monkeypatch, available_years=[2018, 2019, 2020, 2021, 2022])
    cfg = {
        "year_start": 2018,
        "year_end": 2022,
        "bbox": "385000,6675000,387000,6677000",
        "srs": "EPSG:3067",
        "provider": "nls",
        "provider_options": {"api_key": "test-key"},
        "output_json": str(tmp_path / "index_manifest.json"),
        "year_availability_output_json": str(tmp_path / "year_availability_report.json"),
    }
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")
    result = runner.invoke(app, ["nls-index-json", str(config_path)])
    assert result.exit_code == 0, result.output
    data = json.loads((tmp_path / "index_manifest.json").read_text(encoding="utf-8"))
    assert data["provider"] == "nls"
    assert data["years_included"]


def test_nls_index_json_validation_error_exits_2(tmp_path):
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps({"year_start": 2030, "year_end": 2020, "bbox": "0,0,1,1"}))
    result = runner.invoke(app, ["nls-index-json", str(config_path)])
    assert result.exit_code == 2


def test_nls_index_json_rejects_config_without_provider_when_srs_wrong(monkeypatch, tmp_path):
    """A config missing 'provider' is still NLS (CLI is named nls-index-json) and must trigger
    the EPSG:3067 guard. Otherwise a Polish-default EPSG:2180 bbox could reach NLS endpoints.
    """
    _patch_fetchers(monkeypatch, available_years=[2018])
    cfg = {
        "year_start": 2018,
        "year_end": 2018,
        "bbox": "210300,521900,210500,522100",
        "srs": "EPSG:2180",  # Polish CRS — must be rejected
        # No "provider" field on purpose
        "provider_options": {"api_key": "test-key"},
        "output_json": str(tmp_path / "index_manifest.json"),
        "year_availability_output_json": str(tmp_path / "year_availability_report.json"),
    }
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")
    result = runner.invoke(app, ["nls-index-json", str(config_path)])
    assert result.exit_code == 2
    assert "EPSG:3067" in result.output


def test_nls_download_json_uses_index_manifest_from_config_output_json(monkeypatch, tmp_path):
    """Standalone download-json must read the index manifest from the same config's output_json,
    not from DownloadConfig.index_manifest's default of artifacts/index_manifest.json.
    """
    import httpx
    from satmap_dataset.models import IndexManifest, YearStatus

    # Pre-write an IndexManifest at the path the config's output_json points to.
    index_path = tmp_path / "custom_index_manifest.json"
    manifest = IndexManifest(
        provider="nls",
        year_start=2018,
        year_end=2018,
        bbox="385000,6675000,387000,6677000",
        srs="EPSG:3067",
        years_requested=[2018],
        year_statuses=[YearStatus(year=2018, typename_exists=True, feature_count=1, status="has_features")],
        years_available_wfs=[2018],
        years_included=[2018],
        common_tile_ids=["nls_2018"],
        tile_sources_by_year={2018: {"nls_2018": "https://example.test/wcs?year=2018"}},
        tile_bboxes_by_year={2018: {"nls_2018": [0, 0, 1, 1]}},
        passed=True,
    )
    index_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    def handler(request):
        # Real RGB tile (above the partial-coverage threshold).
        import numpy as np
        import tifffile
        import io

        buf = io.BytesIO()
        tifffile.imwrite(buf, np.full((64, 64, 3), 128, dtype=np.uint8), photometric="rgb")
        return httpx.Response(200, content=buf.getvalue())

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "satmap_dataset.providers.nls.provider._make_async_client",
        lambda **kw: httpx.AsyncClient(transport=transport, **kw),
    )

    cfg_payload = {
        "year_start": 2018,
        "year_end": 2018,
        "bbox": "385000,6675000,387000,6677000",
        "srs": "EPSG:3067",
        "provider": "nls",
        "provider_options": {"api_key": "test-key"},
        "output_json": str(index_path),  # the SIVL convention: this is the INDEX path
        "year_availability_output_json": str(tmp_path / "year_availability_report.json"),
        "download_root": str(tmp_path / "downloads"),
    }
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps(cfg_payload), encoding="utf-8")
    result = runner.invoke(app, ["nls-download-json", str(config_path)])
    assert result.exit_code == 0, result.output

    # Download manifest landed beside the index, not at artifacts/dataset_manifest_download.json.
    download_manifest_path = tmp_path / "dataset_manifest_download.json"
    assert download_manifest_path.is_file(), "download manifest must land beside the index"
    data = json.loads(download_manifest_path.read_text(encoding="utf-8"))
    assert data["years_included"] == [2018]
    assert (tmp_path / "downloads" / "2018" / "nls_2018.tif").is_file()


def test_nls_run_json_keeps_index_and_download_manifests_separate(monkeypatch, tmp_path):
    """Regression: run-json must not let download_cfg.output_json overwrite the index manifest."""
    import httpx

    _patch_fetchers(monkeypatch, available_years=[2018, 2019])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"FAKE_TIFF_BYTES")

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "satmap_dataset.providers.nls.provider._make_async_client",
        lambda **kw: httpx.AsyncClient(transport=transport, **kw),
    )

    cfg = {
        "year_start": 2018,
        "year_end": 2019,
        "bbox": "385000,6675000,387000,6677000",
        "srs": "EPSG:3067",
        "provider": "nls",
        "provider_options": {"api_key": "test-key"},
        "output_json": str(tmp_path / "index_manifest.json"),
        "year_availability_output_json": str(tmp_path / "year_availability_report.json"),
        "download_root": str(tmp_path / "downloads"),
    }
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")
    result = runner.invoke(app, ["nls-run-json", str(config_path)])
    assert result.exit_code == 0, result.output

    index_manifest = json.loads((tmp_path / "index_manifest.json").read_text(encoding="utf-8"))
    download_manifest_path = tmp_path / "dataset_manifest_download.json"
    assert download_manifest_path.is_file(), "download manifest must land beside the index"
    download_manifest = json.loads(download_manifest_path.read_text(encoding="utf-8"))
    assert index_manifest["kind"] == "index_manifest"
    assert download_manifest["kind"] == "dataset_manifest"
    # Ensure index wasn't clobbered:
    assert "tile_sources_by_year" in index_manifest
