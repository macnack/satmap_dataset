from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.config import DownloadConfig
from satmap_dataset.providers.naip import provider as provider_module


def test_download_signs_mpc_urls(monkeypatch, tmp_path: Path) -> None:
    unsigned = "https://naipeuwest.blob.core.windows.net/naip/v002/md/2023/example.tif"
    signed = unsigned + "?sv=test-sas"
    index = {
        "provider": "naip",
        "year_start": 2023,
        "year_end": 2023,
        "bbox": "-76.66,39.26,-76.64,39.28",
        "srs": "EPSG:4326",
        "strict_years": False,
        "min_years": 1,
        "wfs_bbox_axes_swapped": False,
        "years_requested": [2023],
        "year_statuses": [],
        "years_available_wfs": [2023],
        "years_included": [2023],
        "years_excluded_with_reason": {},
        "common_tile_ids": [],
        "tile_sources_by_year": {"2023": {"tile-a": unsigned}},
        "tile_bboxes_by_year": {},
        "tile_acquisition_by_year": {},
        "passed": True,
        "errors": [],
        "warnings": [],
        "run_parameters": {},
        "provider_metadata": {"stac_host": "planetary_computer"},
    }
    index_path = tmp_path / "index_manifest.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")

    signed_calls: list[str] = []

    async def fake_sign(client, href, *, sas_url):
        signed_calls.append(href)
        return signed

    async def fake_download(client, url, output_path, **_kwargs):
        assert url == signed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"TIFF")
        return True

    monkeypatch.setattr(provider_module, "_sign_planetary_computer_url", fake_sign)
    monkeypatch.setattr(provider_module, "_download_asset_with_retry", fake_download)

    cfg = DownloadConfig(
        index_manifest=index_path,
        download_root=tmp_path / "downloads",
        output_json=tmp_path / "download_manifest.json",
        provider="naip",
        mode="stac",
        concurrency=1,
        sleep_min=0.0,
        sleep_max=0.0,
    )
    code, path = asyncio.run(provider_module.NaipProvider()._download_async(cfg))
    assert code == 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["years_included"] == [2023]
    assert signed_calls == [unsigned]
    assert any(Path(a).exists() for a in payload["assets"])
