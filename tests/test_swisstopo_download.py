from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.config import DownloadConfig
from satmap_dataset.providers.swisstopo import provider as provider_module


def test_download_builds_wms_and_geotags(monkeypatch, tmp_path: Path) -> None:
    index = {
        "provider": "swisstopo",
        "year_start": 1950,
        "year_end": 1951,
        "bbox": "2600000,1199000,2602000,1201000",
        "srs": "EPSG:2056",
        "strict_years": False,
        "min_years": 1,
        "wfs_bbox_axes_swapped": False,
        "years_requested": [1950, 1951],
        "year_statuses": [],
        "years_available_wfs": [1950, 1951],
        "years_included": [1950, 1951],
        "years_excluded_with_reason": {},
        "common_tile_ids": [],
        "tile_sources_by_year": {
            "1950": {"swissimage_1950": "wms://swisstopo/ch.swisstopo.swissimage-product/1950"},
            "1951": {"swissimage_1951": "wms://swisstopo/ch.swisstopo.swissimage-product/1951"},
        },
        "tile_bboxes_by_year": {},
        "tile_acquisition_by_year": {},
        "passed": True,
        "errors": [],
        "warnings": [],
        "run_parameters": {},
        "provider_metadata": {},
    }
    index_path = tmp_path / "index_manifest.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")

    seen_urls: list[str] = []

    async def fake_download(client, url, output_path, **_kwargs):
        seen_urls.append(url)
        assert "TIME=195" in url or "TIME=1950" in url or "TIME=1951" in url
        assert "LAYERS=ch.swisstopo.swissimage-product" in url
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Minimal little-endian TIFF header-ish payload; geotag will rewrite.
        output_path.write_bytes(b"II*\x00" + b"\x00" * 128)
        return True

    def fake_tag(path, bbox, width, height, srs):
        assert srs == "EPSG:2056"
        assert width > 0 and height > 0
        path.write_bytes(b"II*\x00GEOTAGGED" + b"\x00" * 64)

    monkeypatch.setattr(provider_module, "_download_asset_with_retry", fake_download)
    monkeypatch.setattr(provider_module, "_tag_wms_tile_as_geotiff", fake_tag)

    cfg = DownloadConfig(
        index_manifest=index_path,
        download_root=tmp_path / "downloads",
        output_json=tmp_path / "download_manifest.json",
        provider="swisstopo",
        mode="wms_tiled",
        bbox="2600000,1199000,2602000,1201000",
        srs="EPSG:2056",
        px_per_meter=1.0,
        concurrency=1,
        sleep_min=0.0,
        sleep_max=0.0,
    )
    code, path = asyncio.run(provider_module.SwisstopoProvider()._download_async(cfg))
    assert code == 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["years_included"] == [1950, 1951]
    assert payload["mode"] == "wms_tiled"
    assert len(seen_urls) == 2
    assert all(Path(a).exists() for a in payload["assets"])
