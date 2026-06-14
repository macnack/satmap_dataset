from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.providers.lantmateriet import provider as provider_module


class _FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code

    @property
    def request(self) -> httpx.Request:
        return httpx.Request("GET", "https://example.invalid/")


class _FakeStream:
    def __init__(self, status_code: int) -> None:
        self._response = _FakeResponse(status_code)
        self.calls = 0

    async def __aenter__(self) -> "_FakeStream":
        return self

    async def __aexit__(self, *_exc) -> None:
        return None

    def raise_for_status(self) -> None:
        raise httpx.HTTPStatusError(
            f"status {self._response.status_code}", request=self._response.request, response=httpx.Response(self._response.status_code)
        )

    async def aiter_bytes(self):  # pragma: no cover - not invoked in error path
        yield b""


class _FakeClient:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.stream_calls = 0

    def stream(self, method: str, url: str) -> _FakeStream:  # noqa: ARG002
        self.stream_calls += 1
        return _FakeStream(self.status_code)


def test_403_is_not_retried(tmp_path) -> None:
    client = _FakeClient(403)
    output_path = tmp_path / "tile.tif"

    ok = asyncio.run(
        provider_module._download_asset_with_retry(
            client,  # type: ignore[arg-type]
            "https://example.invalid/tile.tif",
            output_path,
            retries=5,
            retry_delay=0.01,
            sleep_min=0.0,
            sleep_max=0.0,
        )
    )

    assert ok is False
    # Single attempt — 4xx must not be retried.
    assert client.stream_calls == 1


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_5xx_is_retried(status: int, tmp_path) -> None:
    client = _FakeClient(status)
    output_path = tmp_path / "tile.tif"

    asyncio.run(
        provider_module._download_asset_with_retry(
            client,  # type: ignore[arg-type]
            "https://example.invalid/tile.tif",
            output_path,
            retries=2,
            retry_delay=0.0,
            sleep_min=0.0,
            sleep_max=0.0,
        )
    )

    assert client.stream_calls == 3  # initial + 2 retries


def test_wms_fallback_skipped_without_url(monkeypatch, tmp_path) -> None:
    from satmap_dataset.config import DownloadConfig

    config = DownloadConfig(
        bbox="0,0,1,1",
        provider="lantmateriet",
    )
    monkeypatch.delenv("SATMAP_LANTMATERIET_WMS_URL", raising=False)
    monkeypatch.delenv("SATMAP_LANTMATERIET_WMS_LAYER", raising=False)

    urls = provider_module.LantmaterietProvider._build_wms_fallback_urls(
        config=config,
        years=[2015, 2016],
        options={},
    )
    assert urls == {}


def test_wms_fallback_emits_url_when_configured() -> None:
    from satmap_dataset.config import DownloadConfig

    config = DownloadConfig(bbox="100,200,300,400", provider="lantmateriet")
    urls = provider_module.LantmaterietProvider._build_wms_fallback_urls(
        config=config,
        years=[2018],
        options={
            "wms_url": "https://example.invalid/wms",
            "wms_layer": "annual_layer",
        },
    )
    assert 2018 in urls
    assert urls[2018].startswith("https://example.invalid/wms?")
    assert "LAYERS=annual_layer" in urls[2018]
