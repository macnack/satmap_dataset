from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.providers.sentinel2 import provider as provider_module
from satmap_dataset.providers.sentinel2.provider import (
    DEFAULT_PLANETARY_COMPUTER_SAS_URL,
    STAC_HOST_EARTH_SEARCH,
    STAC_HOST_PLANETARY_COMPUTER,
    STAC_URL_BY_HOST,
    _resolve_search_options,
    _resolve_stac_host,
    _sign_planetary_computer_url,
)


def test_default_stac_host_is_earth_search() -> None:
    assert _resolve_stac_host({}) == STAC_HOST_EARTH_SEARCH


def test_explicit_planetary_computer_host_resolves() -> None:
    assert _resolve_stac_host({"stac_host": "planetary_computer"}) == STAC_HOST_PLANETARY_COMPUTER


def test_unknown_host_raises() -> None:
    with pytest.raises(ValueError, match="stac_host"):
        _resolve_stac_host({"stac_host": "googl-earth-engine"})


def test_resolve_search_options_picks_mpc_url_for_mpc_host() -> None:
    opts = _resolve_search_options({"stac_host": "planetary_computer"})
    assert opts.url == STAC_URL_BY_HOST[STAC_HOST_PLANETARY_COMPUTER]


def test_resolve_search_options_picks_earth_search_url_by_default() -> None:
    opts = _resolve_search_options({})
    assert opts.url == STAC_URL_BY_HOST[STAC_HOST_EARTH_SEARCH]


def test_resolve_search_options_explicit_stac_url_overrides_host_default() -> None:
    opts = _resolve_search_options(
        {"stac_host": "planetary_computer", "stac_url": "https://example.org/custom"}
    )
    assert opts.url == "https://example.org/custom"


def test_resolve_search_options_explicit_stac_host_beats_stale_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SATMAP_SENTINEL2_STAC_URL", "https://stale.example.org/search")
    opts = _resolve_search_options({"stac_host": "planetary_computer"})
    assert opts.url == STAC_URL_BY_HOST[STAC_HOST_PLANETARY_COMPUTER]


class _FakeMpcSignTransport(httpx.AsyncBaseTransport):
    """Asserts the SAS sign endpoint is called and returns a deterministic href."""

    def __init__(self) -> None:
        self.calls: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        href = request.url.params.get("href")
        signed = (
            f"{href}?st=2026-01-01T00%3A00%3A00Z"
            "&sig=fake-signature"
            "&sv=2025-07-05"
        )
        body = {"href": signed, "msft:expiry": "2026-01-01T01:00:00Z"}
        return httpx.Response(200, json=body)


def test_sign_helper_calls_sas_endpoint_with_href_param() -> None:
    transport = _FakeMpcSignTransport()

    async def runner() -> str:
        async with httpx.AsyncClient(transport=transport) as client:
            return await _sign_planetary_computer_url(
                client,
                "https://sentinel2l2a01.blob.core.windows.net/sentinel2-l2/foo.tif",
                sas_url=DEFAULT_PLANETARY_COMPUTER_SAS_URL,
            )

    signed = asyncio.run(runner())
    assert "sig=fake-signature" in signed
    assert len(transport.calls) == 1
    sent = transport.calls[0]
    assert sent.url.path == "/api/sas/v1/sign"
    assert sent.url.params.get("href") == "https://sentinel2l2a01.blob.core.windows.net/sentinel2-l2/foo.tif"


def test_sign_helper_raises_when_response_lacks_href() -> None:
    class _BadTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"msft:expiry": "x", "href": ""})

    async def runner() -> None:
        async with httpx.AsyncClient(transport=_BadTransport()) as client:
            await _sign_planetary_computer_url(
                client, "https://example.invalid/foo.tif", sas_url=DEFAULT_PLANETARY_COMPUTER_SAS_URL
            )

    with pytest.raises(RuntimeError, match="no href"):
        asyncio.run(runner())
