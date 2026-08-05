from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.io.atomic import part_path_for, write_bytes_atomic, write_stream_atomic
from satmap_dataset.pipeline import downloader


class _ByteStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        return None


class _FailingStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b"PARTIAL"
        raise httpx.ReadError("connection reset mid-stream")

    async def aclose(self) -> None:
        return None


def test_write_bytes_atomic_creates_final_file(tmp_path: Path) -> None:
    out = tmp_path / "tile.tif"
    asyncio.run(write_bytes_atomic(out, b"COMPLETE"))
    assert out.read_bytes() == b"COMPLETE"
    assert not part_path_for(out).exists()


def test_write_stream_atomic_cleans_up_part_on_failure(tmp_path: Path) -> None:
    out = tmp_path / "tile.tif"

    class _Stream:
        async def aiter_bytes(self):
            yield b"PARTIAL"
            raise httpx.ReadError("boom")

    with pytest.raises(httpx.ReadError):
        asyncio.run(write_stream_atomic(out, _Stream()))

    assert not out.exists()
    assert not part_path_for(out).exists()


def test_download_with_retry_leaves_no_partial_on_midstream_failure(tmp_path: Path) -> None:
    out = tmp_path / "tile.tif"
    request = httpx.Request("GET", "https://example.invalid/tile.tif")

    class _Client:
        def stream(self, method: str, url: str):
            class _Ctx:
                async def __aenter__(self):
                    return httpx.Response(200, request=request, stream=_FailingStream())

                async def __aexit__(self, *_exc):
                    return None

            return _Ctx()

    ok = asyncio.run(
        downloader._download_with_retry(
            _Client(),  # type: ignore[arg-type]
            "https://example.invalid/tile.tif",
            out,
            retries=0,
            retry_delay=0.0,
            sleep_min=0.0,
            sleep_max=0.0,
        )
    )

    assert ok is False
    assert not out.exists()
    assert not part_path_for(out).exists()
