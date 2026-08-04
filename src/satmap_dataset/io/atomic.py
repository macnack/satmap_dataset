from __future__ import annotations

from pathlib import Path
from typing import Any

import aiofiles


def part_path_for(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + ".part")


def unlink_quiet(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


async def write_bytes_atomic(output_path: Path, data: bytes) -> None:
    """Write *data* to *output_path* via a sibling ``.part`` file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = part_path_for(output_path)
    try:
        async with aiofiles.open(part_path, "wb") as file_handle:
            await file_handle.write(data)
        part_path.replace(output_path)
    except Exception:
        unlink_quiet(part_path)
        raise


async def write_stream_atomic(output_path: Path, stream: Any) -> None:
    """Drain ``stream.aiter_bytes()`` into *output_path* via a sibling ``.part`` file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = part_path_for(output_path)
    try:
        async with aiofiles.open(part_path, "wb") as file_handle:
            async for chunk in stream.aiter_bytes():
                await file_handle.write(chunk)
        part_path.replace(output_path)
    except Exception:
        unlink_quiet(part_path)
        raise
