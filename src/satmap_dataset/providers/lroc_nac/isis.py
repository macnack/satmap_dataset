"""ISIS tool discovery and argv builders for LROC NAC projection.

No ISIS import — only PATH checks and command-line construction so CI can mock
``shutil.which`` / subprocess without installing USGS ISIS.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

REQUIRED_TOOLS: tuple[str, ...] = ("lronac2isis", "spiceinit", "cam2map")

WhichFn = Callable[[str], str | None]


def resolve_isis_tools(which: WhichFn | None = None) -> dict[str, Path | None]:
    """Return absolute paths for required ISIS CLIs (None when missing)."""
    lookup = which or shutil.which
    resolved: dict[str, Path | None] = {}
    for name in REQUIRED_TOOLS:
        found = lookup(name)
        resolved[name] = Path(found) if found else None
    return resolved


def missing_isis_tools(tools: dict[str, Path | None] | None = None) -> list[str]:
    """Names of required tools that are not on PATH."""
    resolved = tools if tools is not None else resolve_isis_tools()
    return [name for name in REQUIRED_TOOLS if resolved.get(name) is None]


def isis_available(tools: dict[str, Path | None] | None = None) -> bool:
    return not missing_isis_tools(tools)


def build_lronac2isis_cmd(
    from_img: Path,
    to_cub: Path,
    *,
    binary: str | Path = "lronac2isis",
) -> list[str]:
    return [str(binary), f"from={from_img}", f"to={to_cub}"]


def build_spiceinit_cmd(
    from_cub: Path,
    *,
    binary: str | Path = "spiceinit",
) -> list[str]:
    return [str(binary), f"from={from_cub}"]


def build_cam2map_cmd(
    from_cub: Path,
    to_map: Path,
    *,
    map_file: Path | None = None,
    binary: str | Path = "cam2map",
) -> list[str]:
    cmd = [str(binary), f"from={from_cub}", f"to={to_map}"]
    if map_file is not None:
        cmd.append(f"map={map_file}")
    return cmd
