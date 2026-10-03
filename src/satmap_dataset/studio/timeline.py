"""Year timeline helpers for satmap-studio (pure status derivation + HTML)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

YearLevel = Literal["rendered", "downloaded", "available", "missing"]

_YEAR_FROM_ASSET = re.compile(r"(?:^|[/\\])year_(\d{4})\.(?:tif|tiff)$", re.IGNORECASE)


@dataclass(frozen=True)
class YearCell:
    year: int
    level: YearLevel
    requested: bool = True
    available: bool = False
    downloaded: bool = False
    rendered: bool = False


def _as_int_set(values: Iterable[Any] | None) -> set[int]:
    out: set[int] = set()
    if not values:
        return out
    for value in values:
        try:
            out.add(int(value))
        except (TypeError, ValueError):
            continue
    return out


def _years_from_asset_paths(assets: Sequence[Any] | None) -> set[int]:
    years: set[int] = set()
    if not assets:
        return years
    for asset in assets:
        text = str(asset).replace("\\", "/")
        match = _YEAR_FROM_ASSET.search(text)
        if match:
            years.add(int(match.group(1)))
            continue
        # download layout: .../<year>/<tile>.tif
        parts = text.split("/")
        for part in parts:
            if part.isdigit() and len(part) == 4:
                years.add(int(part))
                break
    return years


def _available_years_from_index(payload: Mapping[str, Any] | None) -> set[int]:
    if not payload:
        return set()
    available = _as_int_set(payload.get("years_available_wfs")) | _as_int_set(
        payload.get("years_included")
    )
    for status in payload.get("year_statuses") or []:
        if not isinstance(status, Mapping):
            continue
        try:
            year = int(status["year"])
        except (KeyError, TypeError, ValueError):
            continue
        if status.get("status") == "has_features" or int(status.get("feature_count") or 0) > 0:
            available.add(year)
    return available


def _years_from_stage_manifest(payload: Mapping[str, Any] | None) -> set[int]:
    if not payload:
        return set()
    years = _as_int_set(payload.get("years_included"))
    years |= _years_from_asset_paths(payload.get("assets"))
    return years


def _years_present_under_download_root(download_root: Path | None) -> set[int]:
    if download_root is None or not download_root.is_dir():
        return set()
    years: set[int] = set()
    for child in download_root.iterdir():
        if child.is_dir() and child.name.isdigit() and len(child.name) == 4:
            if any(child.iterdir()):
                years.add(int(child.name))
    return years


def _years_present_under_render_root(render_root: Path | None) -> set[int]:
    if render_root is None or not render_root.is_dir():
        return set()
    years: set[int] = set()
    for path in render_root.iterdir():
        if not path.is_file():
            continue
        match = _YEAR_FROM_ASSET.search(path.name)
        if match:
            years.add(int(match.group(1)))
    return years


def load_json_mapping(path: Path | None) -> dict[str, Any] | None:
    """Load a JSON object from disk; return None if missing/invalid."""
    if path is None or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def derive_year_timeline(
    *,
    year_start: int,
    year_end: int,
    index_payload: Mapping[str, Any] | None = None,
    download_payload: Mapping[str, Any] | None = None,
    render_payload: Mapping[str, Any] | None = None,
    download_root: Path | None = None,
    render_root: Path | None = None,
) -> list[YearCell]:
    """Derive per-year status cells for the closed [year_start, year_end] range."""
    if year_end < year_start:
        year_start, year_end = year_end, year_start

    available = _available_years_from_index(index_payload)
    downloaded = _years_from_stage_manifest(download_payload) | _years_present_under_download_root(
        download_root
    )
    rendered = _years_from_stage_manifest(render_payload) | _years_present_under_render_root(
        render_root
    )

    cells: list[YearCell] = []
    for year in range(year_start, year_end + 1):
        is_available = year in available
        is_downloaded = year in downloaded
        is_rendered = year in rendered
        if is_rendered:
            level: YearLevel = "rendered"
        elif is_downloaded:
            level = "downloaded"
        elif is_available:
            level = "available"
        else:
            level = "missing"
        cells.append(
            YearCell(
                year=year,
                level=level,
                requested=True,
                available=is_available,
                downloaded=is_downloaded,
                rendered=is_rendered,
            )
        )
    return cells


_LEVEL_COLORS: dict[YearLevel, tuple[str, str]] = {
    # (background, text)
    "rendered": ("#1b7f4a", "#ffffff"),
    "downloaded": ("#2f6fed", "#ffffff"),
    "available": ("#c9850a", "#1a1a1a"),
    "missing": ("#e8e8e8", "#666666"),
}


def timeline_html(cells: Sequence[YearCell], *, title: str | None = "Year timeline") -> str:
    """Compact horizontal HTML timeline (Streamlit ``unsafe_allow_html``)."""
    if not cells:
        return ""

    legend = (
        '<span style="margin-right:12px;"><span style="display:inline-block;width:10px;height:10px;'
        'background:#1b7f4a;border-radius:2px;margin-right:4px;"></span>rendered</span>'
        '<span style="margin-right:12px;"><span style="display:inline-block;width:10px;height:10px;'
        'background:#2f6fed;border-radius:2px;margin-right:4px;"></span>downloaded</span>'
        '<span style="margin-right:12px;"><span style="display:inline-block;width:10px;height:10px;'
        'background:#c9850a;border-radius:2px;margin-right:4px;"></span>available</span>'
        '<span><span style="display:inline-block;width:10px;height:10px;'
        'background:#e8e8e8;border:1px solid #bbb;border-radius:2px;margin-right:4px;"></span>'
        "requested / missing</span>"
    )

    chips: list[str] = []
    for cell in cells:
        bg, fg = _LEVEL_COLORS[cell.level]
        border = "1px solid #bbbbbb" if cell.level == "missing" else f"1px solid {bg}"
        title_attr = (
            f"{cell.year}: {cell.level}"
            f" (available={cell.available}, downloaded={cell.downloaded}, rendered={cell.rendered})"
        )
        chips.append(
            f'<div title="{title_attr}" style="flex:0 0 auto;min-width:52px;padding:6px 4px;'
            f'text-align:center;border-radius:4px;background:{bg};color:{fg};border:{border};'
            f'font:600 12px/1.2 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;">'
            f"{cell.year}</div>"
        )

    heading = f"<div style='font-weight:600;margin-bottom:6px;'>{title}</div>" if title else ""
    return (
        f"<div style='margin:4px 0 12px 0;'>{heading}"
        f"<div style='font-size:12px;color:#555;margin-bottom:8px;'>{legend}</div>"
        f"<div style='display:flex;flex-wrap:wrap;gap:6px;'>{''.join(chips)}</div></div>"
    )
