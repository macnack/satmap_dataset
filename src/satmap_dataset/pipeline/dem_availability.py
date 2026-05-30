from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from satmap_dataset.config import DemAvailabilityConfig
from satmap_dataset.geoportal import dem_skorowidz_client
from satmap_dataset.geoportal.http import RetryPolicy
from satmap_dataset.models import DemAvailabilityEntry, DemAvailabilityReport

logger = logging.getLogger("satmap_dataset.dem_availability")

_FULL_THRESHOLD = 99.9


def _parse_bbox(value: str) -> tuple[float, float, float, float]:
    parts = [float(p.strip()) for p in value.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must have format xmin,ymin,xmax,ymax")
    return (parts[0], parts[1], parts[2], parts[3])


def _coverage_pct(
    aoi: tuple[float, float, float, float],
    tile_bboxes: list[tuple[float, float, float, float]],
    *,
    grid: int = 200,
) -> float:
    """Percent of the AOI rectangle covered by the union of tile rectangles.

    Both the AOI and the tile bboxes must be in the SAME coordinate convention
    (the report uses the swapped WFS query space for both, so the ratio is
    orientation-invariant). Computed by sampling a ``grid`` x ``grid`` lattice of
    cell centres over the AOI — no geometry dependency.
    """
    import numpy as np

    a0, b0, a1, b1 = aoi
    if a1 <= a0 or b1 <= b0:
        return 0.0
    if not tile_bboxes:
        return 0.0
    ax = a0 + (np.arange(grid) + 0.5) * (a1 - a0) / grid
    by = b0 + (np.arange(grid) + 0.5) * (b1 - b0) / grid
    gx, gy = np.meshgrid(ax, by)
    covered = np.zeros((grid, grid), dtype=bool)
    for t0, u0, t1, u1 in tile_bboxes:
        lo_a, hi_a = (t0, t1) if t0 <= t1 else (t1, t0)
        lo_b, hi_b = (u0, u1) if u0 <= u1 else (u1, u0)
        covered |= (gx >= lo_a) & (gx <= hi_a) & (gy >= lo_b) & (gy <= hi_b)
    return float(round(covered.mean() * 100.0, 1))


def _classify(pct: float) -> str:
    if pct >= _FULL_THRESHOLD:
        return "full"
    if pct > 0.0:
        return "partial"
    return "none"


def _formats_from_urls(urls: list[str]) -> list[str]:
    found: set[str] = set()
    for url in urls:
        name = Path(url).name.lower()
        if name.endswith(".xyz.zip"):
            found.add("xyz.zip")
        elif name.endswith(".zip"):
            found.add("zip")
        elif name.endswith(".xyz"):
            found.add("xyz")
        elif name.endswith(".asc"):
            found.add("asc")
        elif name.endswith((".tif", ".tiff")):
            found.add("tif")
    return sorted(found)
