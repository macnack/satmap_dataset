"""Canonical bbox axis-order conventions.

Project convention (config, manifests, CLI ``--bbox``, render target grid):
    ``xmin,ymin,xmax,ymax`` in logical (x, y) order. For EPSG:2180 that is
    (easting, northing).

WFS / GUGiK skorowidz query convention:
    OGC WFS 2.0 ``BBOX`` uses CRS *authority* axis order. EPSG:2180 authority
    order is (northing, easting), so the query string is
    ``ymin,xmin,ymax,xmax``.

Downloaded GeoTIFF tie points may still encode axes swapped relative to the
project convention; render infers that separately (``source_axis_mode``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Bbox:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.min_x, self.min_y, self.max_x, self.max_y)

    def swap_axes(self) -> Bbox:
        return Bbox(
            min_x=self.min_y,
            min_y=self.min_x,
            max_x=self.max_y,
            max_y=self.max_x,
        )

    def format(self) -> str:
        return f"{self.min_x},{self.min_y},{self.max_x},{self.max_y}"


def parse(value: str) -> Bbox:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must have format xmin,ymin,xmax,ymax")
    min_x, min_y, max_x, max_y = parts
    if min_x >= max_x or min_y >= max_y:
        raise ValueError("bbox must satisfy xmin<xmax and ymin<ymax")
    return Bbox(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y)


def format_bbox(bbox: Bbox) -> str:
    return bbox.format()


def swap_axes_tuple(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    min_x, min_y, max_x, max_y = bbox
    return min_y, min_x, max_y, max_x


def swap_axes_str(bbox: str) -> str:
    return parse(bbox).swap_axes().format()


def wfs_query_axes_swapped(srs: str) -> bool:
    """True when WFS BBOX for *srs* uses authority axis order (swap vs project)."""
    return (srs or "").strip().upper() == "EPSG:2180"


def wfs_query_bbox_str(project_bbox: str, srs: str) -> str:
    """Convert a project-order bbox string to a WFS ``BBOX`` parameter value."""
    if wfs_query_axes_swapped(srs):
        return swap_axes_str(project_bbox)
    return project_bbox


def overlap_area(
    a: Bbox | tuple[float, float, float, float],
    b: Bbox | tuple[float, float, float, float],
) -> float:
    if isinstance(a, Bbox):
        a = a.as_tuple()
    if isinstance(b, Bbox):
        b = b.as_tuple()
    min_x = max(a[0], b[0])
    min_y = max(a[1], b[1])
    max_x = min(a[2], b[2])
    max_y = min(a[3], b[3])
    if min_x >= max_x or min_y >= max_y:
        return 0.0
    return (max_x - min_x) * (max_y - min_y)


def tile_bboxes_look_swapped_vs_project(
    tile_samples: list[tuple[float, float, float, float]],
    project_bbox: Bbox | tuple[float, float, float, float],
) -> bool:
    """True when stored tile bboxes look like authority-order, not project-order."""
    if not tile_samples:
        return False
    if isinstance(project_bbox, Bbox):
        request = project_bbox.as_tuple()
    else:
        request = project_bbox

    swapped_better = 0
    for sample in tile_samples:
        normal_overlap = overlap_area(sample, request)
        swapped_overlap = overlap_area(swap_axes_tuple(sample), request)
        if swapped_overlap > normal_overlap:
            swapped_better += 1
    return swapped_better > (len(tile_samples) // 2)


def collect_tile_bbox_samples(
    tile_bboxes_by_year: dict[int, dict[str, list[float]]],
    *,
    max_samples: int = 25,
) -> list[tuple[float, float, float, float]]:
    samples: list[tuple[float, float, float, float]] = []
    for year_map in tile_bboxes_by_year.values():
        for bbox_value in year_map.values():
            if len(bbox_value) != 4:
                continue
            try:
                samples.append(tuple(float(v) for v in bbox_value))
            except (TypeError, ValueError):
                continue
            if len(samples) >= max_samples:
                return samples
    return samples
