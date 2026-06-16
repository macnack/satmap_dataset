from __future__ import annotations

import json
import logging
from pathlib import Path

from satmap_dataset.config import TrajectoryConfig
from satmap_dataset.models import CellEntry, TrajectoryManifest
from satmap_dataset.trajectory import Cell, TrackPoint, load_track, select_cells

logger = logging.getLogger(__name__)


def _bbox_str(bbox: tuple[float, float, float, float]) -> str:
    return ",".join(f"{v:.3f}" for v in bbox)


def _name_stem(track_path: Path) -> str:
    return track_path.name if track_path.is_dir() else track_path.stem


def _union_bbox(cells: list[Cell]) -> str:
    xmin = min(c.bbox_2180[0] for c in cells)
    ymin = min(c.bbox_2180[1] for c in cells)
    xmax = max(c.bbox_2180[2] for c in cells)
    ymax = max(c.bbox_2180[3] for c in cells)
    return _bbox_str((xmin, ymin, xmax, ymax))


def _build_manifest(
    config: TrajectoryConfig,
    points: list[TrackPoint],
    cells: list[Cell],
    cell_m: float,
) -> TrajectoryManifest:
    entries = [
        CellEntry(
            name=c.name,
            ix=c.ix,
            iy=c.iy,
            bbox=_bbox_str(c.bbox_2180),
            bbox_wgs84=_bbox_str(c.bbox_wgs84),
            center_lat=c.center_lat,
            center_lon=c.center_lon,
        )
        for c in cells
    ]
    return TrajectoryManifest(
        track_path=str(config.track_path),
        point_count=len(points),
        srs=config.srs,
        cell_m=cell_m,
        year_start=config.year_start,
        year_end=config.year_end,
        union_bbox_2180=_union_bbox(cells) if cells else "0,0,0,0",
        cell_count=len(cells),
        cells=entries,
    )


def _write_geojson(path: Path, points: list[TrackPoint], cells: list[Cell]) -> None:
    features: list[dict] = [
        {
            "type": "Feature",
            "properties": {"kind": "track"},
            "geometry": {
                "type": "LineString",
                "coordinates": [[p.lon, p.lat] for p in points],
            },
        }
    ]
    for c in cells:
        lon0, lat0, lon1, lat1 = c.bbox_wgs84
        features.append(
            {
                "type": "Feature",
                "properties": {"kind": "cell", "name": c.name},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1], [lon0, lat0]]
                    ],
                },
            }
        )
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2),
        encoding="utf-8",
    )


def run(config: TrajectoryConfig) -> tuple[int, Path]:
    track_path = Path(config.track_path)
    points = load_track(track_path)
    cell_m = config.cell_km * 1000.0
    cells = select_cells(
        points, cell_m=cell_m, srs=config.srs, name_stem=_name_stem(track_path)
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _build_manifest(config, points, cells, cell_m)
    manifest_path = output_dir / "trajectory_tiles.json"

    ok = True
    if config.download:
        ok = _download_cells(config, cells, manifest, output_dir)

    # Write manifest before the optional GeoJSON sidecar so the primary artifact
    # is always committed even if the preview write fails.
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    if config.preview:
        _write_geojson(output_dir / "trajectory_tiles.geojson", points, cells)

    return (0 if ok else 1), manifest_path


def _download_cells(
    config: TrajectoryConfig,
    cells: list[Cell],
    manifest: TrajectoryManifest,
    output_dir: Path,
) -> bool:
    # NOTE: To update per-cell download outcomes, mutate manifest.cells[i].download_status
    # (CellEntry is a mutable Pydantic model). Do NOT attempt to mutate `cells` entries
    # — Cell is a frozen dataclass and has no download_status field.
    # Implemented in Task 6.
    return True
