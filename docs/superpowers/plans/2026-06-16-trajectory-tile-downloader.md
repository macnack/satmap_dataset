# Trajectory Tile Downloader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a GPS trajectory into a manifest of 1 km grid windows it crosses (EPSG:2180) and optionally download the source orthophoto (2020–2025) for each window, reusing the existing Geoportal index/download stages.

**Architecture:** Pure logic (`trajectory.py`: track parsing + grid selection) is separated from the network-bound stage (`pipeline/trajectory.py`: manifest, preview, download orchestration). New Pydantic `TrajectoryConfig`/`TrajectoryManifest` follow the repo's config/manifest conventions. A new CLI command exposes flag and JSON forms.

**Tech Stack:** Python 3.10+, Pydantic v2, pyproj (via `providers/lantmateriet/crs.transform_point`), Typer CLI, pytest.

**Reference spec:** `docs/superpowers/specs/2026-06-15-trajectory-tile-downloader-design.md`

---

### Task 1: Track loading (`load_track`)

**Files:**
- Create: `src/satmap_dataset/trajectory.py`
- Test: `tests/test_trajectory_load.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_trajectory_load.py
from pathlib import Path

import pytest

from satmap_dataset.trajectory import TrackPoint, load_track

SAMPLE_IGC = (
    "ANAV9A1\n"
    "HFDTEDATE:190426,01\n"
    "B1039295142136N01750376EA0011800175\n"  # 51.70227, 17.83960
    "B1039305142200N01750400EA0011800175\n"
    "Lsomethingelse\n"
)


def test_load_igc_parses_b_records(tmp_path: Path):
    p = tmp_path / "track.igc"
    p.write_text(SAMPLE_IGC, encoding="latin-1")
    pts = load_track(p)
    assert len(pts) == 2
    assert pts[0].lat == pytest.approx(51.70227, abs=1e-4)
    assert pts[0].lon == pytest.approx(17.83960, abs=1e-4)


def test_load_igc_southwest_hemisphere(tmp_path: Path):
    p = tmp_path / "s.igc"
    p.write_text("B1039295142136S01750376WA000\n", encoding="latin-1")
    pts = load_track(p)
    assert pts[0].lat < 0 and pts[0].lon < 0


def test_load_dir_autodetects_single_igc(tmp_path: Path):
    (tmp_path / "track.igc").write_text(SAMPLE_IGC, encoding="latin-1")
    pts = load_track(tmp_path)
    assert len(pts) == 2


def test_load_dir_rejects_multiple_igc(tmp_path: Path):
    (tmp_path / "a.igc").write_text(SAMPLE_IGC, encoding="latin-1")
    (tmp_path / "b.igc").write_text(SAMPLE_IGC, encoding="latin-1")
    with pytest.raises(ValueError):
        load_track(tmp_path)


def test_load_csv_lat_lon(tmp_path: Path):
    p = tmp_path / "t.csv"
    p.write_text("lat,lon\n51.5,17.8\n51.6,17.9\n", encoding="utf-8")
    pts = load_track(p)
    assert pts == [TrackPoint(51.5, 17.8), TrackPoint(51.6, 17.9)]


def test_load_csv_latitude_longitude_aliases(tmp_path: Path):
    p = tmp_path / "t.csv"
    p.write_text("time,Latitude,Longitude\n1,51.5,17.8\n", encoding="utf-8")
    pts = load_track(p)
    assert pts == [TrackPoint(51.5, 17.8)]


def test_load_csv_missing_columns_raises(tmp_path: Path):
    p = tmp_path / "t.csv"
    p.write_text("x,y\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_track(p)


def test_load_empty_track_raises(tmp_path: Path):
    p = tmp_path / "t.csv"
    p.write_text("lat,lon\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_track(p)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_trajectory_load.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'satmap_dataset.trajectory'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/satmap_dataset/trajectory.py
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrackPoint:
    lat: float
    lon: float


def load_track(path: Path | str) -> list[TrackPoint]:
    """Load a trajectory as WGS84 lat/lon points from a CSV, an IGC file, or a
    directory containing exactly one ``*.igc``."""
    path = Path(path)
    if path.is_dir():
        igc_files = sorted(path.glob("*.igc"))
        if len(igc_files) != 1:
            raise ValueError(
                f"expected exactly one .igc file in {path}, found {len(igc_files)}"
            )
        path = igc_files[0]
    suffix = path.suffix.lower()
    if suffix == ".igc":
        points = _load_igc(path)
    elif suffix == ".csv":
        points = _load_csv(path)
    else:
        raise ValueError(f"unsupported track format: {path.suffix!r} (use .csv or .igc)")
    if not points:
        raise ValueError(f"no track points parsed from {path}")
    return points


def _load_igc(path: Path) -> list[TrackPoint]:
    points: list[TrackPoint] = []
    for line in path.read_text(encoding="latin-1").splitlines():
        if not line.startswith("B") or len(line) < 24:
            continue
        try:
            lat = int(line[7:9]) + int(line[9:14]) / 60000.0
            if line[14] == "S":
                lat = -lat
            lon = int(line[15:18]) + int(line[18:23]) / 60000.0
            if line[23] == "W":
                lon = -lon
        except ValueError:
            continue
        points.append(TrackPoint(lat=lat, lon=lon))
    return points


def _load_csv(path: Path) -> list[TrackPoint]:
    points: list[TrackPoint] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError(f"empty CSV: {path}")
        lower = {name.lower(): name for name in reader.fieldnames}
        lat_key = lower.get("lat") or lower.get("latitude")
        lon_key = lower.get("lon") or lower.get("longitude")
        if lat_key is None or lon_key is None:
            raise ValueError(
                f"CSV must have lat/lon (or latitude/longitude) columns, got {reader.fieldnames}"
            )
        for row in reader:
            try:
                lat = float(row[lat_key])
                lon = float(row[lon_key])
            except (TypeError, ValueError):
                continue
            points.append(TrackPoint(lat=lat, lon=lon))
    return points
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_trajectory_load.py -q`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/trajectory.py tests/test_trajectory_load.py
git commit -m "feat(trajectory): load track from CSV/IGC/dir"
```

---

### Task 2: Grid cell selection (`select_cells`)

**Files:**
- Modify: `src/satmap_dataset/trajectory.py`
- Test: `tests/test_trajectory_grid.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_trajectory_grid.py
import math

import pytest

from satmap_dataset.trajectory import (
    Cell,
    TrackPoint,
    _cells_from_projected,
    select_cells,
)


def test_single_point_one_cell():
    assert _cells_from_projected([(500.0, 500.0)], cell_m=1000.0, origin=(0.0, 0.0)) == [(0, 0)]


def test_segment_crosses_multiple_cells():
    # densified line from x=500 to x=2500 at y=500 spans cells 0,1,2 in x
    cells = _cells_from_projected(
        [(500.0, 500.0), (2500.0, 500.0)], cell_m=1000.0, origin=(0.0, 0.0)
    )
    assert cells == [(0, 0), (1, 0), (2, 0)]


def test_dedup_and_sorted():
    cells = _cells_from_projected(
        [(100.0, 100.0), (200.0, 200.0), (100.0, 100.0)],
        cell_m=1000.0,
        origin=(0.0, 0.0),
    )
    assert cells == [(0, 0)]


def test_origin_offset_and_negative_index():
    cells = _cells_from_projected([(-1.0, -1.0)], cell_m=1000.0, origin=(0.0, 0.0))
    assert cells == [(-1, -1)]


def test_invalid_cell_size():
    with pytest.raises(ValueError):
        _cells_from_projected([(0.0, 0.0)], cell_m=0.0, origin=(0.0, 0.0))


def test_select_cells_builds_aligned_bbox():
    # Two points near Kepno, PL -> at least one cell, bbox aligned to 1000 m grid.
    pts = [TrackPoint(51.70227, 17.83960), TrackPoint(51.70250, 17.84050)]
    cells = select_cells(pts, cell_m=1000.0, srs="EPSG:2180", name_stem="t")
    assert len(cells) >= 1
    c = cells[0]
    assert isinstance(c, Cell)
    xmin, ymin, xmax, ymax = c.bbox_2180
    assert math.isclose(xmax - xmin, 1000.0)
    assert math.isclose(ymax - ymin, 1000.0)
    assert math.isclose(xmin % 1000.0, 0.0, abs_tol=1e-6)
    assert c.name == f"t_x{c.ix}_y{c.iy}"
    # wgs84 bbox brackets the cell; center lat/lon inside the wgs84 bbox
    wlon0, wlat0, wlon1, wlat1 = c.bbox_wgs84
    assert wlon0 < c.center_lon < wlon1
    assert wlat0 < c.center_lat < wlat1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_trajectory_grid.py -q`
Expected: FAIL with `ImportError: cannot import name 'Cell'` (or `_cells_from_projected`)

- [ ] **Step 3: Write minimal implementation (append to `trajectory.py`)**

```python
# append to src/satmap_dataset/trajectory.py
import math


@dataclass(frozen=True)
class Cell:
    ix: int
    iy: int
    bbox_2180: tuple[float, float, float, float]
    bbox_wgs84: tuple[float, float, float, float]
    center_lat: float
    center_lon: float
    name: str


def _transform(src_crs: str, dst_crs: str, x: float, y: float) -> tuple[float, float]:
    from satmap_dataset.providers.lantmateriet.crs import transform_point

    try:
        return transform_point(src_crs, dst_crs, x, y)
    except Exception as exc:  # noqa: BLE001 - surface both backends
        raise RuntimeError(
            "Trajectory projection requires pyproj or the PROJ 'proj' CLI in PATH."
        ) from exc


def _densify(x0: float, y0: float, x1: float, y1: float, step: float):
    dist = math.hypot(x1 - x0, y1 - y0)
    n = max(1, int(dist // step) + 1)
    for i in range(n + 1):
        t = i / n
        yield x0 + (x1 - x0) * t, y0 + (y1 - y0) * t


def _cells_from_projected(
    projected: list[tuple[float, float]],
    *,
    cell_m: float,
    origin: tuple[float, float],
) -> list[tuple[int, int]]:
    if cell_m <= 0:
        raise ValueError("cell_m must be > 0")
    ox, oy = origin
    seen: set[tuple[int, int]] = set()
    if len(projected) == 1:
        segments = [(projected[0], projected[0])]
    else:
        segments = list(zip(projected, projected[1:]))
    for (x0, y0), (x1, y1) in segments:
        for x, y in _densify(x0, y0, x1, y1, cell_m / 2.0):
            seen.add((math.floor((x - ox) / cell_m), math.floor((y - oy) / cell_m)))
    return sorted(seen)


def select_cells(
    points: list[TrackPoint],
    *,
    cell_m: float = 1000.0,
    origin: tuple[float, float] = (0.0, 0.0),
    srs: str = "EPSG:2180",
    name_stem: str = "track",
) -> list[Cell]:
    """Select the fixed-grid cells a track crosses (no buffer)."""
    ox, oy = origin
    projected = [_transform("EPSG:4326", srs, p.lon, p.lat) for p in points]
    indices = _cells_from_projected(projected, cell_m=cell_m, origin=origin)
    cells: list[Cell] = []
    for ix, iy in indices:
        xmin = ix * cell_m + ox
        ymin = iy * cell_m + oy
        xmax = xmin + cell_m
        ymax = ymin + cell_m
        clon, clat = _transform(srs, "EPSG:4326", (xmin + xmax) / 2.0, (ymin + ymax) / 2.0)
        a_lon, a_lat = _transform(srs, "EPSG:4326", xmin, ymin)
        b_lon, b_lat = _transform(srs, "EPSG:4326", xmax, ymax)
        cells.append(
            Cell(
                ix=ix,
                iy=iy,
                bbox_2180=(xmin, ymin, xmax, ymax),
                bbox_wgs84=(min(a_lon, b_lon), min(a_lat, b_lat), max(a_lon, b_lon), max(a_lat, b_lat)),
                center_lat=clat,
                center_lon=clon,
                name=f"{name_stem}_x{ix}_y{iy}",
            )
        )
    return cells
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_trajectory_grid.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/trajectory.py tests/test_trajectory_grid.py
git commit -m "feat(trajectory): select fixed-grid cells a track crosses"
```

---

### Task 3: Manifest model (`TrajectoryManifest`)

**Files:**
- Modify: `src/satmap_dataset/models.py` (append after existing manifest classes)
- Test: `tests/test_trajectory_manifest.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trajectory_manifest.py
from satmap_dataset.models import CellEntry, TrajectoryManifest


def test_manifest_round_trips():
    m = TrajectoryManifest(
        track_path="gps_001",
        point_count=3204,
        srs="EPSG:2180",
        cell_m=1000.0,
        year_start=2020,
        year_end=2025,
        union_bbox_2180="410000.000,395000.000,443000.000,427000.000",
        cell_count=1,
        cells=[
            CellEntry(
                name="gps_001_x440_y430",
                ix=440,
                iy=430,
                bbox="440000.000,430000.000,441000.000,431000.000",
                bbox_wgs84="17.83,51.70,17.85,51.71",
                center_lat=51.705,
                center_lon=17.84,
            )
        ],
    )
    restored = TrajectoryManifest.model_validate_json(m.model_dump_json())
    assert restored.cell_count == 1
    assert restored.cells[0].name == "gps_001_x440_y430"
    assert restored.cells[0].download_status is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_trajectory_manifest.py -q`
Expected: FAIL with `ImportError: cannot import name 'CellEntry'`

- [ ] **Step 3: Write minimal implementation (append to `models.py`)**

```python
# append to src/satmap_dataset/models.py

class CellEntry(BaseModel):
    name: str
    ix: int
    iy: int
    bbox: str  # "xmin,ymin,xmax,ymax" in `srs` axis order
    bbox_wgs84: str  # "lon_min,lat_min,lon_max,lat_max"
    center_lat: float
    center_lon: float
    download_status: str | None = None  # None|"ok"|"failed"|"skipped"


class TrajectoryManifest(BaseModel):
    track_path: str
    point_count: int = Field(..., ge=0)
    srs: str
    cell_m: float = Field(..., gt=0.0)
    year_start: int = Field(..., ge=1900)
    year_end: int = Field(..., ge=1900)
    union_bbox_2180: str
    cell_count: int = Field(..., ge=0)
    cells: list[CellEntry] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=_utc_now)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_trajectory_manifest.py -q`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/models.py tests/test_trajectory_manifest.py
git commit -m "feat(trajectory): TrajectoryManifest + CellEntry models"
```

---

### Task 4: Config model (`TrajectoryConfig`)

**Files:**
- Modify: `src/satmap_dataset/config.py` (append after `RunConfig`/before `MosaicConfig` or at end)
- Test: `tests/test_trajectory_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trajectory_config.py
import pytest

from satmap_dataset.config import TrajectoryConfig


def test_defaults():
    c = TrajectoryConfig(track_path="gps_001", output_dir="out")
    assert c.cell_km == 1.0
    assert c.year_start == 2020 and c.year_end == 2025
    assert c.srs == "EPSG:2180"
    assert c.download is False and c.preview is True
    assert c.mode == "hybrid" and c.profile == "train"


def test_year_order_validated():
    with pytest.raises(ValueError):
        TrajectoryConfig(track_path="t", output_dir="o", year_start=2025, year_end=2020)


def test_cell_km_positive():
    with pytest.raises(ValueError):
        TrajectoryConfig(track_path="t", output_dir="o", cell_km=0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_trajectory_config.py -q`
Expected: FAIL with `ImportError: cannot import name 'TrajectoryConfig'`

- [ ] **Step 3: Write minimal implementation (append to `config.py`)**

```python
# append to src/satmap_dataset/config.py

class TrajectoryConfig(BaseModel):
    track_path: Path
    output_dir: Path
    cell_km: float = Field(default=1.0, gt=0.0)
    srs: str = "EPSG:2180"
    year_start: int = Field(default=2020, ge=1900)
    year_end: int = Field(default=2025, ge=1900)
    download: bool = False
    preview: bool = True
    mode: str = "hybrid"
    profile: str = "train"
    wms_fallback_missing_years: bool = True
    concurrency: int = Field(default=6, ge=1, le=64)
    retries: int = Field(default=3, ge=0, le=20)
    retry_delay: float = Field(default=1.0, gt=0.0)
    timeout: float = Field(default=120.0, gt=0.0)
    sleep_min: float = Field(default=0.6, ge=0.0)
    sleep_max: float = Field(default=2.2, ge=0.0)
    overwrite: bool = False

    @model_validator(mode="after")
    def _validate(self) -> "TrajectoryConfig":
        if self.year_end < self.year_start:
            raise ValueError("year_end must be >= year_start")
        return self
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_trajectory_config.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/config.py tests/test_trajectory_config.py
git commit -m "feat(trajectory): TrajectoryConfig"
```

---

### Task 5: Stage `run()` — manifest + preview (no download)

**Files:**
- Create: `src/satmap_dataset/pipeline/trajectory.py`
- Test: `tests/test_trajectory_pipeline.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_trajectory_pipeline.py
import json
from pathlib import Path

from satmap_dataset.config import TrajectoryConfig
from satmap_dataset.models import TrajectoryManifest
from satmap_dataset.pipeline import trajectory as traj_stage


def _write_csv(tmp_path: Path) -> Path:
    p = tmp_path / "track.csv"
    p.write_text(
        "lat,lon\n51.70227,17.83960\n51.70250,17.84050\n51.70300,17.84200\n",
        encoding="utf-8",
    )
    return p


def test_run_writes_manifest_and_preview(tmp_path: Path):
    csv_path = _write_csv(tmp_path)
    out = tmp_path / "out"
    cfg = TrajectoryConfig(track_path=csv_path, output_dir=out, download=False)
    code, path = traj_stage.run(cfg)
    assert code == 0
    assert path == out / "trajectory_tiles.json"
    assert path.exists()
    manifest = TrajectoryManifest.model_validate_json(path.read_text())
    assert manifest.point_count == 3
    assert manifest.cell_count >= 1
    assert manifest.cells[0].name.startswith("track_x")
    # preview geojson written
    gj = json.loads((out / "trajectory_tiles.geojson").read_text())
    assert gj["type"] == "FeatureCollection"
    assert any(f["geometry"]["type"] == "LineString" for f in gj["features"])
    assert any(f["geometry"]["type"] == "Polygon" for f in gj["features"])


def test_run_no_preview(tmp_path: Path):
    csv_path = _write_csv(tmp_path)
    out = tmp_path / "out"
    cfg = TrajectoryConfig(track_path=csv_path, output_dir=out, preview=False)
    code, _ = traj_stage.run(cfg)
    assert code == 0
    assert not (out / "trajectory_tiles.geojson").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_trajectory_pipeline.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'satmap_dataset.pipeline.trajectory'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/satmap_dataset/pipeline/trajectory.py
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
    config: TrajectoryConfig, points: list[TrackPoint], cells: list[Cell]
) -> TrajectoryManifest:
    cell_m = config.cell_km * 1000.0
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
    manifest = _build_manifest(config, points, cells)
    manifest_path = output_dir / "trajectory_tiles.json"

    if config.preview:
        _write_geojson(output_dir / "trajectory_tiles.geojson", points, cells)

    if config.download:
        ok = _download_cells(config, cells, manifest, output_dir)
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return (0 if ok else 1), manifest_path

    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return 0, manifest_path


def _download_cells(config, cells, manifest, output_dir) -> bool:  # noqa: ANN001
    # Implemented in Task 6.
    return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_trajectory_pipeline.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/pipeline/trajectory.py tests/test_trajectory_pipeline.py
git commit -m "feat(trajectory): stage run() writes manifest + geojson preview"
```

---

### Task 6: Download integration (`_download_cells`)

**Files:**
- Modify: `src/satmap_dataset/pipeline/trajectory.py`
- Test: `tests/test_trajectory_download.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_trajectory_download.py
from pathlib import Path

from satmap_dataset.config import TrajectoryConfig
from satmap_dataset.models import TrajectoryManifest
from satmap_dataset.pipeline import trajectory as traj_stage


def _csv(tmp_path: Path) -> Path:
    p = tmp_path / "track.csv"
    p.write_text("lat,lon\n51.70227,17.83960\n51.70250,17.84050\n", encoding="utf-8")
    return p


def test_download_invokes_stages_per_cell(tmp_path: Path, monkeypatch):
    calls = {"index": [], "download": []}

    def fake_index_run(cfg):
        calls["index"].append(cfg.bbox)
        out = Path(cfg.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{}", encoding="utf-8")
        return 0, out

    def fake_download_run(cfg):
        calls["download"].append(cfg.bbox)
        out = Path(cfg.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{}", encoding="utf-8")
        return 0, out

    monkeypatch.setattr(traj_stage.index_builder, "run", fake_index_run)
    monkeypatch.setattr(traj_stage.downloader, "run", fake_download_run)

    out = tmp_path / "out"
    cfg = TrajectoryConfig(track_path=_csv(tmp_path), output_dir=out, download=True)
    code, path = traj_stage.run(cfg)
    assert code == 0
    manifest = TrajectoryManifest.model_validate_json(path.read_text())
    n = manifest.cell_count
    assert len(calls["index"]) == n
    assert len(calls["download"]) == n
    assert all(c.download_status == "ok" for c in manifest.cells)


def test_download_failure_marks_cell_and_exit_1(tmp_path: Path, monkeypatch):
    def fake_index_run(cfg):
        out = Path(cfg.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{}", encoding="utf-8")
        return 0, out

    def fail_download_run(cfg):
        return 1, Path(cfg.output_json)

    monkeypatch.setattr(traj_stage.index_builder, "run", fake_index_run)
    monkeypatch.setattr(traj_stage.downloader, "run", fail_download_run)

    out = tmp_path / "out"
    cfg = TrajectoryConfig(track_path=_csv(tmp_path), output_dir=out, download=True)
    code, path = traj_stage.run(cfg)
    assert code == 1
    manifest = TrajectoryManifest.model_validate_json(path.read_text())
    assert all(c.download_status == "failed" for c in manifest.cells)


def test_download_idempotent_skip(tmp_path: Path, monkeypatch):
    out = tmp_path / "out"
    # Pre-create the per-cell download manifest so the cell is skipped.
    cfg0 = TrajectoryConfig(track_path=_csv(tmp_path), output_dir=out, download=False)
    code0, path0 = traj_stage.run(cfg0)
    from satmap_dataset.models import TrajectoryManifest as TM

    name = TM.model_validate_json(path0.read_text()).cells[0].name
    cell_dir = out / name
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / "dataset_manifest_download.json").write_text("{}", encoding="utf-8")

    called = {"download": 0}

    def fake_index_run(cfg):
        o = Path(cfg.output_json)
        o.parent.mkdir(parents=True, exist_ok=True)
        o.write_text("{}", encoding="utf-8")
        return 0, o

    def fake_download_run(cfg):
        called["download"] += 1
        return 0, Path(cfg.output_json)

    monkeypatch.setattr(traj_stage.index_builder, "run", fake_index_run)
    monkeypatch.setattr(traj_stage.downloader, "run", fake_download_run)

    cfg = TrajectoryConfig(track_path=_csv(tmp_path), output_dir=out, download=True)
    code, path = traj_stage.run(cfg)
    assert code == 0
    manifest = TrajectoryManifest.model_validate_json(path.read_text())
    # First cell (pre-seeded) skipped; only the remaining cells trigger a download.
    skipped = [c for c in manifest.cells if c.download_status == "skipped"]
    assert any(c.name == name for c in skipped)
    assert called["download"] == manifest.cell_count - len(skipped)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_trajectory_download.py -q`
Expected: FAIL (`_download_cells` is a stub returning True; assertions on calls/status fail)

- [ ] **Step 3: Write minimal implementation**

Replace the import block and the `_download_cells` stub in `src/satmap_dataset/pipeline/trajectory.py`:

```python
# add to the imports at the top of src/satmap_dataset/pipeline/trajectory.py
from satmap_dataset.config import DownloadConfig, IndexConfig, TrajectoryConfig
from satmap_dataset.pipeline import downloader, index_builder
```

```python
# replace the _download_cells stub
def _download_cells(
    config: TrajectoryConfig,
    cells: list[Cell],
    manifest: TrajectoryManifest,
    output_dir: Path,
) -> bool:
    any_ok = False
    for cell, entry in zip(cells, manifest.cells):
        cell_dir = output_dir / cell.name
        download_manifest = cell_dir / "dataset_manifest_download.json"
        if download_manifest.exists() and not config.overwrite:
            entry.download_status = "skipped"
            any_ok = True
            logger.info("Trajectory: skip existing cell=%s", cell.name)
            continue
        cell_dir.mkdir(parents=True, exist_ok=True)
        bbox = entry.bbox
        index_config = IndexConfig(
            year_start=config.year_start,
            year_end=config.year_end,
            bbox=bbox,
            srs=config.srs,
            output_json=cell_dir / "index_manifest.json",
            year_availability_output_json=cell_dir / "year_availability_report.json",
        )
        index_code, index_path = index_builder.run(index_config)
        if index_code != 0:
            entry.download_status = "failed"
            logger.error("Trajectory: index failed cell=%s", cell.name)
            continue
        download_config = DownloadConfig(
            index_manifest=index_path,
            download_root=cell_dir / "downloads",
            mode=config.mode,
            profile=config.profile,
            bbox=bbox,
            srs=config.srs,
            wms_fallback_missing_years=config.wms_fallback_missing_years,
            concurrency=config.concurrency,
            retries=config.retries,
            retry_delay=config.retry_delay,
            timeout=config.timeout,
            sleep_min=config.sleep_min,
            sleep_max=config.sleep_max,
            overwrite=config.overwrite,
            output_json=download_manifest,
        )
        download_code, _ = downloader.run(download_config)
        if download_code != 0:
            entry.download_status = "failed"
            logger.error("Trajectory: download failed cell=%s", cell.name)
            continue
        entry.download_status = "ok"
        any_ok = True
    return any_ok
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_trajectory_download.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/pipeline/trajectory.py tests/test_trajectory_download.py
git commit -m "feat(trajectory): per-cell index+download with idempotent skip"
```

---

### Task 7: CLI commands (`trajectory`, `trajectory-json`)

**Files:**
- Modify: `src/satmap_dataset/cli.py`
- Test: `tests/test_trajectory_cli.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_trajectory_cli.py
import json
from pathlib import Path

from typer.testing import CliRunner

from satmap_dataset.cli import app

runner = CliRunner()


def _csv(tmp_path: Path) -> Path:
    p = tmp_path / "track.csv"
    p.write_text("lat,lon\n51.70227,17.83960\n51.70250,17.84050\n", encoding="utf-8")
    return p


def test_trajectory_flag_form(tmp_path: Path):
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        ["trajectory", "--track", str(_csv(tmp_path)), "--out", str(out), "--cell-km", "1.0"],
    )
    assert result.exit_code == 0, result.output
    manifest_path = out / "trajectory_tiles.json"
    assert manifest_path.exists()
    # last stdout line is the artifact path (repo contract)
    last_line = result.output.strip().splitlines()[-1]
    assert last_line.endswith("trajectory_tiles.json")


def test_trajectory_json_form(tmp_path: Path):
    out = tmp_path / "out"
    cfg = tmp_path / "cfg.json"
    cfg.write_text(
        json.dumps(
            {"track_path": str(_csv(tmp_path)), "output_dir": str(out), "download": False}
        ),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["trajectory-json", str(cfg)])
    assert result.exit_code == 0, result.output
    assert (out / "trajectory_tiles.json").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_trajectory_cli.py -q`
Expected: FAIL (`No such command 'trajectory'`)

- [ ] **Step 3: Write minimal implementation**

First confirm the Typer app object name and import style:

Run: `grep -n "^app = typer\|app = typer.Typer\|from satmap_dataset.pipeline import\|@app.command" src/satmap_dataset/cli.py | head`

Then append two commands to `src/satmap_dataset/cli.py` (use the existing module-level `app`, `console`, `typer`, and `Path` already imported there):

```python
# append near the other @app.command() definitions in src/satmap_dataset/cli.py
@app.command("trajectory")
def trajectory_cmd(
    track: Path = typer.Option(..., "--track", help="Track file (.csv/.igc) or a directory with one .igc."),
    out: Path = typer.Option(..., "--out", help="Output directory for the manifest, preview, and downloads."),
    cell_km: float = typer.Option(1.0, "--cell-km", min=0.0001, help="Grid cell size in km."),
    year_start: int = typer.Option(2020, "--year-start"),
    year_end: int = typer.Option(2025, "--year-end"),
    download: bool = typer.Option(False, "--download/--no-download", help="Download source orthophoto for each window."),
    preview: bool = typer.Option(True, "--preview/--no-preview", help="Write a GeoJSON preview."),
) -> None:
    from satmap_dataset.config import TrajectoryConfig
    from satmap_dataset.pipeline import trajectory as trajectory_stage

    config = TrajectoryConfig(
        track_path=track,
        output_dir=out,
        cell_km=cell_km,
        year_start=year_start,
        year_end=year_end,
        download=download,
        preview=preview,
    )
    try:
        code, path = trajectory_stage.run(config)
    except (ValueError, RuntimeError) as exc:  # invalid input / no projection backend
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)
    console.print(str(path.resolve()))
    raise typer.Exit(code)


@app.command("trajectory-json")
def trajectory_json_cmd(
    config_json: Path = typer.Argument(..., help="JSON file mapped 1:1 onto TrajectoryConfig."),
) -> None:
    import json as _json

    from satmap_dataset.config import TrajectoryConfig
    from satmap_dataset.pipeline import trajectory as trajectory_stage

    payload = _json.loads(Path(config_json).read_text(encoding="utf-8"))
    try:
        config = TrajectoryConfig(**payload)
        code, path = trajectory_stage.run(config)
    except (ValueError, RuntimeError) as exc:  # invalid config/input
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)
    console.print(str(path.resolve()))
    raise typer.Exit(code)
```

Note: if `grep` shows the artifact path is emitted with `typer.echo`/`print` elsewhere, match that style instead of `console.print` so the path lands on stdout as the last line.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_trajectory_cli.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/cli.py tests/test_trajectory_cli.py
git commit -m "feat(trajectory): CLI trajectory and trajectory-json commands"
```

---

### Task 8: Full suite + just recipe + README note

**Files:**
- Modify: `justfile`
- Modify: `configs/run/README.md` (or `README.md` if that's where CLI usage lives — confirm with grep)

- [ ] **Step 1: Run the full test suite**

Run: `pytest -q`
Expected: PASS (all existing tests + the new trajectory tests). If anything unrelated fails, stop and report — do not "fix" by editing tests.

- [ ] **Step 2: Add a `just` recipe**

Run: `grep -n "run-location-json\|^run-json\|cli " justfile | head`
Then add (match the surrounding recipe style; this is the typical form):

```make
# Trajectory -> grid windows manifest (+ optional download with download=1)
trajectory track out cell_km="1.0" download="0":
    python -m satmap_dataset.cli trajectory --track {{track}} --out {{out}} --cell-km {{cell_km}} {{ if download == "1" { "--download" } else { "--no-download" } }}
```

If your `just` version doesn't support inline conditionals, use two recipes (`trajectory` / `trajectory-download`) instead — verify with `just --evaluate` or `just trajectory --help` style dry checks.

- [ ] **Step 3: Document usage**

Add to the CLI usage docs (location confirmed via grep in Step 2 of Task 7 / repo README):

```markdown
## Trajectory tile downloader

Select the 1 km grid windows a GPS track crosses (EPSG:2180) and optionally
download source orthophoto (2020–2025) for each:

```bash
# manifest + preview only
python -m satmap_dataset.cli trajectory \
    --track /home/maciej/Github/sat_test/samolot/gps_001 \
    --out trajectory_gps001

# + download each window
python -m satmap_dataset.cli trajectory \
    --track /home/maciej/Github/sat_test/samolot/gps_001 \
    --out trajectory_gps001 --download
```
```

- [ ] **Step 4: Commit**

```bash
git add justfile configs/run/README.md
git commit -m "docs(trajectory): just recipe + usage notes"
```

---

## Self-Review

**Spec coverage:**
- Manifest of crossed 1 km windows → Tasks 2, 3, 5. ✓
- Built-in download (second step) → Task 6; CLI `--download` → Task 7. ✓
- Fixed grid EPSG:2180 aligned to origin (0,0) → Task 2 (`_cells_from_projected`). ✓
- Only strictly-crossed cells, no buffer → Task 2 (densify + bin, no dilation). ✓
- Years 2020–2025 → Task 4 defaults; Task 6 passes range to `IndexConfig`/`DownloadConfig`. ✓
- Input CSV lat/lon + IGC + dir autodetect → Task 1. ✓
- Preview (GeoJSON always; HTML folium optional) → Task 5 writes GeoJSON. **Deviation:** HTML/folium preview is omitted to avoid a new dependency; GeoJSON opens in QGIS/geojson.io. Acceptable per spec's "best-effort, no new heavy deps". ✓
- Exit codes (2 invalid input, 1 zero download successes, 0 otherwise) → input errors raise `ValueError` (Task 1) surfacing non-zero via CLI; download exit 1 → Task 6/Task 5. **Note:** input `ValueError` currently propagates as a traceback/exit 1 from Typer, not a clean exit 2. If a clean exit-2 is required, wrap `run()` body in the CLI commands with `try/except (ValueError, RuntimeError) as e: console.print(f"[red]{e}[/red]"); raise typer.Exit(2)`. Add this in Task 7 Step 3.
- `download_status` per year (spec) vs per cell (plan) → **Deviation:** stored per cell (`"ok"/"failed"/"skipped"`), since the per-cell download runs the whole year range in one stage call. Per-year detail remains available inside each cell's `dataset_manifest_download.json`. ✓

**Placeholder scan:** No TBD/TODO; the Task 5 `_download_cells` stub is explicitly replaced in Task 6. ✓

**Type consistency:** `Cell`, `TrackPoint`, `CellEntry`, `TrajectoryManifest`, `TrajectoryConfig`, `select_cells`, `_cells_from_projected`, `load_track`, `run`, `_download_cells` names match across tasks. `index_builder.run`/`downloader.run` return `(int, Path)` matching usage. ✓

**Action item folded in:** Task 7 Step 3 must include the `try/except → Exit(2)` wrapper for clean invalid-input exit codes.
