# Trajectory Tile Downloader — Design

Date: 2026-06-15
Status: Approved (pending spec review)

## Problem

Given a GPS trajectory (e.g. a glider flight log at
`/home/maciej/Github/sat_test/samolot/gps_001`), we want orthophoto coverage of
the flown path **without** downloading one giant bounding square that is mostly
empty (the track is a thin line crossing tens of km). Instead, tile the area into
a fixed grid and keep **only the windows the track actually crosses**.

## Goals

- Turn a trajectory into a **manifest of grid windows** it intersects — the
  primary, inspectable artifact.
- Optionally **download** the source orthophoto for those windows, reusing the
  existing Geoportal pipeline (no duplicated download logic).
- Be self-contained in `satmap_dataset` and follow repo conventions
  (`run() -> (exit_code, Path)`, manifest JSON contract, exit codes, last stdout
  line = artifact path).

## Non-goals

- No render to a shared NN grid (download-only of source tiles).
- No buffer around the track — only strictly-intersected cells.
- No new heavy dependencies (preview HTML is best-effort via folium if present).

## Decisions (from brainstorming)

| Aspect | Decision |
|--------|----------|
| Output | Manifest of windows **+ built-in download** (second step) |
| Window | **1 km × 1 km**, fixed grid in EPSG:2180 aligned to global origin (multiples of 1000 m) |
| Selection | Only cells the track line **strictly crosses** (no buffer) |
| Years | **2020–2025** (attempt each; skip years missing in Geoportal) |
| Input | Canonical **CSV lat/lon**; plus a thin **IGC** reader so `gps_001` works directly |

## Architecture

Three layers, mirroring the repo's stage pattern:

1. `src/satmap_dataset/trajectory.py` — **pure logic, no network**:
   - `load_track(path) -> list[TrackPoint]`
   - `select_cells(points, cell_m=1000.0, origin=(0.0, 0.0), srs="EPSG:2180") -> list[Cell]`
2. `src/satmap_dataset/pipeline/trajectory.py` — `run(config: TrajectoryConfig) -> tuple[int, Path]`:
   loads track → selects cells → writes manifest → writes preview → optional download.
3. CLI in `cli.py`: `trajectory` (flag form) and `trajectory-json` (JSON config form).

New Pydantic models:
- `config.py`: `TrajectoryConfig`.
- `models.py`: `TrajectoryManifest` (on-disk contract).

## Components

### `load_track(path) -> list[TrackPoint]`
`TrackPoint = (lat: float, lon: float)` (WGS84). Auto-detection:
- **Directory** → find the single `*.igc` inside (error if zero or many).
- **`*.csv`** → read header, locate `lat`/`lon` columns (case-insensitive;
  accept `latitude`/`longitude`). Skip rows that don't parse.
- **`*.igc`** → parse `B` records. Field layout (0-indexed within the record):
  `B`(1) time(6) lat `DDMMmmm`(7) `N/S`(1) lon `DDDMMmmm`(8) `E/W`(1) …
  - `lat = int(d[0:2]) + int(d[2:7]) / 60000.0`, negate for `S`.
  - `lon = int(l[0:3]) + int(l[3:8]) / 60000.0`, negate for `W`.
  Validated against sample `B1039295142136N01750376E…` → (51.7023, 17.8396).

Empty result → caller raises (exit 2).

### `select_cells(points, cell_m, origin, srs) -> list[Cell]`
- Project each point WGS84 → `srs` (EPSG:2180) via pyproj; fall back to the PROJ
  `proj`/`cs2cs` CLI (same dual approach + error message style as `cli.py`).
- **Densify** each consecutive segment in projected meters at a step `< cell_m`
  (e.g. `cell_m / 2`) so sparse CSV inputs can't skip a cell. (1 Hz glider data is
  ~25 m/fix, already finer than the cell — densify just makes it robust.)
- Bin each densified point to `ix = floor((x - ox) / cell_m)`, `iy = floor((y - oy) / cell_m)`.
- Dedup `(ix, iy)`, sort for determinism.
- `Cell` fields: `ix`, `iy`, `bbox_2180=(xmin,ymin,xmax,ymax)` where
  `xmin = ix*cell_m+ox` … `ymax=(iy+1)*cell_m+oy`; `bbox_wgs84`; `center_lat/lon`;
  `name` = `f"{stem}_x{ix}_y{iy}"` (stem from track filename).

### `TrajectoryConfig` (config.py)
- `track_path: Path`
- `cell_km: float = 1.0` (`gt=0`)
- `srs: str = "EPSG:2180"`
- `year_start: int = 2020`, `year_end: int = 2025` (`year_start <= year_end`)
- `download: bool = False`
- `output_dir: Path` (manifest + preview + per-cell downloads root)
- `preview: bool = True`
- Download passthrough subset reused by per-cell configs: `mode="hybrid"`,
  `profile="train"`, `wms_fallback_missing_years`, `concurrency`, `retries`,
  `retry_delay`, `timeout`, `sleep_min`, `sleep_max`, `overwrite`.

### `TrajectoryManifest` (models.py)
`track_path`, `point_count`, `srs`, `cell_m`, `year_start`, `year_end`,
`union_bbox_2180`, `cell_count`, `cells: list[CellEntry]`. Each `CellEntry`:
`name`, `ix`, `iy`, `bbox` (`"xmin,ymin,xmax,ymax"` string, repo convention),
`bbox_wgs84`, `center_lat`, `center_lon`, and (when downloaded) `download_status`
per year (`downloaded` / `missing` / `failed`).

## Data flow

```
track file ──load_track──▶ [(lat,lon)…]
        └─select_cells──▶ project→2180, densify, bin, dedup ──▶ [Cell…]
                                   │
                         write TrajectoryManifest (always)
                         write preview: GeoJSON (always) + HTML (folium if present)
                                   │  if config.download:
                    per Cell × year 2020..2025:
                      IndexConfig(bbox, year, srs, mode) ─index_builder.run─▶ index
                      DownloadConfig(bbox, …)            ─downloader.run────▶ <out>/<cell_name>/
                      idempotent: skip if asset already on disk
```

## Error handling (repo exit-code contract)

- Missing file / no `.igc` (or multiple) in folder / empty track / CSV without
  lat-lon columns → **exit 2** (invalid input).
- Neither pyproj nor PROJ CLI available → error naming both → **exit 2**.
- Download: continue-on-error per cell; record per-cell/year status in manifest.
  `--download` with **zero** successes → **exit 1**; otherwise **exit 0**.
- Manifest is always written before any download attempt.

## CLI

```
python -m satmap_dataset.cli trajectory \
    --track /home/maciej/Github/sat_test/samolot/gps_001 \
    --cell-km 1.0 --year-start 2020 --year-end 2025 \
    --out trajectory_gps001 [--download] [--no-preview]
```
`trajectory-json <config.json>` maps 1:1 onto `TrajectoryConfig`. Last stdout line
is the absolute manifest path. Optional `just` recipe `trajectory-json`.

## Testing (TDD, no network in unit tests)

- IGC `B`-record parse against a small fixture (incl. S/W hemisphere).
- CSV parse: header detection, bad-row skipping, `latitude`/`longitude` aliases.
- Projection + binning: known lat/lon → known `(ix, iy)` and bbox.
- Densification: a sparse 2-point segment spanning several cells yields all
  crossed cells; dedup leaves no duplicates; output is deterministic/sorted.
- Manifest schema round-trips (model_validate ↔ json).
- Orchestration: `pipeline.trajectory.run` with `index_builder`/`downloader`
  monkeypatched — asserts per-cell calls, idempotent skip, exit codes.

## Open assumptions

- Canonical input is CSV lat/lon; IGC support is a convenience so `gps_001` works
  without manual conversion. If IGC is unwanted, drop the IGC branch — no other
  change.
- Grid origin `(0, 0)` in EPSG:2180 so windows are reusable/aligned across flights.
