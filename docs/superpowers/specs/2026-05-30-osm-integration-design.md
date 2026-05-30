# OSM Stage Integration into Location Pipeline — Design

**Date:** 2026-05-30
**Status:** Approved (brainstorming)
**Topic:** Wire the OSM semantic-label stage into `run-all-location-json` so OSM masks are fetched per-location alongside the orthophoto pipeline, using a reliable OSM-API backend that actually works in this environment.

---

## Problem

The OSM stage (`pipeline/osm.py`) exists with CLI commands (`osm-location-json`, etc.) but:

1. It is **not wired into the location orchestration** — `run-all-location-json` runs only `index→download→render→validate`. OSM must be run manually as a separate command.
2. Its committed backend (`overpass_client.py`) is **rate-limited / times out** in this environment and never delivers data for non-trivial areas.
3. A working approach was proven manually: download the OSM `/api/0.6/map` extract (split into quadrants when the node limit is hit), parse ways with `timestamp`+`version`, reconstruct historical state per snapshot date by a timestamp heuristic, and rasterize to the render grid. This works reliably but lives in a throwaway script.

Goal: make OSM masks a first-class, reliable part of the per-location run, fetched together with the location config, with exact-history (Overpass) deferred as a documented future extension.

---

## Decisions (from brainstorming)

1. **Orchestration placement:** OSM runs as a **separate, isolated step** inside `run-all-location-json`, *after* `run_all.run()` succeeds — not as a 5th stage inside `run_all.run()`. An OSM failure is recorded in the run's failure list but **does not invalidate the finished orthophoto artifacts**.
2. **Backend now:** OSM API `/api/0.6/map` + timestamp-heuristic historical filter (the proven approach). Reliable, fast, works everywhere. History is **approximate** (exact only for features untouched since the snapshot date).
3. **Backend future:** Overpass `[date:"…"]` (exact history) is kept behind a config seam as a documented future extension, not implemented now.
4. **Default on:** `fetch_osm` defaults to `True` in `base.json`; OSM leci razem z lokalizacją.

---

## Architecture

```
run-all-location-json (orchestrator CLI)
  └─ for each location JSON:
       config = _build_run_config_from_base_and_location(...)
       if validation artifact already passed: skip
       exit_code, _ = run_all.run(config)         # index→download→render→validate (UNCHANGED)
       if exit_code != 0: record failure; continue/raise
       # NEW — isolated OSM step, only when orthos OK and fetch_osm enabled
       if fetch_osm:
           try:
               osm_cfg = _build_osm_config_from_base_and_location(base_json, location_json)
               osm_code, osm_path = osm_pipeline.run(osm_cfg)
               if osm_code != 0: record failure (osm)
           except Exception as e:
               record failure (osm)               # NEVER re-raised past the location loop
```

The OSM step depends on the render manifest that `run_all.run()` produced:
- grid: `target_bbox`, `target_width`, `target_height`
- snapshot dates: `tile_acquisition_by_year` → one ISO date per year

Both are already read by `pipeline/osm.py` via `_read_year_date_map()` / `_read_grid()`.

---

## Components

### New: `src/satmap_dataset/osm/osm_api_client.py`

Ports the proven script logic. Pure, testable functions:

- `bbox_epsg2180_to_wgs84(bbox_2180: str) -> str` — moved here (from `ohsome_client.py`); returns `lon_min,lat_min,lon_max,lat_max`.
- `async fetch_osm_xml(bbox_wgs84: str, *, timeout, retry_policy) -> bytes` — GET `https://api.openstreetmap.org/api/0.6/map?bbox=…`. On HTTP 400 (node-limit), **recursively split** the bbox into quadrants and concatenate results. Returns merged raw XML (or a merged element set).
- `parse_ways(xml_chunks) -> tuple[dict[node_id,(lon,lat)], list[Way]]` — parse nodes + ways; each `Way` carries `tags`, `coords`, `ts` (datetime), `ver` (int). Deduplicate ways by id across quadrants.
- `existed_at(way, date_str) -> bool` — historical heuristic: `way.ts <= target OR way.ver > 1`.
- `CATEGORY_TAGS: dict[str, Callable[[dict], bool]]` — tag predicates for the 5 categories (buildings, roads, paths, green, water), matching the current semantic definitions.
- `ways_to_geojson(ways) -> dict` — Polygon when ring closed (≥4 pts, first==last), else LineString.
- `async fetch_and_parse(bbox_wgs84, *, timeout, retry_policy) -> list[Way]` — download (adaptive quadrants) + parse, **once per location**.
- `features_for(ways, category, snapshot_date) -> geojson` — pure: filter the parsed ways by `CATEGORY_TAGS[category]` and `existed_at(snapshot_date)`, return GeoJSON. Called per (year, category) over the already-parsed way-set.

Node-limit detail: OSM `/map` 400s on dense AOIs (Poznań needed 4 quadrants). Splitting is adaptive/recursive so it scales to any density without a hard-coded quadrant count.

### Changed: `src/satmap_dataset/pipeline/osm.py`

- Branch once on `config.backend`:
  - `"osm_api"` → `fetch_and_parse(bbox_wgs84)` once, then `features_for(ways, cat, snapshot_date)` per (year, category).
  - `"overpass"` → `raise NotImplementedError("overpass backend is a future extension; use backend='osm_api'")`.
- Keep the existing per-year × per-category loop, reuse-existing-raster logic, zero-features → `raster_path=None`, and manifest writing.
- Optimization: the OSM `/map` download returns the **current** state and is identical for every year, so download+parse **once per location** and reuse the parsed way-set across all years; only `existed_at(way, snapshot_date)` differs per year. This replaces the current one-request-per-(year, category) pattern.
- Rasters burn `255` (already fixed).

### Changed: `src/satmap_dataset/config.py` — `OsmConfig`

- Add `backend: str = "osm_api"` with a validator allowing `{"osm_api", "overpass"}`.
- Add `osm_render_preview: bool = True` (controls the per-year PNG overlay).
- `overpass_url` stays (used only by the future backend).

### Changed: `src/satmap_dataset/cli.py`

- `run-all-location-json`: add the isolated OSM step described above, gated on a `fetch_osm` flag read from the merged base+location dict (default `True`), plus a CLI toggle `--osm/--no-osm`.
- Reuse: OSM step is skipped when `<osm_root>/osm_manifest.json` exists with `passed=true`, unless `overwrite=true` — mirroring the orthophoto skip-existing behavior.

### Preview (optional): per-year `viz_<year>.png`

When `osm_render_preview` is true, after rasterizing a year, render a downscaled overlay of the 5 category masks on that year's ortho (`rendered_<slug>/year_<year>.tif`), saved to `<osm_root>/viz_<year>.png`. Reuses the color scheme from the manual viz. Skipped silently if the ortho for that year is absent.

### Retained for the future: `overpass_client.py`

Kept as the skeleton of the future exact-history backend (it already emits `[date:]` queries). `ohsome_client.py` is reduced/removed once `bbox_epsg2180_to_wgs84` moves to `osm_api_client.py`.

---

## Data Flow

1. `run_all.run()` writes `artifacts_<slug>/dataset_manifest_render.json` (grid + acquisition dates) and `rendered_<slug>/year_<YYYY>.tif`.
2. `_build_osm_config_from_base_and_location` sets `render_manifest` → that render manifest, `osm_root` → `osm_<slug>`, `output_json` → `osm_<slug>/osm_manifest.json`.
3. `osm_pipeline.run()`:
   - reads `year_date_map` (acquisition date per year) and grid from the render manifest;
   - converts bbox to WGS84;
   - downloads + parses the OSM extent **once** (adaptive quadrants);
   - for each year: apply `existed_at(snapshot_date)` per category over the parsed ways → rasterize each non-empty category to `osm_<slug>/year_<YYYY>_<cat>.tif` (burn 255, EPSG:2180, render grid);
   - writes `osm_manifest.json` (per-year per-category `feature_count` + `raster_path|null`);
   - optionally writes `viz_<YYYY>.png`.

---

## Error Handling

- **OSM step isolation:** the OSM call in `run-all-location-json` is wrapped so any exception or non-zero exit is appended to `failures` and never propagated past the per-location iteration. Orthophoto artifacts remain valid.
- **Zero features for a year/category:** valid, not an error — `raster_path=null` in the manifest, no file written (existing behavior, ML constraint "better empty than wrong").
- **HTTP 400 node-limit:** handled internally by recursive quadrant split, not surfaced as failure.
- **Missing render manifest / no acquisition dates:** `osm_pipeline.run()` returns exit code 1 with an error in the manifest (existing behavior); orchestrator records it as an OSM failure.
- **`backend="overpass"`:** explicit `NotImplementedError` with a clear "future extension" message.

---

## Testing

- `osm_api_client`: `bbox_epsg2180_to_wgs84` known values; `parse_ways` from a small fixture XML (nodes+ways, with `timestamp`/`version`); `existed_at` truth table; `CATEGORY_TAGS` predicates; `ways_to_geojson` polygon-vs-linestring; recursive split triggered on a mocked 400 (fetch mocked — no live network).
- `pipeline/osm.py`: backend branch — `osm_api` path with mocked `fetch_and_parse`/`features_for`; `overpass` path raises `NotImplementedError`; year/category loop, zero-feature null raster, reuse-existing-raster, manifest contents (extend existing pipeline tests; swap mock seam from `overpass_client` to `osm_api_client`).
- `config`: `backend` validator accepts `osm_api`/`overpass`, rejects others; `osm_render_preview` default.
- `cli`: `run-all-location-json` invokes OSM step when `fetch_osm` true (mock `osm_pipeline.run`), skips when false; OSM failure recorded but orthophoto exit path unchanged; `--no-osm` disables.

All network and GDAL calls are mocked in tests (consistent with existing OSM/DEM test style). Live verification (real OSM API) stays a manual smoke step.

---

## Out of Scope (Future Extensions)

- **Overpass `[date:]` exact-history backend** — seam exists (`backend="overpass"`), implementation deferred.
- **osmium full-history extract** — not added; would be a third backend value if exact offline history is later required.
- **DEM into `run-all-location-json`** — the same isolated-step pattern applies and can be added later; not part of this change.
