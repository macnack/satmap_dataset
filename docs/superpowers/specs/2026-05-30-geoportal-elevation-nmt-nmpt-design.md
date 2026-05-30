# Design: ISOK elevation (NMT/NMPT) download via Geoportal WCS

**Date:** 2026-05-30
**Status:** Approved (pending spec review)

## Goal

Add downloading of Polish ISOK 1 m ALS-derived elevation data to `satmap_dataset`,
alongside the existing year-aware orthophoto pipeline:

- **NMT (DTM)** — bare-earth Digital Terrain Model, 1 m grid, mean height error ≤ 0.2 m.
- **NMPT (DSM)** — Digital Surface Model including buildings and vegetation, 1 m grid.

Both are far higher resolution than global SRTM (~30 m) and come from the GUGiK ISOK
project. Data is fetched via **WCS GetCoverage** (server-side clip to AOI), output as
**float32 EPSG:2180 GeoTIFF** in two forms: native 1 m and an ortho-render-aligned copy.

## Decisions (from brainstorming)

| Decision | Choice |
|----------|--------|
| Integration | Extend the geoportal provider (geoportal-scoped, reuse `geoportal/http.py`) |
| Transport | WCS `GetCoverage`, clip to bbox, tile + merge for large AOIs |
| Products | Both NMT (DTM) and NMPT (DSM) |
| Vertical datum | Both selectable; default **EVRF2007-NH** |
| Grid/output | Both — native 1 m EPSG:2180 **and** an ortho-render-aligned resampled copy |

## External services

GUGiK WCS endpoints (WCS 2.0.1, `format=image/tiff`, subset X/Y in EPSG:2180):

- **NMT (DTM):** `https://mapy.geoportal.gov.pl/wss/service/PZGIK/NMT/GRID1/WCS/DigitalTerrainModelFormatTIFF`
  - coverages: `DTM_PL-EVRF2007-NH_TIFF`, `DTM_PL-KRON86-NH_TIFF`
- **NMPT (DSM):** `https://mapy.geoportal.gov.pl/wss/service/PZGIK/NMPT/GRID1/WCS/DigitalSurfaceModelFormatTIFF`
  - coverages: `DSM_PL-EVRF2007-NH_TIFF`, `DSM_PL-KRON86-NH_TIFF`

Endpoint base URLs and coverage-id templates are overridable via `provider_options` /
env vars (mirroring the sentinel2 provider's `_option` pattern), so a change in GUGiK's
naming does not require a code edit.

The WCS GRID1 services serve a **current-best 1 m composite**, not a per-year time
series. Each `GetCoverage` response is already a fully georeferenced EPSG:2180 float32
GeoTIFF with a nodata value, so:

- A single-request AOI (within the server pixel cap) needs **no merge**.
- A tiled AOI merges cleanly because every tile is independently georeferenced.

The endpoints are rate-sensitive like the rest of Geoportal; all requests go through
`geoportal/http.py` (`RetryPolicy` + randomized pre-request jitter). The development
sandbox blocks these endpoints (returns `Unauthorized.` HTML); live verification happens
outside the sandbox.

## Components

### 1. `src/satmap_dataset/geoportal/wcs_client.py`

- `coverage_id(product, datum) -> str` and `endpoint_url(product) -> str` — map
  `product ∈ {nmt, nmpt}` + `datum ∈ {evrf2007, kron86}` to coverage id + endpoint,
  honoring `provider_options`/env overrides.
- `split_bbox(bbox, max_request_px, gsd_m=1.0) -> list[bbox]` — pure function splitting
  an AOI into non-overlapping sub-bboxes each ≤ `max_request_px` per side at the native
  1 m grid. Unit-tested without network.
- `async get_coverage(endpoint, coverage_id, sub_bbox, srs, *, timeout, retry_policy)
  -> bytes` — issue WCS 2.0.1 `GetCoverage` and return GeoTIFF bytes. Uses
  `request_with_retry` from `geoportal/http.py`.
- Optional `describe_coverage(...)` to capture coverage metadata (extent, nodata) for the
  manifest; best-effort, failure is non-fatal.

### 2. `src/satmap_dataset/pipeline/dem.py`

`run(config: DemConfig) -> tuple[int, Path]` (matches the stage `run()` contract):

For each requested product:
1. `split_bbox` → fetch each sub-bbox via `wcs_client.get_coverage`, writing temp tiles.
2. **Merge** tiles → `dem_<slug>/native/{product}_{datum}.tif` (float32, EPSG:2180,
   nodata preserved). Single-tile AOIs skip merge.
3. If `align_to_render`: **resample** the native raster to the ortho render grid →
   `dem_<slug>/aligned/{product}_{datum}.tif`. The target extent/size comes from
   `dataset_manifest_render.json` when present (exact pixel alignment with the RGB
   render); otherwise it is computed from `DemConfig` target params.
4. Accumulate per-product asset paths, pixel dimensions, nodata, tile count,
   warnings/errors.

Merge and resample use `gdalbuildvrt` + `gdalwarp` (`-te`/`-ts`/`-t_srs`) when available,
with a clear actionable error if the gdal CLI is missing — the same posture `render.py`
already takes for cross-CRS reprojection. (pyvips/numpy remain available as a fallback
path if a pure-Python merge is later needed, but gdal is the primary path.)

Writes `dem_manifest.json` and returns `(exit_code, manifest_path)`. The CLI wrapper
prints the manifest path as the last stdout line (shell-composition contract). Exit code
`0` on success, `1` on data/policy failure (partial tile failure, nodata-only coverage,
AOI outside ISOK), `2` reserved for invalid CLI/config (raised at the CLI layer).

### 3. `src/satmap_dataset/config.py` — `DemConfig` (Pydantic v2)

Fields:
- AOI: `bbox` + `srs` (default `EPSG:2180`), **or** `center_lat`/`center_lon` +
  `square_km`/`area_km2` (mutually exclusive), resolved via the existing
  `_resolve_json_center_bbox` / `_lonlat_to_epsg2180` helpers in `cli.py`.
- `products: list[str]` ⊆ `{nmt, nmpt}`, default `["nmt", "nmpt"]`.
- `vertical_datum: str` ∈ `{evrf2007, kron86}`, default `evrf2007`.
- `dem_root: Path` (default derived from `location_name` slug → `dem_<slug>`).
- `align_to_render: bool` default `True`; `render_manifest: Path | None` (optional, for
  exact alignment); `target_width`/`target_height`/`px_per_meter` (fallback grid when no
  render manifest), reusing the paired-validation rules from `RenderConfig`.
- `max_request_px: int` (per-request tile cap, conservative default, e.g. 2048).
- `overwrite`, `timeout`, `retries`, `retry_delay`, `sleep_min`, `sleep_max`
  (same defaults as the download path).
- `location_name: str | None`, `provider_options: dict`.

Validators: bbox `xmin<xmax`/`ymin<ymax`; `products` non-empty subset; `vertical_datum`
enum; center/bbox mutual exclusivity; `target_width`/`target_height` paired.
`_apply_location_paths_policy` is extended to derive `dem_root` from the slug, consistent
with `downloads_`/`rendered_`/`artifacts_`.

### 4. `src/satmap_dataset/models.py` — `DemManifest` (Pydantic v2)

On-disk JSON contract recording: `stage="dem"`, `provider="geoportal"`, `products`,
`vertical_datum`, per-product `coverage_id` + `endpoint`, `bbox`, `srs`,
native asset path + dims + nodata, aligned asset path + dims (when produced),
request `tile_count`, `passed`, `warnings`, `errors`, `run_parameters`, and a `notes`
field documenting the current-best-composite limitation.

### 5. `GeoportalProvider.dem(config)` 

Thin delegator to `pipeline.dem.run`. Kept as a geoportal-specific method; **not** added
to the `Provider` ABC, so `lantmateriet` / `sentinel2` are unaffected.

### 6. CLI surface (`src/satmap_dataset/cli.py`)

Three flavors mirroring the ortho commands:
- `dem` — flag form (`--bbox`/`--center-lat`/`--center-lon`/`--square-km`,
  `--products`, `--vertical-datum`, `--dem-root`, `--align/--no-align`, …).
- `dem-json` — single JSON mapped 1:1 onto `DemConfig`.
- `dem-location-json` (single) and `dem-all-location-json` (batch) — merge
  `configs/run/base.json` with `configs/run/locations/<name>.json` via a new
  `_build_dem_config_from_base_and_location` helper.

Exit codes 0/1/2 honored; artifact path printed as last stdout line.
Optional matching `just` tasks (`dem-location-json`, `dem-all-json`).

### 7. `scripts/manage_location_roots.py`

Extend the root walk to include `dem_<slug>` so `just roots-list/move/delete` cover the
new output root alongside `downloads_`/`rendered_`/`artifacts_`.

## Output directory convention

```
dem_<slug>/
  native/   {nmt,nmpt}_{evrf2007|kron86}.tif            # authoritative 1 m (CRS embedded by GDAL)
  aligned/  {nmt,nmpt}_{evrf2007|kron86}.tif            # matches render grid (CRS embedded by GDAL)
  dem_manifest.json
```

`<slug>` is the existing NFKD→ASCII→lowercase slug. The root is gitignored like the
others.

## Data flow

```
bbox / center  →  DemConfig
                    │
        wcs_client.split_bbox → [sub_bbox...]
                    │  (per sub_bbox)
        wcs_client.get_coverage → GeoTIFF bytes → temp tile
                    │
        dem.run: gdal merge → native/{product}_{datum}.tif
                    │  (if align_to_render)
        dem.run: gdal resample to render grid → aligned/{product}_{datum}.tif
                    │
                 dem_manifest.json   (last stdout line = path)
```

## Testing

Network is mocked throughout.
- `split_bbox` tiling math (cap boundaries, exact-fit, single-tile, non-overlap/coverage).
- `coverage_id` / `endpoint_url` mapping for all product×datum combinations + overrides.
- `DemConfig` validation (bbox order, products subset, datum enum, center/bbox exclusivity,
  paired target dims).
- base+location merge → `DemConfig` (`_build_dem_config_from_base_and_location`).
- `DemManifest` JSON round-trip.

## Known limitations

- WCS GRID1 serves a **current-best composite** — no per-year DEM time series and no
  per-tile acquisition year via this transport (the skorowidz/WFS path, which would carry
  `akt_rok`, was intentionally not used). Documented in the manifest `notes`.
- The aligned copy **resamples/interpolates** the native 1 m grid; the native asset is
  authoritative.
- Merge/align require the **gdal CLI** (`gdalwarp`/`gdalbuildvrt`), already an optional
  dependency exercised by `render.py`.
- The development sandbox cannot reach the GUGiK endpoints; live validation is done
  outside the sandbox.
