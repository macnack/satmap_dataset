# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`satmap_dataset` builds a year-aware orthophoto dataset from the Polish Geoportal (`mapy.geoportal.gov.pl`). It probes WFS for per-year availability, downloads TIFFs (WFS direct + WMS fallback), then renders all years to a shared NN-ready grid via `pyvips`. Outputs are `RGB_U8` GeoTIFFs in `EPSG:2180` plus JSON manifests at every stage.

Python ≥ 3.10. Package layout is `src/satmap_dataset/` with the project installed editable via `pip install -e ".[dev]"`. `direnv` (`.envrc`) auto-activates `.venv` and exports `PYTHONPATH=$PWD/src` along with `SATMAP_LOCATIONS_ROOT`, `SATMAP_LOCATIONS_DIR`, `SATMAP_BASE_JSON`.

System dependency on Linux: `libvips42` and `libvips-tools` (`pyvips` will not import without them).

## Common Commands

```bash
# Install (editable + dev extras)
just install                      # or: python -m pip install -e ".[dev]"

# Tests (pytest is configured with -q and testpaths=tests in pyproject.toml)
pytest                            # all tests
pytest tests/test_models_schema.py            # single file
pytest -k "year_filter"                       # by name
pytest tests/test_render_helpers.py::test_x   # single test

# Single location (merge base.json + location.json, then run end-to-end)
just run-location-json location_json=configs/run/locations/poznan.json
just index-location-json location_json=configs/run/locations/poznan.json

# All locations in a directory (no merge step needed)
just run-all locations_4          # alias for $SATMAP_LOCATIONS_ROOT/locations_4
just index-all-json               # uses default configs/run/locations
just summary-locations            # auto-picks locations dir; prints status table

# Manage on-disk roots (downloads_*, rendered_*, artifacts_*) for a locations dir
just roots-list locations_4
just roots-move locations_4 target_dir=./archive execute=1
just roots-delete locations_4 execute=1

# Direct CLI (skip just; same module either way)
python -m satmap_dataset.cli --help
python -m satmap_dataset.cli run --year-start 2015 --year-end 2025 --bbox "210300,521900,210500,522100" --profile reference
```

CLI exit codes are load-bearing — callers parse them: `0` success, `1` policy/data failure (e.g. not enough years), `2` invalid CLI/config arguments. Every command also prints the absolute path of the artifact it wrote as its last stdout line; orchestrators rely on this.

## Architecture

### Pipeline stages and reuse

Four stages, each implemented as `src/satmap_dataset/pipeline/<stage>.py` exposing a single `run(config) -> tuple[int, Path]`:

1. `index_builder.run` — WFS `GetCapabilities` then per-year `GetFeature` → `index_manifest.json` + `year_availability_report.json`.
2. `downloader.run` — async `httpx`+`aiofiles` of `url_do_pobrania` URLs from the index, plus WMS-tiled fallback for years missing in WFS → `dataset_manifest_download.json`.
3. `render.run` — pyvips composes per-year mosaics on a shared grid → `dataset_manifest_render.json` + `<render_root>/year_YYYY.tif` files.
4. `validator.run` — checks asset existence, pixel profile, sizes, EPSG, georef, sidecars → `validation_report.json`.

`pipeline/run_all.py` orchestrates all four and is the entry point for the `run`, `run-json`, and `run-location-json` CLI commands. It implements **idempotent reuse**:

- Index is reused if `_can_reuse_index` matches (year range, bbox, srs, strict/min flags, provider) and tile bboxes don't appear axis-swapped.
- Download is reused if `_can_reuse_download` matches mode/profile/`force_wms_years` and every asset path on disk still exists.
- `run-all-location-json` skips a whole location when `<artifacts_dir>/validation_report.json` already shows `passed=true`.

When changing pipeline behavior, also update these reuse predicates — otherwise `run-all` will silently keep stale outputs.

### Models and configs

- `src/satmap_dataset/models.py` — Pydantic v2 manifest schemas (`IndexManifest`, `DatasetManifest`, `ValidationReport`, `YearAvailabilityReport`). These are the on-disk JSON contract; bumping fields means regenerating fixtures.
- `src/satmap_dataset/config.py` — Pydantic v2 input configs (`IndexConfig`, `DownloadConfig`, `RenderConfig`, `ValidateConfig`, `RunConfig`, plus a legacy `MosaicConfig`). Each enforces invariants (bbox xmin<xmax, mode in `{wms_tiled, wfs_render, hybrid}`, profile in `{train, reference}`, compression matches `deflate|jpeg|jpegNN`, `target_width`/`target_height` paired, etc.). Always construct configs via the model — direct dict access bypasses validation.

### CLI surface (single file: `cli.py`)

Every command has three flavors that share the same underlying `run()` functions:

1. **Flag form** — `index`, `download`, `render`, `mosaic` (alias for render), `validate`, `run`. Long argument lists; useful for ad-hoc invocations.
2. **JSON form** — `index-json`, `download-json`, `render-json`, `validate-json`, `run-json`. Take a single JSON file mapped 1:1 onto the corresponding Pydantic config.
3. **Base + location form** — `*-location-json` (single) and `*-all-location-json` (batch over a directory). Merges `configs/run/base.json` (defaults) with `configs/run/locations/<name>.json` (just `location_name` + `center_lat` + `center_lon`).

The base+location merge logic lives in `_build_*_config_from_base_and_location` helpers and is the part most tests exercise.

### bbox resolution

Two mutually exclusive ways to specify the AOI:

- `--bbox xmin,ymin,xmax,ymax` in the chosen `--srs` (default `EPSG:2180`, project axis order x,y = easting, northing).
- Center mode: `--center-lat`/`--center-lon` (WGS84) plus `--square-km` (default `4.0` → 2 km × 2 km square). EPSG:2180 only.

JSON inputs accept `center_lat`/`center_lon` plus either `square_km` or `area_km2` (mutually exclusive). Resolution goes through `_resolve_json_center_bbox` and uses `pyproj` if available, else shells out to the `proj` CLI — both are acceptable, but errors mention both. See `_lonlat_to_epsg2180` in `cli.py`.

### Location → output directory convention

When a config has `location_name` set, `_apply_location_paths_policy` derives `download_root`, `render_root`, `artifacts_dir` from the slug (NFKD → ASCII → lowercase → non-alnum to `_` → collapse repeats). E.g. `"Poznań"` → `downloads_poznan`, `rendered_poznan`, `artifacts_poznan` under the repo root. These dirs are gitignored. `scripts/manage_location_roots.py` (exposed as `just roots-*`) walks them.

### Raw-tile export (opt-in, not in `run-all`)

The `raw-export` stage (`pipeline/raw_export.py`) turns native download tiles into the layout sat_roma's `raw_tile_pipeline` consumes: it lays `download_root/<year>/*.tif` into `<raw_root>/<provider>/<area>/<year>/` (symlink by default; `link_mode=copy` to materialise), then ingests co-located season-cell stacks `<raw_root>/<provider>/<area>/<cellkey>/year_YYYY.tif` (+ `.tfw`/`.prj`) with provider-aware coverage gating (geoportal `0.5`), and writes the per-area `manifest.yaml`. `raw-test-manifest` builds the cross-location split `test_manifest.yaml`. The ingestion core under `src/satmap_dataset/raw_tiles/` is a **ported copy** of sat_roma `romatch/datasets/raw_tiles.py` (drift-guarded by `tests/test_raw_tiles_core.py`); keep them in sync. `raw_root` defaults to `$SATMAP_RAW_ROOT` or `~/Github/sat_data_raw` — a single shared root, not per-location. CLI: `raw-export`, `raw-export-json`, `raw-export-location-json`, `raw-export-all-location-json`, `raw-test-manifest`; Justfile: `just raw-export-location-json`, `just raw-export-all-json`, `just raw-test-manifest`.

**`cell_mode`** selects the ingest strategy. `footprint` (default) is the verbatim ported `core.ingest_area`: cells keyed on a single tile origin, one covering tile per cell. `world_window` (`raw_tiles/world_window.py`, satmap-only, **not** in sat_roma) handles **mixed-GSD areas**: per spot it snaps the intersection of every year's footprint to the **coarsest** grid present, lossless-crops each year to that identical window, and resamples every year to the coarse GSD so all years are co-registered **and** equal-dimension (one usable stack). The resample (`_equalize_to_grid`) picks the method by GSD ratio: **integer** ratio → exact box-average decimation, no interpolation (Geoportal 0.05→0.25 = 5×); **non-integer** ratio → anti-aliased Lanczos downscale to the coarsest GSD (Lantmäteriet 0.16→0.25 = 1.5625×); equal GSD → identity. `equalize_gsd` (CLI `--equalize-gsd/--raw-gsd`, default on) controls whether this resampling happens: with `--raw-gsd` each year keeps its **native GSD** — only the integer-pixel window crop, fully lossless, no resampling — so years stay co-registered geographically but differ in pixel dimensions (sat_roma's equal-dim split then picks the largest GSD group). Native hi-res also stays in the export `<year>/` folders; per-season `native_gsd`/`window_gsd`/`downsampled`/`resample`/`dims` record provenance. Window geometry uses the real geotransforms (no guessing). Verified: `poznan_15km2` (0.25+0.05 m, offset grids, 5× decimate) and `lulea` (0.25+0.16 m, aligned grids, Lanczos) each collapse to one equal-dim co-registered stack per cell.

**`gmix` workflow** (producing the flat `~/Github/sat_data/<provider>_<area>_<cellkey>_gmix/` cells sat_roma trains on): the nested `world_window` cells must be flattened into that layout — there is no satmap stage for it (`gmix` is a sat_roma naming convention), so `scripts/flatten_gmix.py` (`--location-json` derives provider+area) copies `<raw_root>/<provider>/<area>/<cellkey>/` → `<dest>/<provider>_<area>_<cellkey>_gmix/` (default dest `~/Github/sat_data`). The full chain is **index + download (no render) → world_window raw-export → flatten**; render is skipped because these cells use native download tiles, and 15 km² hi-res downloads are ~20–30 GB. Justfile: `just gmix` (full chain), `just gmix-download` (index+download only), `just gmix-flatten`. The location JSON opts in with `provider` + `cell_mode: "world_window"` + `equalize_gsd: true` (see `configs/run/locations/wroclaw_15km2.json`).

### External services

- WFS catalog: `https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WFS/Skorowidze` (year typenames matched by regex `SkorowidzOrtof\w*?(\d{4})$`).
- WMS fallback: `https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMS/StandardResolutionTime`.

Geoportal is rate-sensitive even for sequential calls. Every request goes through `geoportal/http.py` with a `RetryPolicy` and a randomized pre-request sleep (`sleep_min`/`sleep_max`, defaults `0.6`–`2.2 s`). Don't remove the jitter or batch requests aggressively without testing against a real run.

### LROC NAC provider (Moon, multi-temporal)

`provider="lroc_nac"` sources multi-temporal lunar LROC NAC observations from
the PDS Orbital Data Explorer (ODE) REST API
(`https://oderest.rsl.wustl.edu/live2`). Requires a lunar CRS
(`srs="IAU_2015:30100"`, ocentric lon/lat degrees). `index` enumerates every
overlapping NAC observation across the bbox + year range (each `pdsid` a
distinct tile under its acquisition year — the multi-temporal axis);
`download` pulls the PDS frames. `provider_options`: `product_type`
(default `CDRNAC4`), `page_limit`, `max_pages`, `max_incidence_angle`,
`min_obtime`/`max_obtime`. Downloaded frames are unprojected camera-geometry
rasters — ISIS `cam2map` projection and render are a separate, deferred stage.
Sample configs: `configs/run/lroc_nac_apollo17.{index,download}.json`.

### Profiles and modes

- `mode`: `wfs_render` (WFS only), `wms_tiled` (WMS only — index step is stubbed via `_write_wms_only_index`), `hybrid` (default; WFS-first, WMS for missing years).
- `profile`: `train` (default) or `reference`. `reference` mirrors a legacy `download_map.py` with geometry-driven output sizing (`px_per_meter`) and additional QC fields in the render manifest (`years_source_map`, `coverage_ratio_by_year`, `color_qc_by_year`).

## Conventions worth preserving

- Stage `run()` functions return `(exit_code, artifact_path)` and write a single JSON manifest. Don't return None or print the path elsewhere — the CLI wrapper relies on the tuple and the artifact path is the contract for shell composition.
- New config fields must default-resolve cleanly when missing from `base.json` or a location JSON; existing generated configs in `configs/run/generated/` are checked in and act as fixtures.
- The `mosaic` CLI command is a backwards-compatible alias for `render`. Don't reintroduce a separate mosaic stage.
### bbox axis order

**Authority:** `src/satmap_dataset/geo/bbox.py`. Project bboxes are always `(easting, northing)` for EPSG:2180. WFS/GUGiK skorowidz queries use `wfs_query_bbox_str()` (authority order). Render `source_axis_mode` handles swapped TIFF georef separately.

- `experimental_wfs_swap_bbox_axes` is deprecated and ignored for EPSG:2180.
