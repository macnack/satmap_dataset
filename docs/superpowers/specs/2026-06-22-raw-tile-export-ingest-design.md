# Spec: Raw-tile export + ingest — produce the `raw_tile_pipeline` dataset from downloads

**Date:** 2026-06-22
**Status:** Design (before implementation plan)
**Area:** `src/satmap_dataset/raw_tiles/` (new), `src/satmap_dataset/pipeline/raw_export.py` (new),
`config.py`, `models.py`, `cli.py`, `Justfile`, `tests/`

---

## 1. Goal

Let satmap_dataset **download maps in the format the sat_roma `raw_tile_pipeline` needs**, and
**handle the ingestion half of that pipeline as a new feature** — so a download run can produce
a ready-to-train raw-tile dataset without the manual "move `downloads_*` into
`sat_data_raw/<provider>/<area>/<year>/` and run the sat_roma scripts" step that exists today.

Concretely, add a new opt-in stage/command that turns the **native download tiles** into the
canonical raw-tile layout and ingests them into co-located season-cell stacks plus a split
manifest:

```
sat_data_raw/<provider>/<area>/<year>/*.tif          # raw export (native tiles)
sat_data_raw/<provider>/<area>/<cellkey>/year_YYYY.tif   # ingested cells (+ .tfw/.prj)
sat_data_raw/<provider>/<area>/manifest.yaml         # per-area ingest manifest
sat_data_raw/<split>_manifest.yaml                   # cross-location split manifest (handoff)
```

## 2. Repo boundary (what this feature does NOT do)

satmap_dataset owns **download → raw-export → ingest (cells + manifest)**. sat_roma's
`raw_tile_pipeline` owns **manifest → pairs/viz/training**. The handoff contract is the
**manifest** (per-area `manifest.yaml` + the split manifest). Pair generation
(`gen_pairs_from_manifest.py`) and visualisation (`viz_pairs.py`) stay in sat_roma and consume
the split manifest; they are out of scope here. (Decision: "export + ingest, pairs stay in
sat_roma".)

## 3. Current state and the gap

- The **download** stage already writes native per-year tiles to
  `download_root` = `downloads_<location-slug>/<year>/*.tif` at native GSD in the provider CRS
  (Geoportal `EPSG:2180`, Lantmäteriet `EPSG:3006`, NLS `EPSG:3067`). This is exactly the raw
  input the ingester wants — just under a `downloads_<slug>` root rather than
  `sat_data_raw/<provider>/<area>/<year>/`.
- The **render** stage homogenises/reprojects to `rendered_<slug>/year_YYYY.tif`. Raw-export
  deliberately consumes the **download** output, not render (no resampling). (Decision: "raw
  native tiles".)
- Today bridging to sat_roma is manual: reorganise the tiles and run the sat_roma scripts.

## 4. Architecture

### 4.1 `src/satmap_dataset/raw_tiles/` — ported ingestion core

Port the **portable** functions from sat_roma's `romatch/datasets/raw_tiles.py` (decision:
"port a copy", self-contained, no romatch dependency). Pure deps: the `gdalinfo` CLI + `pyvips`
+ `PyYAML` (all already used by satmap). The ported surface:

- `GeoTransform`, `gdalinfo_json`, `read_geotransform`, `read_crs_wkt`, `_epsg_from_wkt`
  (top-level/depth-1 CRS authority, tolerant of Geoportal WKTs that carry no authority).
- `geotransform_to_tfw_lines`, `write_tfw`, `write_prj_wkt` (corner→center `.tfw`; `.prj`
  verbatim from source WKT).
- `TileInfo`, `read_tile_info`, `detect_year` (parent-dir → `19xx/20xx` filename token →
  `TIFFTAG_DATETIME`).
- `Cell(ulx, uly, w_m, h_m)` (**rectangular** — tiles aren't always square), `cell_key`,
  `derive_cell_grid` (smallest-footprint), `tile_covers_cell`, `world_window_to_pixel`.
- `valid_pixel_fraction`, the EPSG→provider registry loader, `min_coverage_for_epsg`
  (per-provider; `geoportal: 0.5`), `resolve_season_tile`, `ingest_area`, and the
  `build_test_manifest` logic.

satmap already knows `provider` and `area` at download time, so it passes them directly; the
EPSG→provider registry is retained only as a **cross-check** (warn if the tile's detected EPSG
disagrees with the configured provider).

**Drift mitigation:** ship a small shared **fixture/test vector** (a synthetic geotransform →
expected `.tfw` lines, a non-square cell, a coverage value) identical to sat_roma's
`tests/test_raw_tiles.py`, so the two copies are checked against the same expected outputs. The
module header points at the sat_roma original as the source of truth.

### 4.2 `pipeline/raw_export.py` — new stage

`run(config: RawExportConfig) -> tuple[int, Path]`, following the stage contract (return
`(exit_code, artifact_path)`, write one JSON manifest, print the absolute artifact path as the
last stdout line). Steps:

1. Resolve `<provider>` from `config.provider` and `<area>` from the location slug.
2. **Export:** lay the download-stage native tiles into
   `<raw_root>/<provider>/<area>/<year>/*.tif` (symlink by default — `link_mode=copy` to
   materialise). Discover the per-year tiles from the download manifest (preferred) or by
   globbing `download_root/<year>/*.tif`.
3. **Ingest:** run `ingest_area` → `<raw_root>/<provider>/<area>/<cellkey>/year_YYYY.tif` +
   `.tfw`/`.prj`, gating coverage at the per-provider threshold (`--min-coverage` overrides),
   and write the per-area `manifest.yaml`.
4. Write `raw_export_manifest.json` (the stage artifact) into `artifacts_dir`: provider, area,
   raw_root, per-year exported tile counts, cells produced, seasons kept/dropped + coverage,
   and resolved output paths.

Idempotent reuse mirrors the other stages: skip when the per-area `manifest.yaml` exists and
matches (years on disk, `min_coverage`, `cell_size_m`, `link_mode`); reusing requires every
referenced path to still exist.

### 4.3 `RawExportConfig` (config.py) + manifest model (models.py)

- `RawExportConfig` (Pydantic v2, validated like the others): `provider`
  (geoportal|lantmateriet|nls; sentinel2 rejected — not raw orthophoto tiles), `location_name`,
  `download_root` / `download_manifest`, `raw_root` (default from env `SATMAP_RAW_ROOT`, else
  `~/Github/sat_data_raw`), `min_coverage` (optional; per-provider default when unset),
  `link_mode` (`symlink|copy`, default `symlink`), `cell_size_m` (optional), `artifacts_dir`.
- `RawExportManifest` (Pydantic v2) — the on-disk JSON contract for `raw_export_manifest.json`.

### 4.4 CLI surface (cli.py) — the 3-flavour pattern

- **Flag form:** `raw-export` (long args; `--provider`, `--download-root`, `--raw-root`,
  `--location-name`, `--min-coverage`, `--link-mode`, …).
- **JSON form:** `raw-export-json` (one JSON mapped onto `RawExportConfig`).
- **Base+location form:** `raw-export-location-json` (single) and
  `raw-export-all-location-json` (batch over a locations dir), reusing
  `_apply_location_paths_policy` to derive `download_root`/`artifacts_dir` from the slug and
  adding `raw_root` (a single shared root with `<provider>/<area>` subdirs — NOT a
  per-location root).
- **`raw-test-manifest`** — runs the ported `build_test_manifest` over `<raw_root>` and writes
  the cross-location split manifest consumed by sat_roma. (Decision: raw-export is a **separate
  opt-in command**, not wired into `run-all` yet.)

`Justfile`: add `raw-export-location-json`, `raw-export-all-location-json`, `raw-test-manifest`
recipes mirroring the existing `*-location-json` recipes.

## 5. Output layout & handoff contract (recap)

```
<raw_root>/<provider>/<area>/<year>/*.tif                 # exported native tiles
<raw_root>/<provider>/<area>/<cellkey>/year_YYYY.tif      # ingested cells (+ .tfw/.prj)
<raw_root>/<provider>/<area>/manifest.yaml                # per-area ingest manifest
<raw_root>/test_manifest.yaml                             # split manifest (handoff to sat_roma)
<artifacts_dir>/raw_export_manifest.json                 # stage artifact
```

`<provider>` ∈ {geoportal, lantmateriet, nls}; `<area>` is the location slug; `<cellkey>` is
`e<easting>_n<northing>` (SW corner, provider-CRS metres). sat_roma's
`raw_tile_pipeline/gen_pairs_from_manifest.py` consumes `test_manifest.yaml`.

## 6. Deliberate consequences / notes

- **No resampling.** Export symlinks (or copies) native tiles; ingest only symlinks 1:1 or
  losslessly pixel-window crops. Mixed-GSD years stay native; the split-manifest builder picks
  each cell's largest equal-dimension season group (the current sat_roma dataset needs
  co-registered stacks).
- **Provider-aware coverage** (geoportal `0.5`; godło sheets carry nodata borders) and
  shrink-on-load thumbnail coverage are ported, so full-area ingest stays fast.
- **EPSG cross-check:** satmap knows the provider, so a mismatch between the configured provider
  and the tile's detected EPSG is a warning (data sanity), not a routing decision.
- **Sentinel2** is not a raw-orthophoto-tile provider; `raw-export` rejects it.

## 7. Out of scope (YAGNI)

- Pair generation, viz, training (sat_roma `raw_tile_pipeline` + matcher).
- Wiring raw-export into `run-all` (separate command for now).
- A shared cross-repo package for the ingestion core (ported copy + shared test vector instead;
  revisit if drift becomes painful).
- The `SeasonSource` world-window path that would let training sample mixed-GSD stacks without
  the equal-dimension subset.

## 8. Open points for the implementation plan

1. Exact source of the per-year tile list for export: prefer the download manifest
   (`dataset_manifest_download.json`) asset paths over a raw glob — confirm the manifest field.
2. Whether `raw_root` is per-run (artifacts-relative) or a single shared root by default
   (proposed: single shared root via `SATMAP_RAW_ROOT`).
3. The reuse predicate `_can_reuse_raw_export` fields, to match the other stages' idempotency.
4. Tests: port sat_roma's `test_raw_tiles` unit vectors; add a stage test on a tiny 2-year /
   2-tile fixture (assert layout, sidecars, `manifest.yaml`, and the split manifest); add a CLI
   smoke test for `raw-export-location-json`.
