# GSD information in index manifests

**Date:** 2026-06-20
**Status:** Approved, pending implementation plan

## Problem

The pipeline records per-year availability and per-tile acquisition metadata, but
not the **ground sample distance (GSD)** — the native pixel size of the source
orthophoto. GSD varies by year and even by tile within a year (e.g. Poznań 2024 is
0.05 m while 2014/2017/2019 are 0.25 m). Without GSD in the manifest, consumers
cannot tell what resolution a given year actually offers, and cannot choose a
sensible `px_per_meter` (anything above the source GSD is empty upsampling).

## Data source (verified)

Each WFS skorowidz feature exposes `<gugik:piksel>` = GSD in meters, available at
**index time** (no download required). Confirmed live against
`SkorowidzOrtofomapy2024`: values like `0.05`, `0.25`. Sibling fields already parsed
nearby include `akt_rok`, `akt_data`, `kolor`, `url_do_pobrania`.

## Design

### 1. Model changes (`src/satmap_dataset/models.py`)

- `TileAcquisitionMetadata` → add `gsd: float | None = None` (per-tile, meters).
- New `YearGsdSummary(BaseModel)`:
  - `histogram: dict[str, int]` — GSD value as string key (e.g. `"0.05"`) → tile
    count, e.g. `{"0.05": 7, "0.25": 2}`. String keys because JSON object keys must
    be strings and to keep a canonical formatting of the float.
  - `finest: float | None` — min GSD across the year's tiles.
  - `coarsest: float | None` — max GSD across the year's tiles.
- `IndexManifest` → add `gsd_by_year: dict[int, YearGsdSummary] = Field(default_factory=dict)`.
  Per-tile GSD already flows through the existing `tile_acquisition_by_year`.
- `YearAvailabilityReport` → add `gsd_by_year: dict[int, YearGsdSummary] = Field(default_factory=dict)`
  (summary only — this is the user-facing "available years" artifact).

All new fields are **optional with defaults** so existing checked-in manifest
fixtures continue to validate (additive change, per the CLAUDE.md convention that
new fields must default-resolve cleanly).

### 2. Extraction (`src/satmap_dataset/geoportal/wfs_client.py`)

- In `_extract_tile_acquisition_metadata`, read `piksel` via the existing
  `_find_attr_value`, parse to float via a new `_parse_float_or_none` (mirroring the
  existing `_parse_int_or_none`), and include `gsd` in the returned dict. Missing or
  blank tag → `None`.

### 3. Aggregation (`src/satmap_dataset/pipeline/index_builder.py`)

- New helper `_summarize_gsd_by_year(tile_acquisition_by_year) -> dict[int, YearGsdSummary]`:
  for each year, build the histogram from per-tile `gsd` (skip `None`), set
  `finest = min(known)`, `coarsest = max(known)`. A year with no known GSD yields an
  empty `YearGsdSummary` (empty histogram, `finest`/`coarsest` = `None`).
- Canonical GSD string key via a small `_gsd_key(value: float) -> str` helper so
  `0.05` and `0.050` collapse to the same `"0.05"` bucket.
- Populate `gsd_by_year` on both the `IndexManifest` and the
  `YearAvailabilityReport` written by the index stage.

### Chosen approach

Aggregate in `index_builder` (orchestration layer), not in `wfs_client` (raw fetch)
and not as a lazy model property. This keeps the fetch layer returning raw per-tile
data, the model as a pure serializable container, and writes the summary to JSON.
It mirrors the existing `tile_acquisition_by_year` flow.

Alternatives considered and rejected:
- Aggregate inside `wfs_client.get_year_tiles`: summary should reflect the final
  filtered tile set the builder assembles, so the builder is the right place.
- Model `@property`: would not serialize to the on-disk JSON, which is the goal.

## Testing (TDD)

- `wfs_client`: feature XML containing `<gugik:piksel>0.05</gugik:piksel>` parses to
  `gsd=0.05`; a feature missing the tag parses to `gsd=None`.
- `_summarize_gsd_by_year`: a mixed-GSD year produces the correct histogram, finest,
  and coarsest; an all-`None` year produces an empty summary.
- Index run populates `gsd_by_year` in both `index_manifest.json` and
  `year_availability_report.json`.
- Backward compatibility: features without `piksel` do not break the index run.

## Out of scope (YAGNI)

- Download and render manifests are unchanged.
- No backfill of existing on-disk manifests.
- No CLI surface changes.
