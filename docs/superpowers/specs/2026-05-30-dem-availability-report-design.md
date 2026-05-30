# Design: DEM availability report (skorowidz, read-only)

**Date:** 2026-05-30
**Status:** Approved (pending spec review)
**Builds on:** `2026-05-30-geoportal-dem-skorowidz-historical-design.md`

## Goal

Given a location, query the GUGiK skorowidz (WFS) for **all** NMT/NMPT data and report
**what is available and what is not** — across every product, vertical datum, and
acquisition year — **without downloading any raster**. The output is a JSON
provenance/availability file the user inspects to decide which years to fetch (e.g. to
fill a coverage gap in one year from another year's acquisition) and to verify coverage
themselves.

Mirrors the existing location-based workflow: the user has
`configs/run/locations/<name>.json`, runs one `just` task, and gets a report written to
the location's `artifacts_<slug>/` dir plus a short console table.

## Motivating finding (live, 2026-05-30, Przeźmierowo 1 km²)

NMPT availability is split by datum: **EVRF2007** = 2019–2025, **KRON86** = 2010–2019.
For this AOI, NMPT 2019 covers only the eastern sheet (`…-3-4`) → 78% of the AOI, while
NMPT **2024** covers **both** sheets (`…-3-3` west + `…-3-4` east) → full AOI. Without an
availability report there is no way to discover that 2024 fills the 2019 gap.

## Decisions (from discussion)

| Decision | Choice |
|----------|--------|
| Transport | skorowidz WFS only (no downloads; reuses the axis-swap + endpoints already built) |
| Scope | All four `(product, datum)` combinations: NMT/NMPT × KRON86/EVRF2007 |
| Years | All years advertised by each endpoint's GetCapabilities ("all available") |
| Coverage | Per `(product, datum, year)`: `full` / `partial` / `none` + `coverage_pct`, from the union of tile bboxes intersected with the AOI |
| Interface | Location-based, like the other `*-location-json` commands + a `just` task |
| Output | `artifacts_<slug>/dem_availability.json` + a console "✓ / partial / ✗" table |
| Side effects | None — read-only; exit 0 always (a transport error for one combo is recorded, not fatal) |

## External services

The four skorowidz WFS endpoints and the per-year typename patterns
(`Skorowidz(NMT|NMPT)<YYYY>`) from `geoportal/dem_skorowidz_client.py` are reused as-is.
The WFS BBOX is sent in EPSG:2180 `(ymin,xmin,ymax,xmax)` order (the verified axis-swap),
overridable via `provider_options["wfs_swap_bbox_axes"]`.

## Components

### 1. `src/satmap_dataset/pipeline/dem_availability.py` (new)

`run(config: DemAvailabilityConfig) -> tuple[int, Path]` (sync wrapper over async; matches
the stage contract; writes one JSON; returns `(0, output_json)`).

For each `(product, datum)` in the configured set:
1. `year_typenames(product, datum)` (GetCapabilities). A failure is recorded as an error
   for that combo and the loop continues (read-only, never fatal).
2. For each year (all advertised, optionally intersected with a year range if provided):
   `tiles_for_year(product, datum, year, query_bbox, srs, ...)` over the AOI.
   - Build a `DemAvailabilityEntry`: `product`, `datum`, `year`, `godla` (sorted distinct
     sheet names), `tile_count`, `formats` (distinct of `asc`/`xyz`/`zip` from the tile
     URLs), `coverage` (`full`/`partial`/`none`), `coverage_pct`, `acquisition_dates`
     (distinct, from tile acquisition metadata).
   - Years with zero intersecting tiles are recorded with `coverage="none"`,
     `tile_count=0` (this is the explicit "what is NOT there").
3. Concurrency: years within a product are queried sequentially (geoportal is
   rate-sensitive); a small fixed inter-request jitter via `RetryPolicy` is reused.

Coverage computation (`_coverage_for_tiles`): given the AOI bbox and the per-tile bboxes
returned by `get_year_tiles` (already orientation-normalised against the query bbox),
compute the fraction of the AOI area covered by the union of (tile ∩ AOI) rectangles.
A light grid-rasterisation of the rectangles (e.g. 200×200 cells over the AOI) gives a
robust union area without a geometry dependency. `>= 0.999` → `full`; `> 0` → `partial`;
`== 0` → `none`.

### 2. `src/satmap_dataset/config.py` — `DemAvailabilityConfig` (Pydantic v2)

- AOI: `bbox` + `srs` (EPSG:2180), or `center_lat`/`center_lon` + `square_km`/`area_km2`
  resolved by the existing `_resolve_json_center_bbox` (same as DemConfig).
- `products: list[str]` ⊆ `{nmt, nmpt}`, default both.
- `datums: list[str]` ⊆ `{evrf2007, kron86}`, default both.
- `year_start`/`year_end: int | None` — optional filter; default `None` = all advertised
  years.
- `output_json: Path` (default `artifacts/dem_availability.json`; under
  `artifacts_<slug>` in location mode).
- `location_name`, `timeout`, `retries`, `retry_delay`, `sleep_min`, `sleep_max`,
  `provider_options`. Reuses the bbox/products/datum validators (factored to share with
  `DemConfig`).

### 3. `src/satmap_dataset/models.py` — availability manifest

- `DemAvailabilityEntry`: `product`, `datum`, `year`, `godla: list[str]`,
  `tile_count: int`, `formats: list[str]`, `coverage: Literal["full","partial","none"]`,
  `coverage_pct: float`, `acquisition_dates: list[str]`.
- `DemAvailabilityReport`: `kind="dem_availability"`, `generated_at`, `provider`,
  `aoi_bbox`, `srs`, `entries: list[DemAvailabilityEntry]`,
  `errors: dict[str,str]` (keyed `"<product>|<datum>"` for capability failures),
  `full_coverage_options: list[dict]` (the `(product,datum,year)` triples whose
  `coverage == "full"`), `run_parameters`.

### 4. CLI (`src/satmap_dataset/cli.py`)

Three flavors mirroring the existing ortho/DEM commands:
- `dem-availability-json <params.json>` — maps 1:1 onto `DemAvailabilityConfig`.
- `dem-availability-location-json <location.json> [--base-json …]` — merges base+location
  via a new `_build_dem_availability_config_from_base_and_location`, writes to
  `artifacts_<slug>/dem_availability.json`.
- `dem-availability-all-location-json [--locations-dir …] [--continue-on-error]` — batch.

After writing the JSON, each command prints a compact table to the console
(via the existing `console`), one row per `(product, datum, year)` that has data, plus a
trailing list of `(product,datum)` combos with NO data in range, then the artifact path
as the last stdout line (shell-composition contract). Exit code `0`.

### 5. `justfile`

```just
# Report available NMT/NMPT skorowidz data for a single location (no download)
dem-availability location_json:
    python -m satmap_dataset.cli dem-availability-location-json {{location_json}}

# Report availability for all locations in the default dir
dem-availability-all:
    python -m satmap_dataset.cli dem-availability-all-location-json
```

## Console output (illustrative)

```
Przeźmierowo — DEM availability (AOI 1.0 km²)
 product datum     year  tiles  cover   formats
 nmt     kron86    2011    2     full    asc
 nmt     evrf2007  2019    2     full    asc,xyz.zip
 nmpt    evrf2007  2019    1     partial(78%) asc
 nmpt    evrf2007  2024    2     full    asc,xyz.zip
 ...
 No data in range: nmpt/evrf2007 {2020,2021,2022,2023,2025}
artifacts_przezmierowo/dem_availability.json
```

## Data flow

```
location.json + base.json → DemAvailabilityConfig (center→bbox, swap axes)
  → per (product,datum): year_typenames (capabilities)
      → per year: tiles_for_year over AOI → DemAvailabilityEntry (godla, formats, coverage%)
  → DemAvailabilityReport → artifacts_<slug>/dem_availability.json  (+ console table)
```

## Error handling

- A GetCapabilities/GetFeature failure for one `(product,datum)` (or one year) is recorded
  in `errors` / as a `coverage="none"` entry; it never aborts the report. Exit 0.
- No GDAL, no filesystem writes beyond the JSON; no rasters downloaded.

## Testing (network mocked)

- `_coverage_for_tiles`: full AOI single tile → `full`/100%; half-covering tile →
  `partial`/~50%; no tiles → `none`/0%; two tiles tiling the AOI → `full`.
- `DemAvailabilityConfig` validation (products/datums subsets, bbox order, center→bbox).
- `dem_availability.run` with mocked `year_typenames`/`tiles_for_year`: builds entries for
  all combos/years; records `coverage="none"` for empty years; records `errors` when
  capabilities raise; `full_coverage_options` populated; writes JSON; exit 0.
- `DemAvailabilityReport`/`DemAvailabilityEntry` JSON round-trip.
- CLI: `dem-availability-json` writes the report and prints the artifact path last;
  `_build_dem_availability_config_from_base_and_location` resolves AOI + output path under
  `artifacts_<slug>`; bad config → exit 2.

## Known limitations / notes

- Coverage % is computed from tile bbox extents (rectangles), not from actual raster
  validity, so a tile flagged `czy_ark_wypelniony=NIE` (sheet not fully filled) could
  overstate coverage. The report notes this; the user verifies against the JSON.
- `mean_height_error` is not included (the reused `get_year_tiles` does not surface
  `blad_sr_wys`); `acquisition_dates` and `godla` are surfaced. Consistent with the
  skorowidz download stage's documented limitation.
- Read-only discovery only; fetching/compositing chosen years (and writing a per-tile
  provenance sidecar for a gap-filled mosaic) is a separate follow-up feature.
- Dev sandbox blocks geoportal; live runs happen outside it.
