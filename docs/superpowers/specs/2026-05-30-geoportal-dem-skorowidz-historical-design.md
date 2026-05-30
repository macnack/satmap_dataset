# Design: Historical NMT/NMPT download via Geoportal skorowidz (WFS)

**Date:** 2026-05-30
**Status:** Approved (pending spec review)
**Builds on:** `2026-05-30-geoportal-elevation-nmt-nmpt-design.md` (the WCS DEM stage)

## Goal

Add a **year-aware** elevation downloader that pulls historical NMT (DTM) and NMPT
(DSM) tiles from the GUGiK skorowidz (WFS) "Dane do pobrania" service, so the DEM can
be paired per-year with the orthophoto dataset. This complements the existing WCS DEM
stage (which only serves the current-best NMT/KRON86 composite, no history, no NMPT,
no EVRF2007).

Delivered as **Approach A**: a new `transport` mode on the existing `dem` stage —
`transport: "wcs"` (default, unchanged) or `transport: "skorowidz"` (this feature).

## Live-verified facts (research 2026-05-30, real service)

Public, **no authentication** (data is open since 2020). The earlier 401s were a wrong
service path (`/PZGIK/NMT/WFS/Skorowidze`); the correct services use the long Polish
names and return HTTP 200:

| Product | Datum | WFS skorowidz base URL |
|---|---|---|
| NMT | KRON86 | `https://mapy.geoportal.gov.pl/wss/service/PZGIK/NumerycznyModelTerenuKRON86/WFS/Skorowidze` |
| NMT | EVRF2007 | `.../PZGIK/NumerycznyModelTerenuEVRF2007/WFS/Skorowidze` |
| NMPT | KRON86 | `.../PZGIK/NumerycznyModelPokryciaTerenuKRON86/WFS/Skorowidze` |
| NMPT | EVRF2007 | `.../PZGIK/NumerycznyModelPokryciaTerenuEVRF2007/WFS/Skorowidze` |

- **Per-year typenames:** `gugik:SkorowidzNMT<YYYY>` (verified 2000, 2004–2019) and
  `gugik:SkorowidzNMPT<YYYY>`. Each year is a separate FeatureType.
- **Feature fields** (namespace `gugik`, GetFeature over an AOI bbox in EPSG:2180):
  `url_do_pobrania` (full public link, e.g.
  `https://opendata.geoportal.gov.pl/NumDaneWys/NMT/3718/3718_130926_N-33-141-C-a-3-4.asc`),
  `godlo`, `akt_rok`, `akt_data`, `dt_pzgik`, `format` (= "ARC/INFO ASCII GRID" → `.asc`),
  `char_przestrz` (e.g. "1.00 m"), `blad_sr_wys` (mean height error, m), `uklad_xy`
  (= "PL-1992" → EPSG:2180), `uklad_h` (= "PL-KRON86-NH"/"PL-EVRF2007-NH"),
  `czy_ark_wypelniony`, `zrodlo_danych` ("Skaning laserowy").
- These field names are the same ones the existing ORTO `wfs_client.get_year_tiles`
  already parses (`url_do_pobrania`, `godlo`, `akt_rok`, `akt_data`, `dt_pzgik`,
  `uklad_xy`), so that client is reusable.

## Decisions (from brainstorming)

| Decision | Choice |
|----------|--------|
| Integration | **Approach A** — `transport` field on the existing `dem` stage |
| Year selection | **All available DEM years in the requested range** (by `akt_rok` / typename) |
| Output | **Year-keyed**, aligned to the orthophoto render grid (one mosaic per year) |
| Products / datum | Reuse `DemConfig.products` (`nmt`/`nmpt`) and `vertical_datum` (`evrf2007` default; now actually reachable) |
| Default transport | `wcs` (preserves existing `dem` behavior) |
| `.asc` CRS | ASCII GRID carries no CRS → assign `EPSG:2180` on conversion (`uklad_xy=PL-1992`) |

## Components

### 1. `src/satmap_dataset/geoportal/wfs_client.py` (small generalization)

Add a `typename_pattern: re.Pattern[str] | None = None` parameter to `get_capabilities`
(and thread it into `_extract_year_typenames`). When `None`, it uses the current ORTO
regex `SkorowidzOrtof\w*?(\d{4})$` — **no behavior change for the ortho pipeline**.
`get_year_tiles` / `get_feature_count` already accept `base_url` and are reused as-is:
the per-feature bbox/axis-swap normalization, paging, `url_do_pobrania` extraction,
`_is_grid_compatible_with_srs` (PL-1992 ↔ EPSG:2180), and `_tile_id_from_url` all work
for NMT/NMPT. The absent `kolor` field yields `_color_priority` = 0 (harmless).

### 2. `src/satmap_dataset/geoportal/dem_skorowidz_client.py` (new, thin)

- `SKOROWIDZ_ENDPOINTS: dict[(product, datum), url]` for the four combinations above;
  overridable via `provider_options["skorowidz_endpoints"]`.
- `endpoint(product, datum, options) -> str`.
- `typename_pattern(product) -> re.Pattern` → `Skorowidz(NMT|NMPT)(\d{4})` for the
  product (`SkorowidzNMT(\d{4})` / `SkorowidzNMPT(\d{4})`).
- `async year_typenames(product, datum, *, options, timeout, retry_policy) -> dict[int,str]`
  — `get_capabilities(base_url=..., typename_pattern=...)`.
- `async tiles_for_year(product, datum, year, bbox, srs, *, year_to_typename, ...)`
  — delegates to `wfs_client.get_year_tiles(base_url=..., ...)`, returning the tile
  url/bbox/acquisition maps plus the per-tile `blad_sr_wys`/`godlo` it surfaces.

### 3. `src/satmap_dataset/pipeline/dem_skorowidz.py` (new stage body)

`run(config: DemConfig) -> tuple[int, Path]` (sync wrapper over async). For each
`product` in `config.products`:
1. `year_to_typename = year_typenames(product, datum)`; intersect with
   `range(year_start, year_end+1)`.
2. For each available year (sequential, geoportal is rate-sensitive):
   a. `tiles_for_year(...)` over the AOI; if zero tiles → record the year as
      `skipped` (not an error) and continue.
   b. Download each `.asc` to a temp dir (async httpx + `RetryPolicy` + jitter, the
      sentinel2/downloader pattern; reuse `_download_asset_with_retry`).
   c. **Mosaic** `.asc` tiles → `native/year_<YYYY>.tif`:
      `gdalbuildvrt -a_srs EPSG:2180 out.vrt *.asc` then
      `gdal_translate -a_srs EPSG:2180 -co COMPRESS=DEFLATE` clipped to the AOI bbox
      (`-projwin`), float32 preserved. Single-tile years skip the VRT.
   d. If `config.align_to_render`: resample to the ortho render grid via the existing
      `dem._align_to_grid` (one grid for all years, from the render manifest or config)
      → `aligned/year_<YYYY>.tif`.
   e. Record a `DemYearAsset`.
3. Reuse: if `native/year_<YYYY>.tif` exists and `not overwrite`, skip download+mosaic
   (still align if missing).

Output layout (separate from the WCS layout to avoid collisions):

```
dem_<slug>/skorowidz/<product>_<datum>/native/year_<YYYY>.tif
dem_<slug>/skorowidz/<product>_<datum>/aligned/year_<YYYY>.tif
dem_<slug>/dem_manifest.json
```

GDAL merge/align reuse `dem.py`'s `_merge_tiles`/`_align_to_grid` posture (capture
stderr, clear error if the GDAL CLI is missing). The `.asc`→GeoTIFF conversion adds the
`-a_srs EPSG:2180` flag because ASCII GRID has no embedded CRS.

### 4. `src/satmap_dataset/pipeline/dem.py` (dispatch only)

`run(config)`: if `config.transport == "skorowidz"` → `dem_skorowidz.run(config)`,
else the existing WCS `asyncio.run(_run_async(config))`. No other change to the WCS path.

### 5. `src/satmap_dataset/config.py` — `DemConfig` additions

- `transport: str` ∈ `{wcs, skorowidz}`, default `wcs`.
- `year_start: int | None`, `year_end: int | None` (≥ 1900). Validator: required and
  `year_end >= year_start` **when** `transport == "skorowidz"`; ignored for `wcs`.
- `requested_years` property (range) when both set.
- All other fields unchanged and shared (products, vertical_datum, align_to_render,
  render_manifest, dem_root, max_request_px [wcs-only], overwrite, timeout, retries,
  retry_delay, sleep_min/max, provider_options).

### 6. `src/satmap_dataset/models.py` — manifest additions

- New `DemYearAsset`: `year: int`, `native_path`/`native_width`/`native_height`,
  `aligned_path`/`aligned_width`/`aligned_height`, `tile_count: int`,
  `mean_height_error: float | None` (reserved; left `None` in this version — populating it
  would require extending the shared `wfs_client.get_year_tiles` return type), `godla: list[str]`,
  `passed: bool`, `errors: list[str]`.
- `DemProductAsset` gains `years: list[DemYearAsset] = []` (populated only for
  skorowidz; empty for wcs). WCS flat fields unchanged.
- `DemManifest` gains `transport: Literal["wcs","skorowidz"] = "wcs"` and
  `years_requested: list[int] = []`, `years_skipped: dict[int,str] = {}`.

### 7. CLI (`src/satmap_dataset/cli.py`)

- `dem` flag form + `dem-json` + `_build_dem_config_from_base_and_location`: add
  `--transport`, `--year-start`, `--year-end`. In `dem-location-json`, `year_start`/
  `year_end` are inherited from `base.json` (already present there for the ortho run),
  and `transport` may be set per-location or via a `--transport` option (default `wcs`).
- Exit codes 0/1/2 and last-stdout-line contract unchanged.

## Data flow (skorowidz)

```
DemConfig(transport=skorowidz, products, vertical_datum, year_start..year_end, AOI)
  → per product:
      year_typenames(product,datum)  ∩  [year_start..year_end]
        → per year: tiles_for_year (wfs_client.get_year_tiles, base_url+pattern)
            → download .asc tiles
            → gdal mosaic (-a_srs EPSG:2180, clip AOI) → native/year_YYYY.tif
            → [align] gdalwarp to render grid → aligned/year_YYYY.tif
            → DemYearAsset
  → DemManifest(transport=skorowidz, products[...].years[...])  (last stdout line = path)
```

## Error handling

- Per-(product, year) `try/except`: a failing year records its error and does not abort
  other years/products.
- A year with zero tiles over the AOI is **`years_skipped`**, not a failure.
- `passed` = at least one `DemYearAsset` produced successfully and no hard errors that
  prevented manifest write. Exit 0 if `passed` else 1.
- Missing GDAL CLI → clear `RuntimeError` (mirrors the WCS path).
- WFS/HTTP failures go through `RetryPolicy` + jitter.

## Testing (network + GDAL mocked)

- `wfs_client`: `get_capabilities` with a custom `typename_pattern` extracts NMT years;
  with `None` still extracts the ORTO years (regression guard).
- `dem_skorowidz_client`: endpoint + typename-pattern mapping for all 4 product×datum
  combinations; `provider_options` override.
- `dem_skorowidz.run`: year iteration with mocked `tiles_for_year`/download/`_merge_tiles`/
  `_align_to_grid` — asserts per-year native+aligned outputs, `years_skipped` for empty
  years, reuse-without-download path, partial-failure (one year fails → others pass,
  `passed` reflects success), `passed=False` when no year yields data.
- `.asc`→GeoTIFF command construction includes `-a_srs EPSG:2180`.
- `DemManifest`/`DemYearAsset` JSON round-trip with nested `years`.
- `dem.run` dispatch: `transport=skorowidz` calls `dem_skorowidz.run`; `transport=wcs`
  unchanged.
- CLI: `--transport`/`--year-start`/`--year-end` plumb into `DemConfig`; location builder
  inherits years from `base.json`; bad/missing years for skorowidz → exit 2.

## Known limitations / notes

- DEM acquisition years (sparse ALS campaigns) are independent of orthophoto years; this
  feature produces a mosaic per available DEM year, all on the shared ortho grid when
  `align_to_render` is set, so any DEM year overlays the ortho dataset.
- Vertical datum (KRON86 vs EVRF2007) selects the service endpoint and affects elevation
  **values**; the horizontal CRS of every output is EPSG:2180 regardless.
- WCS transport (`pipeline/dem.py`) remains the quick "current-best NMT/KRON86 composite"
  option and is unchanged.
- The dev sandbox blocks geoportal; live runs (and the skorowidz smoke) happen outside it.
