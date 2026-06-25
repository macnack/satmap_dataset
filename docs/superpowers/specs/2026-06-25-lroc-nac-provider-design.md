# LROC NAC provider (multi-temporal lunar imagery) — design

**Date:** 2026-06-25
**Status:** Approved, pending implementation plan
**Branch:** `worktree-planetary-imagery-providers`

## Summary

Add a new orthophoto provider, `lroc_nac`, that sources **multi-temporal lunar
imagery** from the Lunar Reconnaissance Orbiter Narrow Angle Camera (LROC NAC)
via the PDS **Orbital Data Explorer (ODE) REST API**. Given a lunar lat/lon
bounding box and a date range, the provider enumerates every overlapping NAC
observation across the mission (imaging began 2009-09-15, ongoing) and downloads
the raw/calibrated frames. Each acquisition epoch is preserved as a distinct
tile, giving the time axis the existing pipeline's year-keying needs.

This is the **first vertical slice** of a larger planetary-imagery effort
(eventual targets: CTX, HRSC, Moon Trek). It deliberately stops at
**index + download**. Map projection and render are explicitly deferred.

## Goal and motivation

The user wants genuine **multi-temporal** planetary coverage — change detection
over time on the Moon — and, as an immediate deliverable, the ability to
**download real sample tiles** that prove the multi-temporal enumeration works.

LROC NAC is the canonical lunar change-detection source: ~2.86M NAC frames with
dense repeat coverage, the basis of published new-impact-detection work
(e.g. PyNAPLE). ODE exposes this catalog as a scriptable REST API keyed by
footprint and observation time — a clean fit for the provider abstraction.

## Key external facts (verified June 2026)

- **ODE REST base:** `https://oderest.rsl.wustl.edu/live2`
- **Catalog query:** `query=product&target=moon&ihid=LRO&iid=LROC&pt=<code>&output=JSON`
- **Bounding box (all four required):** `westernlon=`, `easternlon=`, `minlat=`,
  `maxlat=`, plus `loc=f` (input bbox intersects product footprint).
- **Date range:** `minobtime=` / `maxobtime=` (UTC, partial dates allowed, e.g.
  `2012-04-03`). Sort `oba`/`obd`.
- **Result flags:** `results=opmf` → ODE id + PDS ids + full metadata + file URLs.
- **NAC product types:** `CDRNAC4` (calibrated, ~2.86M) and `EDRNAC4` (raw,
  ~2.87M) are the dense multi-temporal sources. **Both are in camera geometry —
  NOT map-projected.** The already-projected products (`SDNDTM` orthophotos,
  `SDPPHO`) are GDAL-ready but sparse and single-epoch.
- **Per-product fields:** `pdsid`, `Observation_time`, `UTC_start_time`,
  `Incidence_angle`, `Emission_angle`, `Phase_angle`, `Map_resolution`, footprint
  WKT (`C0`/`GL`/`NP`/`SP`, `Footprints_cross_meridian`, `Pole_state`),
  `Start_orbit_number`, and `Product_files[]` (each `FileName`, `URL`, `Type`,
  `KBytes`).
- **CRS (IAU_2015, GDAL/PROJ-supported, Moon R=1,737,400 m):**
  `IAU_2015:30100` Moon geographic (ocentric lon/lat), `:30110/:30115`
  equirectangular, `:30120/:30125` sinusoidal; polar stereographic variants for
  the caps (exact numeric code resolved at build time via `projinfo`).

**Consequence for this slice:** there is no endpoint returning map-projected,
multi-epoch NAC tiles by bbox+date. ODE solves *search + download*; projection
(ISIS `cam2map`) is a separate, heavier concern deferred to a follow-up spec.
The downloaded frames in this slice are unprojected camera-geometry rasters with
rich footprint/acquisition metadata — sufficient to prove multi-temporal
enumeration and to feed a future projection stage.

## Architecture

New package `src/satmap_dataset/providers/lroc_nac/`, mirroring the
`lantmateriet/` provider layout:

| File | Responsibility |
|------|----------------|
| `ode.py` | ODE REST client: build query URL, page results, parse JSON product records into typed `OdeProduct` objects, group by acquisition year. Analog of `lantmateriet/stac.py`. |
| `crs.py` | Lunar IAU CRS helpers; normalize the request bbox to ODE planetocentric lon/lat (handles 0–360 vs −180–180 longitude). Analog of `lantmateriet/crs.py`. |
| `provider.py` | `LrocNacProvider` implementing `Provider.index()` and `.download()`. |
| `__init__.py` | Exports `LrocNacProvider`. |

Edits to existing files:

- `providers/__init__.py` — add `lroc_nac` to `get_provider()`.
- `config.py` — extend `_validate_provider_srs` so `provider='lroc_nac'`
  accepts `IAU_2015:301xx` lunar codes (default `IAU_2015:30100`); ensure the
  provider arg is recognized wherever providers are validated.
- `cli.py` — recognize `lroc_nac` as a provider value (no new command; the
  existing `index-json` / `download-json` / `run-json` flow drives it).

### Component contracts

- **`ode.OdeProduct`** — typed record: `pdsid`, `observation_time: datetime`,
  `acquisition_year: int`, `incidence_angle`, `emission_angle`,
  `map_resolution`, `footprint_wkt`, `file_url`, `file_bytes`. Pure data; no IO.
- **`ode.search_products(options, bbox, date_range, retry_policy) -> list[OdeProduct]`**
  — async, paginated, polite (RetryPolicy + jitter). Raises on transport error;
  caller records it in the manifest.
- **`crs.normalize_bbox_to_ode(bbox, srs) -> (westlon, eastlon, minlat, maxlat)`**
  — degrees in ODE convention.
- **`LrocNacProvider.index(IndexConfig) -> (exit_code, Path)`** and
  **`.download(DownloadConfig) -> (exit_code, Path)`** — the load-bearing
  `(exit_code, artifact_path)` contract, single JSON manifest each.

## Data flow

1. **index()**
   - Resolve `provider_options` (ODE url, `pt` default `CDRNAC4`, `loc=f`,
     `page_limit`, `max_pages`, optional `max_incidence_angle` filter).
   - Convert bbox via `crs.normalize_bbox_to_ode`.
   - Date range from `year_start`/`year_end` → `minobtime`/`maxobtime`.
   - `ode.search_products(...)` → group `OdeProduct`s by `acquisition_year`.
   - Build `IndexManifest`: `tile_sources_by_year[year] = {pdsid: file_url}`,
     `tile_bboxes_by_year` (footprint bbox), `tile_acquisition_by_year`
     (obs date), incidence-angle/pixel-scale and ODE query echo in
     `provider_metadata`.
   - `evaluate_year_policy(...)` against the distinct years found
     (`min_years`/`strict_years`). Write `index_manifest.json` +
     `year_availability_report.json`. Exit 0 if policy passes, else 1.

2. **download()**
   - Read the index manifest. For each `(year, pdsid, url)` build
     `download_root/<year>/<pdsid>.<ext>`.
   - Download via the existing async + RetryPolicy + jitter machinery (reuse the
     lantmateriet downloader pattern: non-retryable 4xx set, exponential backoff,
     pre-request sleep). PDS/ODE are public services — keep concurrency modest.
   - Write the download-stage `LayerManifest` (`provider='lroc_nac'`,
     `layer='lroc_nac_mono'`, `assets`, `years_source_map[year]='ode'`).
     Exit 0 if assets present and none failed, else 1.

**Why the multi-temporal axis works without schema changes:** NAC observation
years are real Earth calendar years ≥ 1900 (2009→now), satisfying the existing
`year: int = Field(..., ge=1900)` constraint. Multiple observations within one
year are distinct `pdsid` tile_ids under that year in
`tile_sources_by_year[year]`, so a request spanning 2009–2026 yields a per-year
stack of NAC frames — exactly the time series required.

## Sample-download deliverable

A checked-in sample config targets **Apollo 17 / Taurus-Littrow** (≈20.19°N,
30.77°E), a densely re-imaged ROI. Small bbox, e.g.
`westlon=30.6, eastlon=30.9, minlat=20.0, maxlat=20.35`, years 2009–2026,
`pt=CDRNAC4`. Config form to be finalized in the plan (a JSON input under
`configs/run/` driving `index-json` then `download-json`).

**Acceptance:** a real run returns NAC frames spanning **≥2 distinct years** and
writes the files to disk; the index manifest lists those years in
`years_included`.

## Error handling and edge cases

- Empty ODE result → manifest `passed=false`, exit 1, clear message.
- ODE transport error → captured in manifest `errors[]`, exit 1.
- Non-retryable HTTP 4xx on download → fail that asset, do not retry.
- Footprints crossing the meridian / polar `Pole_state` → recorded in metadata,
  not specially handled (no projection in this slice).
- Longitude convention (0–360 vs −180–180) → confirmed against a live ODE probe
  during implementation; `crs.normalize_bbox_to_ode` owns the rule.

## Testing strategy

- **`ode.py` URL builder** — asserts exact ODE params for a given bbox/date/pt.
- **`ode.py` parser** — a recorded real ODE JSON response fixture →
  `OdeProduct` list → correct fields and year grouping (mirrors the lantmateriet
  STAC parser tests).
- **`crs.normalize_bbox_to_ode`** — degree/convention cases.
- **`provider.index()` / `.download()`** — mocked HTTP; assert manifest shape,
  exit codes, year-policy behavior, file layout.
- No drift guard needed (this is not a ported sat_roma core).

## Out of scope (explicit)

- ISIS / `cam2map` projection and any map-projected output → follow-up spec.
- render / validator / raw-export integration → follow-up spec.
- The other planetary providers (CTX global mosaic, CTX individual RDRs, HRSC,
  Moon Trek / QuickMap) → separate specs.
- Already-projected single-epoch products (`SDNDTM`, `SDPPHO`).

## Follow-up specs (decomposition roadmap)

1. **NAC projection stage** — ISIS `cam2map` (conda/Docker) producing
   co-registered IAU-CRS GeoTIFFs; render integration; co-registration of the
   multi-epoch stack.
2. **HRSC (Mars)** — per-orbit map-projected orthoimages via ESA PSA / DLR.
3. **CTX (Mars)** — static global mosaic (plumbing) then individual RDRs via
   PDS/ODE (time axis).
4. **Moon Trek / QuickMap** — WMTS basemap layer (single-epoch reference).
