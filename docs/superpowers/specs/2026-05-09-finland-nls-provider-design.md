# Finland NLS Provider — Design Spec

Status: brainstormed, awaiting implementation
Date: 2026-05-09
Branch: `worktree-finland-provider`

## Goal

Add a Finland (Maanmittauslaitos / National Land Survey) orthophoto provider to
`satmap_dataset`, slotting into the existing `Provider` ABC alongside the Polish
Geoportal and Swedish Lantmäteriet providers. The first cut is a minimal,
year-aware ingestion pipeline that writes the same `IndexManifest` /
`DatasetManifest` schema the rest of the project consumes.

Source: NLS WCS at `https://avoin-karttakuva.maanmittauslaitos.fi/ortokuvat-ja-korkeusmallit/wcs/v2`,
free with an OmaTili API key (CC BY 4.0).

## Locked decisions

1. **Tile model:** one synthesised WCS GetCoverage URL per (AOI, year). No
   per-sheet enumeration. `tile_sources_by_year[year]` has exactly one entry
   keyed `nls_<year>`.
2. **MVP scope:** WCS-only, `ortokuva_vari` (RGB) only, no fallback. Years not
   covered by WCS for the AOI are excluded from the manifest with a clear
   reason.
3. **AOI cap:** validator refuses `bbox > 2000 m × 2000 m` in EPSG:3067 at
   config time. Larger AOIs are an explicit error, not a silent clamp.

## Non-goals (deferred)

- WMTS / time-aware WMS-v2 fallback for years missing from WCS.
- B/W (`ortokuva_mustavalko`) and CIR (`ortokuva_vaaravari`) coverages.
- Auto-tiling AOIs > 2 km.
- Contract (`sopimus-`) tier with higher zoom.
- OGC API Features per-bbox year filtering.

## Architecture

```
src/satmap_dataset/providers/nls/
├── __init__.py        # exports NlsProvider
├── provider.py        # NlsProvider — index() + download()
├── wcs.py             # GetCapabilities / DescribeCoverage / GetCoverage URL building + XML parsing
└── auth.py            # API-key resolution and Basic Auth header
```

Updates to existing files:
- `providers/__init__.py` — register `"nls"` in `get_provider`
- `models.py` — add `"nls"` to the `provider` Literal; add `"wcs"` to
  `DatasetManifest.mode` Literal
- `config.py` — gate the Finland 2 km validator behind a provider name check
- `tests/` — five new test files mirroring the `test_lantmateriet_*` pattern

## Authentication

Resolution order (first hit wins):

1. `provider_options["api_key"]` (passed via JSON config)
2. Env var `SATMAP_NLS_API_KEY`
3. `.secret` file at the project root (single-line UUID)

**Sent as `?api-key=<KEY>` query parameter on every request.** The NLS open
WCS endpoint rejects HTTP Basic Auth (401) — only the WMS docs mention
Basic; the WCS expects the query-string form. Verified empirically against
the live endpoint. The key is appended at request time so manifest URLs on
disk remain key-free.

## Data flow

### Index stage

1. Construct `IndexConfig`. The Finland validator confirms bbox ≤ 2000 m × 2000 m in EPSG:3067.
2. `wcs.fetch_describe_coverage(coverage_id="ortokuva_vari")` → parse `gml:TimePosition` values → set of available years.
3. `evaluate_year_policy(requested_years, available_years, strict_years, min_years)` — reuses existing helper.
4. For each `year in years_included`, build a GetCoverage URL:

```
https://avoin-karttakuva.maanmittauslaitos.fi/ortokuvat-ja-korkeusmallit/wcs/v2
  ?service=WCS&version=2.0.1&request=GetCoverage
  &CoverageID=ortokuva_vari
  &SUBSET=E(<xmin>,<xmax>)
  &SUBSET=N(<ymin>,<ymax>)
  &SUBSET=time("<YYYY>-12-31T00:00:00.000Z")
  &SubsettingCRS=http://www.opengis.net/def/crs/EPSG/0/3067
  &OutputCRS=http://www.opengis.net/def/crs/EPSG/0/3067
  &format=image/tiff
  &geotiff:compression=LZW
  &geotiff:tiling=true
  &geotiff:tilewidth=256&geotiff:tileheight=256
```

5. Emit `IndexManifest` with `provider="nls"`, single-entry `tile_sources_by_year[year] = {"nls_<year>": <url>}`, plus `provider_metadata` with WCS endpoint, coverage ID, native SRS.

### Download stage

1. Read `IndexManifest`.
2. For each job, stream URL with Basic Auth → `<download_root>/<year>/nls_<year>.tif`. Same async/httpx machinery as Lantmäteriet.
3. No fallback. Failed years recorded in `failed[]`.
4. Emit `DatasetManifest(provider="nls", mode="wcs", years_source_map={year: "wcs"})`.

## Error handling

| Failure | Behaviour |
|---|---|
| 401 Unauthorized | Fail fast with a message pointing at the auth resolution order |
| 400 Bad Request from WCS (oversized AOI, bad subset) | Log response body, fail the year, no retry |
| Empty `DescribeCoverage` time list | Empty `years_available_wfs`, manifest `passed=false` with reason |
| AOI outside Finland | Not detected here; downstream validator's pixel-profile check catches |
| Network error / 5xx | Existing retry policy from `geoportal/http.py` (reused) |

## Testing

All offline; no live network calls.

| File | Coverage |
|---|---|
| `test_nls_config.py` | bbox > 2 km rejected; valid 2 km accepted; non-EPSG:3067 srs rejected |
| `test_nls_auth.py` | `provider_options` > env > `.secret` precedence; Basic Auth header shape |
| `test_nls_wcs_urls.py` | DescribeCoverage and GetCoverage URLs match the documented format byte-for-byte for given inputs |
| `test_nls_index_from_fixture.py` | Saved DescribeCoverage XML → expected year list and manifest population |
| `test_nls_download_policy.py` | Mocked httpx — one request per included year, correct path layout, passed/failed wiring |

Fixture: `tests/fixtures/nls/describe_coverage_ortokuva_vari.xml` (sanitised real response, committed).

## Out-of-scope: render and validate

The render and validate stages are provider-agnostic and unchanged. They consume
`DatasetManifest` files written by the download stage. The downloaded GeoTIFFs
are already in EPSG:3067; existing render code re-projects to the configured
`target_srs`.

## Open questions for implementation

- The `mode` literal currently is `{"wms_tiled", "wfs_render", "hybrid", "stac"}` (Sweden added `"stac"`). Adding `"wcs"` requires touching every place that pattern-matches on mode. Audit this during implementation.
- Whether to expose a `nls` example config under `configs/run/base_nls.json` matching the `base_lantmateriet.json` pattern for parity. Defer to a follow-up if it bloats the diff.
