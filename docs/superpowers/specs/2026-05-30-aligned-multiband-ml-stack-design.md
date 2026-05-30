# Design: Aligned multi-band ML stack (RGB + NMT + NMPT + nDSM + valid_mask)

**Date:** 2026-05-30
**Status:** Approved (pending spec review)
**Builds on:** the WCS/skorowidz DEM features + the existing ortho render pipeline.

## Goal

Produce a single **float32 multi-band GeoTIFF** per location that stacks the orthophoto
RGB with the elevation layers (NMT/DTM, NMPT/DSM, nDSM) on the **same grid**, so an ML
dataloader reads **one file** and gets all channels as `(C, H, W)`. A JSON sidecar
describes the bands and their provenance (which source file/year/datum feeds each band).

Band order (default): `[red, green, blue, nmt, nmpt, ndsm, valid_mask]` (7 bands).
- Values are **raw**: RGB as `0–255` floats, elevations in metres, `nDSM = NMPT − NMT`.
- Invalid pixels in value bands are filled with `0.0`.
- A single shared **`valid_mask`** band is `1.0` where RGB **and** NMT **and** NMPT are
  all valid, else `0.0`.

## Decisions (from brainstorming)

| Decision | Choice |
|----------|--------|
| Output format | One `float32` multi-band GeoTIFF + JSON sidecar (manifest) |
| Bands | `rgb, nmt, nmpt, ndsm` (configurable subset; default all) + `valid_mask` always appended |
| Inputs | **Auto-discovery by location** (find rendered ortho + DEM files in the slug dirs) |
| Encoding | Raw values; invalid filled `0.0`; **single shared `valid_mask`** band |
| Reference grid | The orthophoto RGB raster (extent / W×H / CRS); NMT & NMPT resampled to it |
| nDSM | `NMPT − NMT` after both are aligned to the RGB grid |
| Downloads | None — assumes ortho render + DEM downloads already exist for the location |

## Inputs & auto-discovery

Given `location_name` (→ slug), `rgb_year`, `nmt_year`, `nmpt_year`, `vertical_datum`,
resolve:
- RGB: `rendered_<slug>/year_<rgb_year>.tiff` (3-band `RGB_U8` from the ortho render).
- NMT: `dem_<slug>/skorowidz/nmt_<datum>/native/year_<nmt_year>.tif`.
- NMPT: `dem_<slug>/skorowidz/nmpt_<datum>/native/year_<nmpt_year>.tif`.

A missing required input → exit non-zero with an actionable message naming the file and
the command to produce it (`render-location-json` / `dem ... --transport skorowidz`).
The `render_root`/`dem_root` are derived from the slug via the existing
`_apply_location_paths_policy`, overridable.

## Components

### 1. `src/satmap_dataset/config.py` — `StackConfig`

- `location_name: str` (required; drives slug + discovery).
- `rgb_year: int` (required).
- `nmt_year: int | None`, `nmpt_year: int | None`.
- `vertical_datum: str` ∈ `{evrf2007, kron86}`, default `evrf2007`.
- `bands: list[str]` ⊆ `{rgb, nmt, nmpt, ndsm}`, default all; validated/deduped.
  `valid_mask` is always emitted as the final band (not listed in `bands`).
- `resample: str` ∈ `{bilinear, nearest}`, default `bilinear`.
- `render_root`/`dem_root: Path` (auto from slug).
- `output_json: Path` (the stack manifest; default `stack_<slug>/stack_<rgb_year>.json`)
  and the GeoTIFF written next to it as `stack_<slug>/stack_<rgb_year>.tif`.
- `fill_value: float = 0.0`, `nodata_in: float = -9999.0`, `provider_options: dict`.

Validators: `bands` non-empty subset; if `ndsm` ∈ bands then both `nmt_year` and
`nmpt_year` must be set; if `nmt`/`nmpt` ∈ bands the matching year must be set;
`vertical_datum`/`resample` enums.

### 2. `src/satmap_dataset/pipeline/dem_stack.py`

`run(config: StackConfig) -> tuple[int, Path]`. Steps (GDAL CLI + numpy/tifffile — no
new deps; mirrors the `dem.py` GDAL posture: capture stderr, clear error if GDAL missing):

1. **Discover** input paths; error if a required one is absent.
2. **Reference grid** from the RGB raster: read `(xmin, ymin, xmax, ymax, width, height,
   epsg)` via a small `_raster_grid(path)` helper (`gdalinfo -json`, parsed).
3. **Align** NMT and NMPT to that grid with `gdalwarp -t_srs EPSG:<epsg> -te xmin ymin
   xmax ymax -ts W H -r <resample> -dstnodata <nodata_in>` → temp single-band float32.
4. **Assemble** (numpy + tifffile): read RGB `(3,H,W)`, aligned NMT/NMPT `(H,W)`.
   - `ndsm = nmpt - nmt`.
   - `valid = isfinite(nmt) & (nmt != nodata_in) & isfinite(nmpt) & (nmpt != nodata_in)`
     (RGB assumed full over the render grid; if the RGB has a nodata, AND it in too).
   - Fill invalid pixels of `nmt/nmpt/ndsm` with `fill_value`.
   - Build the ordered band list per `config.bands` (rgb→3 bands), append
     `valid_mask = valid.astype(float32)`. Stack → `(C,H,W) float32`.
5. **Write** the array to a plain multi-band TIFF (tifffile), then **georeference** it with
   `gdal_translate -a_srs EPSG:<epsg> -a_ullr xmin ymax xmax ymin -co COMPRESS=DEFLATE`
   → `stack_<slug>/stack_<rgb_year>.tif` (georeferenced float32 multi-band).
6. Write the `StackManifest` sidecar JSON. Return `(0, output_json)`.

Helpers are seams for testing: `_raster_grid`, `_align_to_grid` (reuse `dem._align_to_grid`
where possible), `_assemble_stack` (pure numpy, fully unit-testable), `_georeference`.

### 3. `src/satmap_dataset/models.py` — manifest

- `StackBandDescriptor`: `index:int` (1-based), `name:str`, `role:str`
  (`rgb`/`dtm`/`dsm`/`object_height`/`mask`), `unit:str`, `source:str | None`,
  `year:int | None`, `datum:str | None`, `derived:str | None`.
- `StackManifest`: `kind="ml_stack"`, `generated_at`, `provider`, `location_name`,
  `stack_path:str`, `crs:str`, `width:int`, `height:int`, `dtype="float32"`,
  `fill_value:float`, `bands:list[StackBandDescriptor]`,
  `normalization_hint:dict[str,str]`, `passed:bool`, `run_parameters:dict`.

The manifest IS the band sidecar — it carries full provenance (per band: source file,
year, datum), so the user can verify which acquisition feeds each channel.

### 4. CLI (`src/satmap_dataset/cli.py`)

- `dem-stack-json <params.json>` — maps onto `StackConfig`.
- `dem-stack-location-json <location.json> [--base-json …] [--rgb-year …] [--nmt-year …]
  [--nmpt-year …] [--vertical-datum …]` — merges base+location, auto-discovers paths via a
  new `_build_stack_config_from_base_and_location`, writes to `stack_<slug>/`.
  Years may come from the location/base JSON or the explicit flags (flags win).
- Last stdout line = the manifest path; exit 0 on success, 1 on data failure, 2 on bad
  config.

### 5. `Justfile` + roots + gitignore

```just
# Build an aligned multi-band ML stack (RGB + NMT + NMPT + nDSM + mask) for a location
dem-stack location_json rgb_year nmt_year nmpt_year:
  python -m satmap_dataset.cli dem-stack-location-json {{location_json}} --rgb-year {{rgb_year}} --nmt-year {{nmt_year}} --nmpt-year {{nmpt_year}}
```

Add `stack_*/` to `.gitignore` and a `stack` kind (`stack_root`) to
`scripts/manage_location_roots.py`, consistent with `downloads_`/`rendered_`/`artifacts_`/`dem_`.

## Output

```
stack_<slug>/stack_<rgb_year>.tif    # float32, bands [R,G,B,NMT,NMPT,nDSM,valid_mask]
stack_<slug>/stack_<rgb_year>.json   # StackManifest (band order + provenance + norm hints)
```

Dataloader usage: read the GeoTIFF → `(C,H,W) float32`; read the JSON for band order,
units, `fill_value`, and `valid_mask` index; normalise per the hints (e.g. RGB `/255`,
elevations z-scored per dataset). Mask out `valid_mask == 0` in the loss.

## Data flow

```
location + years + datum
  → discover RGB(rendered_<slug>), NMT/NMPT(dem_<slug>)
  → grid := RGB raster (extent/W×H/CRS)
  → gdalwarp NMT,NMPT → grid
  → numpy assemble [R,G,B,NMT,NMPT,nDSM, valid_mask], fill invalid=0
  → tifffile write plain → gdal_translate -a_srs -a_ullr → stack_<slug>/stack_<year>.tif
  → StackManifest sidecar (provenance)
```

## Error handling

- Missing input file → exit 1 with the path + how to produce it.
- Missing GDAL CLI (`gdalwarp`/`gdal_translate`/`gdalinfo`) → clear `RuntimeError`.
- Grid mismatch impossible (everything resampled to the RGB grid).
- `ndsm` requested without both NMT & NMPT → config validation error (exit 2).

## Testing (GDAL/network mocked; numpy real)

- `_assemble_stack` (pure): given RGB `(3,h,w)`, NMT, NMPT arrays + nodata → correct
  band order, `ndsm == nmpt-nmt` on valid pixels, invalid filled `0.0`, `valid_mask`
  = intersection; band subset honored; mask always last.
- `_raster_grid` parses a `gdalinfo -json` fixture into `(xmin,ymin,xmax,ymax,W,H,epsg)`.
- `StackConfig` validation (bands subset; ndsm requires both years; enums).
- `dem_stack.run` with mocked `_align_to_grid`/`_georeference`/discovery → writes the
  manifest with correct band descriptors + provenance; missing input → exit 1.
- `StackManifest`/`StackBandDescriptor` JSON round-trip.
- CLI: `dem-stack-json` writes manifest + prints path last; location builder discovers
  paths under `rendered_<slug>`/`dem_<slug>` and writes to `stack_<slug>`; bad config → 2.
- Live smoke (outside sandbox): Poznań RGB 2024 + NMT 2024 + NMPT 2024 → a 7-band float32
  GeoTIFF; verify in QGIS / numpy that bands align and `valid_mask` matches coverage.

## Known limitations / notes

- Single shared `valid_mask` (RGB∧NMT∧NMPT). Per-band masks are a possible future option;
  for now a pixel missing any layer is masked everywhere.
- RGB is cast to `float32` (4× the `uint8` size); accepted for single-file convenience.
- RGB validity assumes the render fills the grid; if a future render emits an explicit
  nodata/alpha, AND it into the mask.
- The stack is read-only over existing render + DEM outputs; it does not download or
  render. Run `render-location-json` and the skorowidz `dem` download first.
- Dev sandbox blocks geoportal but the stack step is offline (operates on local files);
  only the upstream render/dem downloads need to run outside the sandbox.
