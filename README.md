# satmap_dataset

`satmap_dataset` is a Python 3 package for building a year-aware orthophoto dataset pipeline.

Current state:
- WFS-first year qualification (`GetCapabilities` + `GetFeature`).
- Real asynchronous TIFF downloading from `url_do_pobrania`.
- Random per-request sleep jitter to avoid synchronized requests.
- Render stage to a shared NN-ready grid (`render` command, pyvips backend).
- JSON artifacts for `index/download/render/validate/run`.

## Scope of Phase 1

- `src/` package layout and installable project metadata.
- Stable CLI subcommands: `index`, `download`, `mosaic`, `validate`, `run`.
- Stable Pydantic models and config contracts.
- Real `index` + `download` implementation.
- Tests for CLI help, model schemas, and year-policy rules.

Current limitations:
- `mosaic` command is now a backward-compatible alias for `render`.

## Installation

```bash
python -m pip install -e ".[dev]"
```

Alternative (requirements files):

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

System dependency (Linux, required by `pyvips`):

```bash
sudo apt-get update
sudo apt-get install -y libvips42 libvips-tools
```

Quick check:

```bash
python -c "import pyvips; print(pyvips.version(0), pyvips.version(1), pyvips.version(2))"
```

## CLI examples

```bash
python -m satmap_dataset.cli --help
python -m satmap_dataset.cli index --year-start 2015 --year-end 2026 --bbox "210300,521900,210500,522100"
python -m satmap_dataset.cli download --index-manifest artifacts/index_manifest.json --sleep-min 0.8 --sleep-max 2.5 --concurrency 4 --profile train
python -m satmap_dataset.cli render --dataset-manifest artifacts/dataset_manifest_download.json --render-root rendered --profile train --px-per-meter 15
python -m satmap_dataset.cli validate --dataset-manifest artifacts/dataset_manifest_render.json --year 2015 --year 2026
python -m satmap_dataset.cli run --year-start 2015 --year-end 2026 --bbox "210300,521900,210500,522100" --sleep-min 0.8 --sleep-max 2.5 --concurrency 4 --render-root rendered --profile train
```

All command examples in this README use relative paths from the repository root.

## Base + Location JSON (Relative Path Examples)

Index all locations:

```bash
python -m satmap_dataset.cli index-all-location-json \
  --locations-dir configs/run/locations \
  --base-json configs/run/base.json \
  --continue-on-error
```

Index one location (`bagno_lawki_biebrzanski_park`):

```bash
python scripts/merge_json_config.py \
  --base configs/run/base.json \
  --override configs/run/locations/bagno_lawki_biebrzanski_park.json \
  --out configs/run/generated/bagno_lawki_biebrzanski_park.run.json

python -m satmap_dataset.cli index-json \
  configs/run/generated/bagno_lawki_biebrzanski_park.run.json
```

Run one location (full pipeline):

```bash
python -m satmap_dataset.cli run-location-json \
  configs/run/locations/bagno_lawki_biebrzanski_park.json \
  --base-json configs/run/base.json
```

Same operations with `just`:

```bash
just index-all-json
just index-all-json locations_4
just run-all locations_4
just run-all locations_dir=locations_4
just summary-locations locations_4
just summary-locations location_9
just summary-locations
just index-location-json location_json=configs/run/locations/bagno_lawki_biebrzanski_park.json
just run-location-json location_json=configs/run/locations/bagno_lawki_biebrzanski_park.json
```

`just summary-locations` without arguments auto-selects:
- one `locations*` directory if exactly one exists under `SATMAP_LOCATIONS_ROOT`,
- interactive picker when multiple are found in a TTY,
- fallback to `SATMAP_LOCATIONS_DIR` (or `SATMAP_LOCATIONS_ROOT/locations`) in non-interactive mode.

Useful env variables (`.envrc`):

```bash
SATMAP_LOCATIONS_ROOT   # default: $PWD/configs/run
SATMAP_LOCATIONS_DIR    # default: $SATMAP_LOCATIONS_ROOT/locations
SATMAP_BASE_JSON        # default: $SATMAP_LOCATIONS_ROOT/base.json
```

No manifest/registration is needed for `locations_2`, `locations_4`, etc.; directory name is enough.
Short singular aliases like `location_9` are also accepted and mapped to `locations_9` when that directory exists.

Each command prints the generated JSON artifact path and exits with code:
- `0` for success,
- `1` for policy/data failure,
- `2` for invalid CLI/config arguments.

`download`/`run` jitter options:
- `--sleep-min`, `--sleep-max`: random delay before each request
- `--concurrency`: worker count
- `--retries`, `--retry-delay`, `--timeout`

`render` options:
- `--profile` (`train` or `reference`)
- `--px-per-meter` (used for geometry-driven output size)
- `--target-width`, `--target-height` (optional explicit override)
- `--auto-size-from-bbox` / `--no-auto-size-from-bbox`
- `--target-bbox` (defaults to index bbox)
- `--target-srs` (default `EPSG:2180`)
- `--resample-method` (`bilinear`/`nearest`)
- `--tile-size`, `--compression deflate`, `--overview-level`
- `--wms-fallback-missing-years` / `--no-wms-fallback-missing-years`
- `--disable-color-norm`

## Reference Parity Mode (match download_map.py)

Use this mode to debug color and geometry against WMS reference output.

Key behaviors:
- output size is computed from bbox and `px_per_meter` (for `200m x 200m` and `15 px/m`: `3000x3000`)
- years missing in WFS can use WMS fallback (`StandardResolutionTime`)
- output manifest includes `years_source_map`, `coverage_ratio_by_year`, `color_qc_by_year`

Example:

```bash
python -m satmap_dataset.cli run \
  --year-start 2015 --year-end 2023 \
  --bbox "210300,521900,210500,522100" \
  --profile reference \
  --px-per-meter 15 \
  --wms-fallback-missing-years \
  --no-experimental-per-year-color-norm \
  --render-root rendered_reference
```

## Training-ready output folder

After `run`, use `rendered/` as input to your dataset class:

```python
dataset = SatelliteSeasonalHomographyDataset(
    maps_path="rendered",  # profile=train output
    num_samples=1000,
)
```

The folder contains `year_YYYY.tif` with consistent width/height and `RGB_U8` profile.

## Sweden / Lantmäteriet provider

The pipeline supports two providers, selected via `--provider` (CLI) or the
`provider` field in JSON configs:

- `geoportal` (default) — Polish PZGiK WFS + WMS.
- `lantmateriet` — Swedish Lantmäteriet STAC API for annual orthophotos
  (Ortofoto Visning Årsvisa, 2006 onwards), with optional WMS fallback.

The two providers share the same downstream stages, so `index → download → render → validate`
and the `run`/`run-json`/`run-location-json` commands work identically — only
the data source changes. EPSG defaults differ: Geoportal uses `EPSG:2180` and
Lantmäteriet uses `EPSG:3006` (SWEREF 99 TM).

> *Ortofoto Nedladdning* (the STAC product) is the **primary** source. As of
> 2026 its license fee is **0 SEK**, but you still need a free Geotorget
> account and an active subscription before `dl1.lantmateriet.se` will return
> assets to your Basic-Auth requests. *Ortofoto Visning, Årsvisa* (the annual
> WMS view service) is a **paid** product (~1.66 MSEK/year as of 2026); we
> only allow it as an opt-in fallback (`wms_fallback_missing_years: true` plus
> an explicit `provider_options.wms_url` and `provider_options.wms_layer`) and
> it is **off by default**. Preserve the `© Lantmäteriet` attribution that the
> STAC item metadata carries.

### Configure the provider

Per-call options live in `provider_options` on the run/index/download config:

| Key                  | Purpose                                                |
| -------------------- | ------------------------------------------------------ |
| `stac_url`           | STAC `/search` endpoint                                |
| `stac_collection`    | Collection ID (string or list)                         |
| `wms_url`            | Annual WMS GetMap endpoint                             |
| `wms_layer`          | WMS layer name                                         |
| `wms_version`        | WMS version (default `1.3.0`)                          |
| `year_policy`        | `exact_only`, `nearest_before`, `nearest_after`, or `nearest_any_with_max_delta` |
| `max_year_delta`     | Required when `year_policy = nearest_any_with_max_delta` |
| `api_key`            | Bearer token for STAC / WMS                            |

Each `provider_options` key has an environment-variable fallback:
`SATMAP_LANTMATERIET_STAC_URL`, `SATMAP_LANTMATERIET_STAC_COLLECTION`,
`SATMAP_LANTMATERIET_WMS_URL`, `SATMAP_LANTMATERIET_WMS_LAYER`,
`SATMAP_LANTMATERIET_API_KEY`, `SATMAP_LANTMATERIET_USERNAME`,
`SATMAP_LANTMATERIET_PASSWORD`.

`dl1.lantmateriet.se` requires Geotorget API credentials (a `lm_…`-style
username plus generated password issued per subscribed product, distinct from
your `lantmateriet.se` website login). 401 means missing/invalid credentials;
403 means the subscription is not yet active for the requested asset. The
downloader treats 4xx as non-retryable — no point burning the tile budget on a
permission error.

### Generate a 2 km bbox around Kisa

```bash
python scripts/make_bbox.py \
  --center-lat 57.985 \
  --center-lon 15.629 \
  --size-meters 2000 \
  --target-srs EPSG:3006
```

The script prints the bbox in EPSG:3006, a WGS84 GeoJSON polygon, and a ready-
to-paste `python -m satmap_dataset.cli run` invocation. It uses `pyproj` if
installed; otherwise it falls back to the system `proj` CLI.

### Index 2010–2024 over Kisa

```bash
python -m satmap_dataset.cli index \
  --provider lantmateriet \
  --year-start 2010 \
  --year-end 2024 \
  --bbox "536194.910,6426213.280,538194.910,6428213.280" \
  --srs EPSG:3006
```

### Run the full pipeline for Kisa

```bash
python -m satmap_dataset.cli run \
  --provider lantmateriet \
  --year-start 2010 \
  --year-end 2024 \
  --bbox "536194.910,6426213.280,538194.910,6428213.280" \
  --srs EPSG:3006 \
  --target-srs EPSG:3006 \
  --sleep-min 0.8 \
  --sleep-max 2.5 \
  --concurrency 4 \
  --render-root rendered_kisa \
  --profile train
```

### Base + location JSON

Pre-built configs live alongside the Polish ones:

- `configs/run/base_lantmateriet.json` — defaults for the Sweden flow.
- `configs/run/locations/kisa_sweden_2km.json` — Kisa center + 4 km² square.

```bash
python scripts/merge_json_config.py \
  --base configs/run/base_lantmateriet.json \
  --override configs/run/locations/kisa_sweden_2km.json \
  --out configs/run/generated/kisa_sweden_2km.run.json

python -m satmap_dataset.cli run-json configs/run/generated/kisa_sweden_2km.run.json
```

To enable the annual WMS fallback for years missing from STAC, set
`wms_fallback_missing_years: true` in the merged config (or pass
`--wms-fallback-missing-years` on the CLI). Falling back will still write
`years_source_map[year] = "wms_fallback"` so downstream stages know the
provenance per year.

## Sentinel-2 / Element84 Earth Search provider

A third provider, `sentinel2`, fetches Sentinel-2 L2A scenes from the
[Element84 Earth Search](https://earth-search.aws.element84.com/v1/) STAC
API. Unlike the Lantmäteriet and Geoportal flows, Sentinel-2 covers the
**whole year, all seasons, including winter** at 10 m / px (3-band TCI
COG). No auth required; assets sit on anonymous S3 (`sentinel-cogs.s3.us-
west-2.amazonaws.com`).

> Sentinel-2 scenes ship in their native MGRS UTM zone (e.g. EPSG:32633,
> EPSG:32635). The render stage now reprojects via `gdalwarp` when the
> source CRS differs from `target_srs`, with a single warp that clips +
> resamples to the AOI grid in one pass. Install GDAL (`gdalwarp` on
> PATH) before running cross-CRS jobs.

### Year selection

Sentinel-2 revisits every ~5 days, so each requested year has many
candidate scenes rather than a single annual capture. The provider picks
**one representative per year** that minimises the distance from a
target day-of-year (default Feb 15, suitable for winter pairs vs the
Lantmäteriet summer renders) and stays under a cloud-cover threshold:

| Key | Default | Purpose |
| --- | --- | --- |
| `provider_options.target_month` | `2` | Target month in DOY distance |
| `provider_options.target_day` | `15` | Target day in DOY distance |
| `provider_options.max_cloud_cover_pct` | `25.0` | Drop scenes with `eo:cloud_cover` above this |
| `provider_options.preferred_asset_key` | `"visual"` | Asset to download (the 10 m TCI COG) |

Set `max_cloud_cover_pct: null` to disable filtering, or move `target_day`
to mid-July for a leaf-on summer pair.

### Run a winter Kisa scene paired with Lantmäteriet

```bash
python scripts/merge_json_config.py \
  --base configs/run/base_sentinel2.json \
  --override configs/run/locations/kisa_winter.json \
  --out configs/run/generated/kisa_winter.run.json

python -m satmap_dataset.cli run-json configs/run/generated/kisa_winter.run.json
```

The output `rendered_kisa_winter/year_<Y>.tiff` lands at the same
EPSG:3006 grid as the Lantmäteriet summer renders for the same AOI, so
the two stacks can be loaded together as season-pair training data.

### Run Helsinki in EPSG:3067 (Finland)

```bash
python scripts/merge_json_config.py \
  --base configs/run/base_sentinel2.json \
  --override configs/run/locations/helsinki_winter.json \
  --out configs/run/generated/helsinki_winter.run.json

python -m satmap_dataset.cli run-json configs/run/generated/helsinki_winter.run.json
```

### Cross-CRS notes

- The provider warps to whatever `target_srs` you set. EPSG:3006 (Sweden),
  EPSG:3067 (Finland TM35FIN) and any UTM zone (`EPSG:326NN` / `EPSG:327NN`)
  are supported out of the box; the `.prj` sidecar is auto-generated for
  UTM zones.
- Reprojected source tiles are cached next to the downloaded COG under
  `_reprojected/<src_stem>__epsg<N>_<sha>.tif` keyed by target_srs +
  bbox + pixel size, so different AOIs / resolutions don't fight over
  the same cache file.
- For a Kisa-sized 2 km AOI the cache footprint is ~240 KB per
  (year, target CRS).

### Attribution

Sentinel-2 imagery is published under the
[Copernicus open data licence](https://sentinels.copernicus.eu/web/sentinel/terms-conditions).
Cite as: *Contains modified Copernicus Sentinel data [year]*.

## Development checks

```bash
pytest
```
