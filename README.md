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

## Development checks

```bash
pytest
```
