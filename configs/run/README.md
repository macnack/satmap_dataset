# Run Configs (Base + Location Overrides)

These files are prepared for `run-json` command.
All examples below use relative paths from the repository root (no absolute user-home paths).

## Structure

- `base.json`: common pipeline defaults.
- `locations/*.json`: location-specific fields (`location_name`, `center_lat`, `center_lon`).
- `area_km2` lives in `base.json` by default.
- Output dirs are auto-generated from `location_name` slug:
  - `downloads_<slug>`
  - `rendered_<slug>`
  - `artifacts_<slug>`
  where slug policy is: lowercase, remove diacritics, replace non-alnum with `_`, collapse repeated `_`.

## Build Final Config

```bash
python scripts/merge_json_config.py \
  --base configs/run/base.json \
  --override configs/run/locations/poznan.json \
  --out configs/run/generated/poznan.run.json
```

## Run Pipeline

```bash
python -m satmap_dataset.cli run-json \
  configs/run/generated/poznan.run.json
```

`run-json` supports center-based fields directly and auto-converts them to bbox (`EPSG:2180`).

## Locations Included

- `ladzin.json`
- `poznan.json`
- `spudlow_pole.json`
- `zagan_poligon.json`
- `bagno_lawki_biebrzanski_park.json`
- `sosnowiec.json`
- `goniadz.json`
- `zelazny_most.json`

## Index All Locations (Sequential)

```bash
python scripts/index_all_locations.py \
  --year-start 2015 \
  --year-end 2025
```

Continue through all locations even when some have no WFS years:

```bash
python scripts/index_all_locations.py \
  --year-start 2015 \
  --year-end 2025 \
  --continue-on-error
```

## Run All Locations (No Merge Step)

```bash
python -m satmap_dataset.cli run-all-location-json \
  --locations-dir configs/run/locations \
  --base-json configs/run/base.json \
  --continue-on-error
```

## Index/Download/Validate All Locations (No Merge Step)

```bash
python -m satmap_dataset.cli index-all-location-json \
  --locations-dir configs/run/locations \
  --base-json configs/run/base.json \
  --continue-on-error

python -m satmap_dataset.cli download-all-location-json \
  --locations-dir configs/run/locations \
  --base-json configs/run/base.json \
  --continue-on-error

python -m satmap_dataset.cli validate-all-location-json \
  --locations-dir configs/run/locations \
  --base-json configs/run/base.json \
  --continue-on-error
```

## One Location Example (`bagno_lawki_biebrzanski_park`)

Index only:

```bash
python scripts/merge_json_config.py \
  --base configs/run/base.json \
  --override configs/run/locations/bagno_lawki_biebrzanski_park.json \
  --out configs/run/generated/bagno_lawki_biebrzanski_park.run.json

python -m satmap_dataset.cli index-json \
  configs/run/generated/bagno_lawki_biebrzanski_park.run.json
```

Full pipeline:

```bash
python -m satmap_dataset.cli run-location-json \
  configs/run/locations/bagno_lawki_biebrzanski_park.json \
  --base-json configs/run/base.json
```

Using `just`:

```bash
just index-location-json location_json=configs/run/locations/bagno_lawki_biebrzanski_park.json
just run-location-json location_json=configs/run/locations/bagno_lawki_biebrzanski_park.json
```
