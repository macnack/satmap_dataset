set shell := ["bash", "-cu"]

default:
  @just --list

# Install package in editable mode with dev extras.
install:
  python -m pip install -e ".[dev]"

# Run all tests.
test:
  pytest

# Merge base + location config into generated run-json.
merge location="poznan":
  mkdir -p configs/run/generated
  python scripts/merge_json_config.py \
    --base configs/run/base.json \
    --override configs/run/locations/{{location}}.json \
    --out configs/run/generated/{{location}}.run.json

# Run pipeline from a generated run-json.
run-json location="poznan":
  loc="{{location}}"; loc="${loc#location=}"; python -m satmap_dataset.cli run-json "configs/run/generated/${loc}.run.json"

# Index from a generated run-json.
index-json location="poznan":
  loc="{{location}}"; loc="${loc#location=}"; python -m satmap_dataset.cli index-json "configs/run/generated/${loc}.run.json"

# Run full pipeline for one location JSON + base JSON.
run-location-json location_json="configs/run/locations/poznan.json" base_json="configs/run/base.json":
  loc="{{location_json}}"; loc="${loc#location_json=}"; base="{{base_json}}"; base="${base#base_json=}"; python -m satmap_dataset.cli run-location-json "$loc" --base-json "$base"

# Index one location JSON + base JSON.
index-location-json location_json="configs/run/locations/poznan.json" base_json="configs/run/base.json":
  loc="{{location_json}}"; loc="${loc#location_json=}"; base="{{base_json}}"; base="${base#base_json=}"; name="$(basename "$loc" .json)"; mkdir -p configs/run/generated; python scripts/merge_json_config.py --base "$base" --override "$loc" --out "configs/run/generated/${name}.run.json"; python -m satmap_dataset.cli index-json "configs/run/generated/${name}.run.json"

# Index all locations from configs/run/locations.
index-all year-start="2015" year-end="2025":
  python scripts/index_all_locations.py \
    --year-start {{year-start}} \
    --year-end {{year-end}}

# Full run for all locations (base + location JSON mode).
run-all locations_dir="configs/run/locations" base_json="configs/run/base.json" continue_on_error="--continue-on-error":
  dir="{{locations_dir}}"; dir="${dir#locations_dir=}"; base="{{base_json}}"; base="${base#base_json=}"; python -m satmap_dataset.cli run-all-location-json --locations-dir "$dir" --base-json "$base" {{continue_on_error}}

# Index all locations (base + location JSON mode).
index-all-json locations_dir="configs/run/locations" base_json="configs/run/base.json" continue_on_error="--continue-on-error":
  dir="{{locations_dir}}"; dir="${dir#locations_dir=}"; base="{{base_json}}"; base="${base#base_json=}"; python -m satmap_dataset.cli index-all-location-json --locations-dir "$dir" --base-json "$base" {{continue_on_error}}

# Download all locations (base + location JSON mode).
download-all-json locations_dir="configs/run/locations" base_json="configs/run/base.json" continue_on_error="--continue-on-error":
  dir="{{locations_dir}}"; dir="${dir#locations_dir=}"; base="{{base_json}}"; base="${base#base_json=}"; python -m satmap_dataset.cli download-all-location-json --locations-dir "$dir" --base-json "$base" {{continue_on_error}}

# Validate all locations (base + location JSON mode).
validate-all-json locations_dir="configs/run/locations" base_json="configs/run/base.json" continue_on_error="--continue-on-error":
  dir="{{locations_dir}}"; dir="${dir#locations_dir=}"; base="{{base_json}}"; base="${base#base_json=}"; python -m satmap_dataset.cli validate-all-location-json --locations-dir "$dir" --base-json "$base" {{continue_on_error}}
