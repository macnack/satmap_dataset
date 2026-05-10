# `configs/run/generated/` — scratch examples

The `*.run.json` files in this directory are **scratch examples** produced by
`scripts/merge_json_config.py`, which merges a provider base config with a
location override into a single ready-to-run config.

**Important:** every file here was generated on the original author's machine
and contains **absolute filesystem paths** pinned to that checkout (e.g.
`/home/maciej/Github/satmap_dataset/...`). They are checked in to follow the
existing convention, but for any other contributor cloning the repo they are
**un-runnable as-is** and **must be regenerated** for your local checkout.

## Regenerate for your checkout

```bash
python scripts/merge_json_config.py \
  --base configs/run/base_<provider>.json \
  --override configs/run/locations/<location>.json \
  --out configs/run/generated/<location>.run.json
```

Replace `<provider>` with `geoportal`, `lantmateriet`, `sentinel2`, etc., and
`<location>` with the location stem under `configs/run/locations/`.

## Smoke files

The `*_smoke_*.run.json` files (e.g. `kisa_winter_smoke_32633.run.json`) are
CRS sanity-checks used during cross-CRS render development, not user content
or onboarding examples.
