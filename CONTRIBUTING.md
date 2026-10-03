# Contributing

Thanks for helping improve `satmap_dataset`.

## Setup

```bash
# System deps (Linux; required by pyvips)
sudo apt-get update
sudo apt-get install -y libvips42 libvips-tools

python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
# Optional Streamlit UI:
# python -m pip install -e ".[studio]"
```

Prefer `pip install -e ".[dev]"` over `requirements*.txt` (those files mirror
`pyproject.toml` for legacy tooling only).

Optional: copy `.secret.template` → `.secret` for provider API keys (never commit
`.secret`). `direnv` + `.envrc` can load it automatically.

## Tests

```bash
pytest
```

- Default suite is offline (mocked HTTP). Do not set `SATMAP_LIVE_TESTS=1` in CI.
- Some raw-tile / world_window tests skip when `gdalwarp` is missing.
- Live smoke (opt-in): `SATMAP_LIVE_TESTS=1 pytest -m live`

## Pull requests

1. Keep changes focused; match existing stage contracts (`run(config) -> (exit_code, Path)`).
2. New config fields must default-resolve when missing from `base.json` / location JSON.
3. If you change index/download reuse behavior, update predicates in
   `pipeline/rgb_pipeline.py` or `pipeline/run_all.py` (whichever holds them on
   your branch) and add tests.
4. Do not commit downloads, rendered rasters, or credentials.

## Code of conduct

Be respectful in issues and PRs. Harassment or abuse is not welcome.
