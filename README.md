# satmap_dataset

Year-aware multi-provider orthophoto dataset builder for ML / remote-sensing research.
Probe availability, download TIFFs, render co-registered year stacks, and emit JSON
manifests at every stage.

**Code license:** [MIT](LICENSE) · **Data:** see [docs/DATA_LICENSING.md](docs/DATA_LICENSING.md)

## Features

- Pipeline: **index → download → render → validate** (optional DEM / OSM / raw-export)
- Providers: Geoportal (PL), Lantmäteriet (SE), NLS (FI), Sentinel-2, LROC NAC (lunar)
- Shared NN-ready grid (`RGB_U8` GeoTIFF) with Pydantic manifest contracts
- Location JSON + `just` recipes for batch runs
- Optional Streamlit UI (`satmap-studio`)

## Provider maturity

| Provider | Status | Notes |
|----------|--------|-------|
| `geoportal` | **Supported** | Polish PZGiK WFS + WMS; default path |
| `lantmateriet` | **Supported** | Swedish STAC; Geotorget credentials |
| `nls` | **Supported** | Finnish WCS; API key required; dedicated `nls-*-json` CLIs stop at download |
| `sentinel2` | **Experimental** | Earth Search COGs; set `target_srs` / GDAL for cross-CRS |
| `lroc_nac` | **Deferred** | PDS index + download only; ISIS projection/render out of scope |

## Quick start

```bash
sudo apt-get update && sudo apt-get install -y libvips42 libvips-tools
python -m pip install -e ".[dev]"

pytest   # offline suite

# Tiny Geoportal run (EPSG:2180 bbox)
python -m satmap_dataset.cli run \
  --year-start 2015 --year-end 2020 \
  --bbox "210300,521900,210500,522100" \
  --profile train --render-root rendered
```

Each CLI command prints the absolute artifact path as its **last stdout line**.
Exit codes: `0` success, `1` policy/data failure, `2` invalid config.

## Configuration

Defaults live in `configs/run/base.json`. Locations are small JSON files under
`configs/run/locations/` (`location_name`, `center_lat` / `center_lon`, area).

```bash
just run-location-json location_json=configs/run/locations/poznan.json
# or
python -m satmap_dataset.cli run-location-json \
  configs/run/locations/poznan.json \
  --base-json configs/run/base.json
```

Env vars (see `.envrc`):

| Variable | Purpose |
|----------|---------|
| `SATMAP_LOCATIONS_ROOT` | Config root (default `configs/run`) |
| `SATMAP_LOCATIONS_DIR` | Locations directory |
| `SATMAP_BASE_JSON` | Base defaults JSON |
| `SATMAP_RAW_ROOT` | Raw-tile export root (default `~/sat_data_raw`) |
| `SATMAP_GMIX_DEST` | Flattened gmix dest (default `~/sat_data`) |
| `SATMAP_NLS_API_KEY` | NLS API key |
| `SATMAP_LANTMATERIET_*` | Lantmäteriet credentials |

Copy `.secret.template` → `.secret` for local keys (gitignored).

## Pipeline & artifacts

| Stage | Artifact |
|-------|----------|
| index | `artifacts_*/index_manifest.json` |
| download | `dataset_manifest_download.json` + `downloads_*/` |
| render | `dataset_manifest_render.json` + `rendered_*/year_YYYY.tif` |
| validate | `validation_report.json` |

Multimodal stack (RGB + DEM + OSM): `location-run-json` / Studio.
Architecture details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

Be gentle with Geoportal: requests use randomized sleep jitter by default
(`--sleep-min` / `--sleep-max`).

## Provider notes (summary)

**Geoportal (default)** — WFS-first hybrid mode with WMS fallback for missing years.
Profiles: `train` (default) and `reference` (geometry-driven sizing + extra QC fields).

**NLS** — free API key; WCS 2 km request cap (auto grid). License often CC BY 4.0.

```bash
python -m satmap_dataset.cli nls-run-json configs/run/base_nls.json
```

**Lantmäteriet** — STAC primary (`mode: "stac"` / `hybrid`); index picks the
best-covering item per year with GSD + attribution, uses Geotorget Basic auth
for search and download, and geotags WMS fallbacks in `EPSG:3006`. Preserve
`© Lantmäteriet` attribution. See `configs/run/base_lantmateriet.json` and
`configs/run/locations/kisa_sweden_2km.json`.

Optional **Swedish DEM** (Markhöjdmodell 1 m DTM) uses a **different** national
API — `https://api.lantmateriet.se/stac-hojd/v1` (collection `dtm-cog`) — not
ortofoto `stac-bild` and not Polish Geoportal WCS. Enable with
`provider=lantmateriet`, `transport=stac_hojd`, `products=["nmt"]`,
`vertical_datum=rh2000` (Studio DEM checkbox is opt-in for Sweden). Asset
download needs a Geotorget **Markhöjdmodell Nedladdning** subscription; set
`SATMAP_LANTMATERIET_DEM_USERNAME`/`PASSWORD` (or shared
`SATMAP_LANTMATERIET_*` if that account is entitled). See
`docs/DATA_LICENSING.md`.

**Sentinel-2** — Element84 Earth Search; Copernicus terms apply. Prefer one
representative scene per year via `provider_options` (cloud cover, target DOY).

**LROC NAC** — lunar CRS `IAU_2015:30100`; index/download only until projection lands.

## Advanced: gmix / raw-export (sat_roma handoff)

Opt-in path for mixed-GSD co-registered cells (skips render):

```bash
just gmix location_json=configs/run/locations/wroclaw_15km2.json
```

Defaults write under `$SATMAP_RAW_ROOT` (`~/sat_data_raw`) then flatten to
`$SATMAP_GMIX_DEST` (`~/sat_data`). See `configs/run/locations/wroclaw_15km2.json`
for `cell_mode: "world_window"`.

## Trajectory tiles

GPS track → 1 km windows in EPSG:2180 (+ optional orthophoto download):

```bash
python -m satmap_dataset.cli trajectory --track path/to/gps_001 --out trajectory_gps001
python -m satmap_dataset.cli trajectory --track path/to/gps_001 --out trajectory_gps001 --download
```

## Web UI

```bash
python -m pip install -e ".[dev,studio]"
just studio
```

Map AOI picker, year/GSD probe, location-run (RGB + DEM + OSM), optional raw-export.

## Development

```bash
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md). CI runs on Python 3.10–3.12 with libvips.

## Docs map

| Doc | Audience |
|-----|----------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Contributors |
| [docs/DATA_LICENSING.md](docs/DATA_LICENSING.md) | Anyone downloading data |
| [SECURITY.md](SECURITY.md) | Credentials / reporting |
| [CHANGELOG.md](CHANGELOG.md) | Releases |
| [CLAUDE.md](CLAUDE.md) | Agent / deep contracts |
| [docs/superpowers/](docs/superpowers/) | Internal planning archive only |

## Acknowledgments

Polish GUGiK / Geoportal, Lantmäteriet, Maanmittauslaitos (NLS), Copernicus
Sentinel program, NASA LROC / PDS, OpenStreetMap contributors. Raw-tile ingest
core is ported from sat_roma (`romatch/datasets/raw_tiles.py`) with a satmap-only
`world_window` extension.
