# MARS-LVIG basemap configs (4 places)

Reproduce the four site orthophotos used for trajectory overlay:

| Place | Sequences | Provider | Native / delivered GSD |
|-------|-----------|----------|-------------------------|
| HKairport | `HKairport*` (6) | `landsd_hk` z19 | ~0.28 m (current mosaic) |
| HKisland | `HKisland*` (6) | `landsd_hk` z19 | ~0.28 m (current mosaic) |
| AMtown | `AMtown*` (3) | Armenia Cadastre `Ortho_2021_20cm` | 20 cm product (WMS) |
| AMvalley | `AMvalley*` (3) | Armenia Cadastre `Ortho_2021_20cm` | 20 cm product (WMS) |

Set the dataset root (bags / vio / streams):

```bash
export MARS_LVIG_ROOT=/media/maciej/fifek/mars_lvig   # default if unset
```

## Download all four

```bash
python scripts/download_mars_lvig_maps.py \
  --config configs/run/mars_lvig/places.json \
  --out mars_lvig_maps
```

Outputs under `--out/<place>/`: GeoTIFF, `track.geojson`, `overlay.png`.

## HK only (trajectory CLI)

```bash
python -m satmap_dataset.cli trajectory-json \
  configs/run/mars_lvig/trajectory_hkairport.json

python -m satmap_dataset.cli trajectory-json \
  configs/run/mars_lvig/trajectory_hkisland.json
```

Paths inside the trajectory JSONs use `${MARS_LVIG_ROOT}` — the download script
expands that; for raw `trajectory-json` substitute the absolute track path.
