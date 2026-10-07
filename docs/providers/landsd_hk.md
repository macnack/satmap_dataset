# LandsD HK Imagery (`landsd_hk`)

**Status: experimental.** Stitches Hong Kong Lands Department Imagery Map API
XYZ PNG tiles into an EPSG:3857 GeoTIFF for evaluation AOIs (e.g. MARS-LVIG
trajectories). The public XYZ service is a **current mosaic** — not year-aware.

## Coverage and GSD

| Zoom | Approx. GSD at HK latitudes |
|------|-----------------------------|
| 18 | ~0.55 m |
| 19 | ~0.28 m (default target via `gsd_m: 0.3`) |

API docs: [Imagery Map API](https://portal.csdi.gov.hk/csdi-webpage/apidoc/ImageryMapAPI).

Open GeoTIFF orthophotos (DOP5000 / TDOP / DOP5000-1982, 0.2–0.3 m) are a
separate DATA.GOV.HK product — not wired here.

## Config

```bash
just run-location-json \
  location_json=configs/run/locations/landsd_hk/hkairport.json \
  base_json=configs/run/base_landsd_hk.json
```

`provider_options`:

| Key | Default | Meaning |
|-----|---------|---------|
| `gsd_m` | `0.3` | Target ground GSD → zoom pick |
| `zoom` | — | Force zoom (overrides `gsd_m`) |
| `imagery_year` | `year_end` | Synthetic year label for the current mosaic |
| `max_tiles` | `1024` | Hard cap on tile count |
| `min_zoom` / `max_zoom` | `15` / `19` | Zoom clamp |

Mode `hybrid` aliases to `wms_tiled`. Requires a projected EPSG CRS (e.g.
`EPSG:3857`, `EPSG:2326`, UTM 50N).

## Multi-year alternatives

True multi-year ~30 cm stacks over HK typically need Esri World Imagery Wayback
via ArcGIS Content Packages / negotiated licence, or an external tool such as
GEHistoricalImagery — not this provider. See [DATA_LICENSING.md](../DATA_LICENSING.md).

## Attribution

Preserve Lands Department attribution when displaying maps (logo + copyright
notice per LandsD Map API terms).
