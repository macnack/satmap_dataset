# Design: Optional Swedish DEM (Markhöjdmodell via STAC-höjd)

**Date:** 2026-10-03  
**Status:** Implemented (opt-in; download needs Geotorget credentials)

## Goal

Add an optional national Swedish elevation path that does **not** pretend Polish
Geoportal WCS/NMPT works for Sweden and does **not** overload the Lantmäteriet
ortofoto STAC collection (`stac-bild`).

## Chosen source

| Item | Choice |
|------|--------|
| Product | **Markhöjdmodell Nedladdning** (1 m DTM / terrain model) |
| API | STAC catalog `https://api.lantmateriet.se/stac-hojd/v1` |
| Default collection | `dtm-cog` (national COG index; regional `mhm-*` also exist) |
| Why not WCS | Same auth wall; STAC matches existing Lantmäteriet client patterns and stays clearly separate from ortofoto |
| Why not ortofoto STAC | Different catalog (`stac-hojd` vs `stac-bild`); different Geotorget product |

Catalog/search is publicly readable (CC BY 4.0 metadata). Asset bytes on
`dl1.lantmateriet.se/hojd/...` require an authorized Geotorget account for
Markhöjdmodell (ortofoto credentials typically return 401).

## Config surface

```python
DemConfig(
    provider="lantmateriet",
    transport="stac_hojd",       # auto-defaulted when provider=lantmateriet
    products=["nmt"],            # DTM only; nmpt rejected
    vertical_datum="rh2000",
    srs="EPSG:3006",
    bbox="...",
    provider_options={
        # optional overrides:
        # "stac_hojd_url": "...",
        # "stac_hojd_collection": "dtm-cog",
    },
)
```

Env (see `.secret.template`):

- `SATMAP_LANTMATERIET_DEM_USERNAME` / `SATMAP_LANTMATERIET_DEM_PASSWORD` (preferred)
- fallback: `SATMAP_LANTMATERIET_USERNAME` / `PASSWORD`
- optional: `SATMAP_LANTMATERIET_STAC_HOJD_URL`, `SATMAP_LANTMATERIET_STAC_HOJD_COLLECTION`

## Components

1. `providers/lantmateriet/dem.py` — STAC-höjd option resolution + asset selection  
2. `pipeline/dem_lantmateriet.py` — search → download → merge/clip → optional align → `LayerManifest`  
3. `pipeline/dem.run` dispatches on `provider=lantmateriet` / `transport=stac_hojd`  
4. `LantmaterietProvider.dem`  
5. Studio: DEM checkbox enabled for Lantmäteriet but **default off**

## Non-goals

- Year-aware historical DEM (Markhöjdmodell tiles are current composites)  
- Swedish DSM/NMPT equivalent  
- Live download verification without Geotorget Markhöjdmodell credentials  

## Licensing

CC BY 4.0 / © Lantmäteriet — see `docs/DATA_LICENSING.md`.
