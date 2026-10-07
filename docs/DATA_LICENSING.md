# Data and third-party terms

This repository’s **code** is MIT-licensed (see [`LICENSE`](../LICENSE)).
Downloaded imagery and map products remain under their providers’ terms. You
are responsible for compliance when you run the pipeline.

| Source | Typical use in this project | Notes |
|--------|----------------------------|-------|
| Polish Geoportal (PZGiK / GUGiK) | Orthophoto WFS/WMS, DEM WCS/skorowidz | Public geoportal services; respect rate limits (built-in jitter). Check current GUGiK reuse terms for your use case. |
| Lantmäteriet (Sweden) | STAC orthophoto download | Attribution required (`© Lantmäteriet`). Geotorget subscription may be required even when fees are 0 SEK. Annual WMS view service is a separate paid product. |
| NLS / Maanmittauslaitos (Finland) | Orthophoto WCS | Often CC BY 4.0; requires a free API key. Preserve attribution. |
| Copernicus Sentinel-2 | L2A visual COGs via Earth Search | [Copernicus open data terms](https://sentinels.copernicus.eu/web/sentinel/terms-conditions). Cite: *Contains modified Copernicus Sentinel data [year]*. |
| LROC NAC / PDS ODE | Lunar frames | NASA PDS data policies; projection tools (ISIS) are separate. |
| LandsD Imagery Map API (HK) | Current-mosaic XYZ tiles (`landsd_hk`, experimental) | HKSAR Government copyright; LandsD Map API Terms + IP Rights Notice. Attribution (Lands Department logo + copyright) required when displaying maps. Do not hammer the tile API. Separate open GeoTIFF orthophotos (DOP5000/TDOP) on DATA.GOV.HK have their own terms. |
| OpenStreetMap / Overpass | Label rasters | [ODbL](https://www.openstreetmap.org/copyright); attribute OSM contributors. |

Do not scrape provider endpoints aggressively. Defaults sleep between requests
for Geoportal; keep concurrency modest.

## Lands Department (Hong Kong) Imagery Map API

**Status: experimental.** The `landsd_hk` provider stitches public LandsD Imagery
XYZ tiles for evaluation AOIs. Imagery remains under HKSAR Government copyright.

- [Imagery Map API docs](https://portal.csdi.gov.hk/csdi-webpage/apidoc/ImageryMapAPI)
- Follow the LandsD Map API Terms of Use and Intellectual Property Rights Notice
  (attribution / logo requirements when displaying maps).
- Built-in request sleep and concurrency caps apply; do not raise them aggressively.
- The XYZ endpoint is a **current mosaic** (not a year archive). For multi-year
  ~30 cm stacks, look at licensed Esri Wayback export or DATA.GOV.HK open
  orthophoto GeoTIFFs — see [docs/providers/landsd_hk.md](providers/landsd_hk.md).
