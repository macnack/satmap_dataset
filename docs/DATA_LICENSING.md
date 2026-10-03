# Data and third-party terms

This repository’s **code** is MIT-licensed (see [`LICENSE`](../LICENSE)).
Downloaded imagery and map products remain under their providers’ terms. You
are responsible for compliance when you run the pipeline.

| Source | Typical use in this project | Notes |
|--------|----------------------------|-------|
| Polish Geoportal (PZGiK / GUGiK) | Orthophoto WFS/WMS, DEM WCS/skorowidz | Public geoportal services; respect rate limits (built-in jitter). Check current GUGiK reuse terms for your use case. |
| Lantmäteriet (Sweden) | STAC orthophoto (`stac-bild`); optional DEM Markhöjdmodell via STAC-höjd (`stac-hojd`, collection `dtm-cog`) | Attribution required (`© Lantmäteriet`). Orthophoto and Markhöjdmodell are **separate** Geotorget products (often separate credentials). Markhöjdmodell catalog is CC BY 4.0; asset download still needs an authorized Geotorget account. Annual WMS view service is a separate paid product. |
| NLS / Maanmittauslaitos (Finland) | Orthophoto WCS | Often CC BY 4.0; requires a free API key. Preserve attribution. |
| Copernicus Sentinel-2 | L2A visual COGs via Earth Search | [Copernicus open data terms](https://sentinels.copernicus.eu/web/sentinel/terms-conditions). Cite: *Contains modified Copernicus Sentinel data [year]*. |
| LROC NAC / PDS ODE | Lunar frames | NASA PDS data policies; projection tools (ISIS) are separate. |
| OpenStreetMap / Overpass | Label rasters | [ODbL](https://www.openstreetmap.org/copyright); attribute OSM contributors. |

Do not scrape provider endpoints aggressively. Defaults sleep between requests
for Geoportal; keep concurrency modest.
