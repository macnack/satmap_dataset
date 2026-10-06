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
| Esri World Imagery Wayback | Archived World Imagery basemap tiles (experimental `esri_wayback` provider) | **Restrictive.** Esri Master Agreement: no scraping/downloading/storing outside Esri Content Packages, no AI/ML training outside Esri software. See [below](#esri-world-imagery-wayback). |
| LROC NAC / PDS ODE | Lunar frames | NASA PDS data policies; projection tools (ISIS) are separate. |
| OpenStreetMap / Overpass | Label rasters | [ODbL](https://www.openstreetmap.org/copyright); attribute OSM contributors. |

Do not scrape provider endpoints aggressively. Defaults sleep between requests
for Geoportal; keep concurrency modest.

## Esri World Imagery Wayback

**Status: experimental. As far as we can tell, Esri's published terms do not
allow what this provider does (bulk tile download, offline storage outside
Esri software, ML training) unless Esri gives you separate permission. Do not
use it for a dataset you train on, publish or redistribute until you have that
permission in writing.** We implemented it so the pipeline can be evaluated
technically and used under a licence negotiated with Esri. Shipping the code
does not grant any rights to the data.

Every Wayback layer (for example
[World Imagery (Wayback 2014-02-20)](https://www.arcgis.com/home/item.html?id=903f0abe9c3b452dafe1ca5b8dd858b9))
and the live [World Imagery](https://www.arcgis.com/home/item.html?id=10df2279f9684e4a9f6a7f08febac2a9)
item say *"This work is licensed under the Esri Master License Agreement"* and
*"This layer is not intended to be used to export tiles for offline."*
These are the current terms as checked on 2026-10-06:

- **[Esri Master Agreement (E204, revised 2025-08-01)](https://www.esri.com/en-us/legal/terms/master-agreement)**
  ([PDF](https://assets.esri.com/content/dam/esrisites/en-us/media/legal/ma-full/ma-full.pdf)):
  - §3.2(a): Data may only be used with the Esri Products it was provided for.
  - §3.2(b): static representations (PDF/JPEG/HTML, ArcGIS Web Maps, StoryMaps)
    are allowed **with attribution** to Esri and its licensors.
  - §3.2(c): *"Customer may take Online Services basemaps offline through Esri
    Content Packages … Customer may not otherwise scrape, download, or store Data."*
  - §3.3(h): *"Customer may not use Data outside of the Software and Online
    Services to teach or train machine systems, models, … including neural
    networks ('AI/ML Systems')."*
- **[ArcGIS Online terms of use FAQ / summary](https://www.esri.com/content/dam/arcgisonline/docs/tou_summary.pdf)**
  (last updated 2025-04-21). You **may not**:
  - *"Systematically harvest basemap tiles through any method other than using
    Esri Content Packages"*;
  - redistribute basemap tiles;
  - *"Download, redistribute or self-host any content hosted by Esri"*;
  - make commercial use of Living Atlas content without a licence from Esri.

  Use must be together with Esri software or an ArcGIS Online subscription.
- [Product-Specific Terms of Use (E300)](https://www.esri.com/en-us/legal/terms/product-specific-scope-of-use)
  and [third-party data terms](https://www.esri.com/legal/third-party-data)
  add to the above.

**Attribution** (required wherever imagery is shown): *Esri, Vantor, Earthstar
Geographics, and the GIS User Community*. Older releases list more sources
(CNES/Airbus DS, USDA FSA, USGS, Aerogrid, IGN, IGP), and the per-version
`source` / `provider_name` fields in the index manifest name the imagery
vendor. The provider writes the attribution and a licence notice into every
index/download manifest (`provider_metadata.attribution`,
`provider_metadata.license_notice`).

Note that Wayback often re-serves national orthophotos. For example, Warsaw's
2019, 2021 and 2023 versions are GUGiK *Poland Orthos*. Where that happens,
the original source (Geoportal, this repo's default provider) is usually the
better-licensed way to get the same pixels.
