# SIVL benchmark areas

Fifteen orthophoto AOIs from the SIVL benchmark, recorded here as structured
configs. Each `areaN.json` carries the centre lat/lon, the rectangle dimensions
(`width_meters` × `height_meters` = 4800 × 2987 m, ≈ 14.3 km²), the expected
ground sample distance, the years that ship with SIVL, and the inferred
country.

## Coverage

| Area | Center (lat, lon) | Country | Years |
|------|-------------------|---------|-------|
| area1  | 60.16680, 24.33430 | Finland | 2011, 2015, 2018 |
| area2  | 60.20607, 24.37938 | Finland | 2011, 2013, 2014, 2015, 2017, 2018 |
| area3  | 60.25166, 25.04226 | Finland | 2006, 2008, 2011, 2012, 2013, 2014, 2015, 2017, 2018, 2020 |
| area4  | 60.21333, 24.88401 | Finland | 2013, 2014, 2015, 2017, 2018, 2020 |
| area5  | 60.56371, 24.64580 | Finland | 2011, 2013, 2014, 2015, 2017, 2018, 2020 |
| area6  | 60.62518, 24.70387 | Finland | 2002, 2011, 2013, 2014, 2015, 2017, 2018, 2020 |
| area7  | 60.59209, 24.71266 | Finland | 2002, 2011, 2013, 2014, 2015, 2017, 2018, 2020 |
| area8  | 59.64694, 18.20729 | **Sweden** | 2008, 2010, 2011, 2013, 2014, 2017, 2019, 2020 |
| area9  | 60.51560, 24.34963 | Finland | 2011, 2017, 2019, 2021 |
| area10 | 61.49930, 23.84915 | Finland | 2012, 2014, 2020, 2021 |
| area11 | 61.45997, 24.06081 | Finland | 2005, 2013, 2014, 2019, 2020, 2021 |
| area12 | 60.24405, 24.94654 | Finland | 2013, 2014, 2017, 2018, 2020 |
| area13 | 60.22567, 25.04692 | Finland | 2011, 2012, 2013, 2020 |
| area14 | 60.19489, 24.94013 | Finland | 2011, 2012, 2013, 2015 |
| area15 | 60.44960, 22.26436 | Finland | 2015, 2016, 2018, 2020 |

## What works today and what doesn't

- **area8 (Sweden)** can be fetched through the existing `lantmateriet`
  provider once we add a runnable location override; coordinates fall well
  within Lantmäteriet's STAC catalogue. Note that GSD here will be 0.16 m or
  0.25 m depending on the i-zone, **finer** than the ~1 m of SIVL — set
  `target_srs: EPSG:3006` and downsample at render to match.
- **All other areas (Finland)** are outside Lantmäteriet. To download these
  we need a Finland-specific provider that talks to **Maanmittauslaitos
  (NLS Finland)** orthophoto services — that work isn't done yet. Tracking
  notes:
  - NLS Finland publishes orthophotos via WMTS / WMS through their
    "Avoindata" (open data) portal, with credentials issued at
    https://omatili.maanmittauslaitos.fi/.
    Default CRS is **EPSG:3067** (ETRS89-TM35FIN), not EPSG:3006.
  - A future `nls_finland` provider would mirror the `lantmateriet` shape:
    `providers/nls_finland/{provider,wmts,year_policy}.py`, with provider
    options for service URL, layer name, and an API key/Bearer token.
  - The `expected_years` array in each area JSON is the SIVL-listed years.
    The provider's job is to map those onto whatever NLS Finland actually
    has in its catalogue per AOI.

## Rectangular AOIs

Existing `_resolve_json_center_bbox` in `cli.py` only resolves *square*
bboxes via `area_km2` / `square_km`. The SIVL areas are **4800 m × 2987 m**
rectangles. When we wire these up, the bbox builder needs a
`width_meters`/`height_meters` path (those fields are already on the
configs). Pre-computing the bbox per area and pinning it as a literal
`bbox` string is the quick alternative if you want to test one area before
that work lands.

## Attribution

When eventually publishing results derived from these areas:

- For area8 (Sweden / Lantmäteriet): `© Lantmäteriet, CC BY 4.0`.
- For Finland areas (NLS Finland / Maanmittauslaitos): `© Maanmittauslaitos`
  with the licence string from their open-data terms (currently CC BY 4.0).
