"""Tooltip descriptions for satmap-studio form fields."""

PROVIDER = (
    "Orthophoto source. Sets the base config JSON, native CRS (EPSG:2180 Poland, "
    "EPSG:3006 Sweden, EPSG:3067 Finland, or UTM for Sentinel-2), and whether "
    "credentials are required."
)
LOCATION_NAME = (
    "Human-readable label used to derive output folder names "
    "(downloads_<slug>, rendered_<slug>, artifacts_<slug>) and saved location JSON."
)
CENTER_LAT = (
    "AOI center latitude in WGS84 (EPSG:4326). Click the map or use Nominatim search "
    "to update. Combined with longitude and area to build the projected bbox."
)
CENTER_LON = (
    "AOI center longitude in WGS84 (EPSG:4326). East positive. Used with latitude "
    "and area_km² to compute xmin,ymin,xmax,ymax in the provider CRS."
)
NLS_API_KEY = (
    "Free API key from omatili.maanmittauslaitos.fi. Alternatively set "
    "SATMAP_NLS_API_KEY or a single-line .secret file at the repo root."
)
WRITE_SECRET = (
    "Save the NLS API key to .secret in the repo root before index/download. "
    "Do not commit this file to git."
)
LANT_API_KEY = (
    "Optional Bearer token for Lantmäteriet STAC/WMS. Env: SATMAP_LANTMATERIET_API_KEY."
)
LANT_USERNAME = (
    "Geotorget API username (lm_… style) for dl1.lantmateriet.se downloads. "
    "Env: SATMAP_LANTMATERIET_USERNAME."
)
LANT_PASSWORD = (
    "Geotorget generated password for the subscribed product. "
    "Env: SATMAP_LANTMATERIET_PASSWORD."
)
AREA_KM2 = (
    "Square AOI area in km². Side length = √(area_km2) × 1000 m "
    "(e.g. 4 km² → 2 km × 2 km). Larger areas increase download time and disk use."
)
SEARCH_QUERY = (
    "Place name or address looked up via OpenStreetMap Nominatim (online). "
    "Pick a result to move the AOI center."
)
YEAR_START = "First calendar year included in the index and download (inclusive)."
YEAR_END = "Last calendar year included in the index and download (inclusive)."
PX_PER_METER = (
    "Output raster resolution: pixels per meter on the render grid. "
    "15 px/m on a 2 km side → 30 000×30 000 RGB_U8 tiles. "
    "Source GSD from WFS/STAC is shown separately after index."
)
PROFILE = (
    "train: consistent NN-ready grid. reference: geometry-driven sizing, "
    "WMS fallback options, extra QC fields in the render manifest."
)
RUN_DEM = (
    "Download ISOK NMT/NMPT elevation and align to the RGB ReferenceGrid "
    "(same bbox, width, height as render output)."
)
RUN_OSM = (
    "Fetch OSM semantic labels (buildings, roads, paths, green, water) via Overpass "
    "and align to the RGB grid using per-year acquisition dates."
)
VALIDATE = (
    "After RGB render, check asset existence, pixel profile, EPSG, georef, and "
    "sidecars; writes validation_report.json."
)
MODE = (
    "hybrid: WFS/STAC first, WMS for missing years. wfs_render: WFS only. "
    "wms_tiled: WMS only. stac: STAC-only providers (Lantmäteriet, Sentinel-2)."
)
WMS_FALLBACK = (
    "When a requested year has no WFS/STAC tiles, fetch that year via WMS "
    "StandardResolutionTime (Geoportal) or configured WMS (Lantmäteriet)."
)
MIN_YEARS = (
    "Minimum count of years with data required for index/run to pass policy checks. "
    "Run still exits 1 if fewer years are available."
)
STRICT_YEARS = (
    "Require every year in the range to be available. Fails if any year is missing."
)
CONCURRENCY = (
    "Parallel download workers. Lower values are safer for Geoportal rate limits."
)
SLEEP_MIN = (
    "Minimum random delay (seconds) before each HTTP request during download. "
    "Jitter reduces synchronized bursts against rate-limited services."
)
SLEEP_MAX = (
    "Maximum random delay (seconds) before each HTTP request during download."
)
RAW_EXPORT = (
    "After a successful RGB download, export native tiles into the sat_roma raw layout "
    "under raw_root/<provider>/<area>/ and write raw_export_manifest.json."
)
CELL_MODE = (
    "footprint: one tile origin per cell (default). world_window: snap mixed-GSD areas "
    "to a coarse grid and optionally resample to equal dimensions."
)
EQUALIZE_GSD = (
    "world_window only: resample all years to the coarsest native GSD so stacks are "
    "co-registered and equal-dimension. Off keeps native GSD per year."
)
RAW_ROOT = (
    "Root for raw tile export. Default: SATMAP_RAW_ROOT env or ~/Github/sat_data_raw."
)
CHECK_INDEX = (
    "Probe WFS/STAC for each requested year without downloading. Writes index_manifest.json "
    "and year_availability_report.json with per-year GSD (finest/coarsest) and feature counts."
)
CHECK_DEM = (
    "Query ISOK NMT/NMPT skorowidz coverage for the AOI (no download). "
    "Shows which elevation products and years have tiles overlapping the bbox."
)
RUN_FULL_STACK = (
    "Run RGB index→download→render, then DEM and OSM aligned to the same grid, "
    "optional validation, and optional raw-export if enabled in Run settings."
)
SAVE_LOCATION_JSON = (
    "Write configs/run/locations/<slug>.json and configs/run/generated/<slug>.run.json "
    "so you can replay the same run via just run-location-json or run-json."
)
