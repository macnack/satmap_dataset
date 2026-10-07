"""LandsD Imagery Map API (XYZ PNG tiles).

API docs: https://portal.csdi.gov.hk/csdi-webpage/apidoc/ImageryMapAPI

URL template::

    https://mapapi.geodata.gov.hk/gs/api/v1.0.0/xyz/imagery/{sr}/{z}/{x}/{y}.png

``sr`` is ``WGS84`` (Web Mercator XYZ) or ``HK80``. This provider uses WGS84 so
the existing EPSG:3857 stitch helpers apply. Zoom 19 ≈ 0.28 m GSD at Hong Kong
latitudes; zoom 18 ≈ 0.55 m.
"""

from __future__ import annotations

DEFAULT_XYZ_BASE = "https://mapapi.geodata.gov.hk/gs/api/v1.0.0/xyz/imagery"
DEFAULT_SR = "WGS84"
DEFAULT_MIN_ZOOM = 15
DEFAULT_MAX_ZOOM = 19  # z20 often soft / overzoom
DEFAULT_USER_AGENT = (
    "satmap_dataset/0.1 (+https://github.com/macnack/satmap_dataset; "
    "research eval; provider=landsd_hk)"
)
ATTRIBUTION = (
    "Map from Lands Department; Aerial Photograph from Lands Department "
    "(© The Government of the Hong Kong SAR)"
)
LICENSE_NOTICE = (
    "LandsD Map API / imagery is protected by HKSAR Government copyright. "
    "Use is subject to the LandsD Map API Terms and IP Rights Notice; include "
    "the Lands Department logo and copyright notice when displaying maps. "
    "Open Digital Orthophoto GeoTIFF products (DOP5000 / TDOP / DOP5000-1982) "
    "are separately available under DATA.GOV.HK terms with attribution. "
    "Do not hammer the tile API (built-in sleep/concurrency apply)."
)


def tile_url(base: str, *, sr: str, z: int, x: int, y: int) -> str:
    return f"{base.rstrip('/')}/{sr}/{z}/{x}/{y}.png"
