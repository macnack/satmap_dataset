"""WMS GetMap URL builder for swisstopo SWISSIMAGE Zeitreise."""

from __future__ import annotations

from urllib.parse import urlencode

DEFAULT_WMS_URL = "https://wms.geo.admin.ch/"
DEFAULT_WMS_LAYER = "ch.swisstopo.swissimage-product"
DEFAULT_WMS_VERSION = "1.3.0"
DEFAULT_IMAGE_FORMAT = "image/tiff"


def build_get_map_url(
    base_url: str,
    *,
    layer: str,
    bbox: tuple[float, float, float, float],
    srs: str,
    width: int,
    height: int,
    year: int,
    version: str = DEFAULT_WMS_VERSION,
    image_format: str = DEFAULT_IMAGE_FORMAT,
) -> str:
    """Build a Zeitreise GetMap URL with ``TIME=<year>``.

    For projected CRS (EPSG:2056) WMS 1.3.0 keeps axis order easting,northing.
    """
    crs_param = "CRS" if version.startswith("1.3") else "SRS"
    minx, miny, maxx, maxy = bbox
    if version.startswith("1.3") and srs.upper() == "EPSG:4326":
        bbox_str = f"{miny:.6f},{minx:.6f},{maxy:.6f},{maxx:.6f}"
    else:
        bbox_str = f"{minx:.6f},{miny:.6f},{maxx:.6f},{maxy:.6f}"
    params = {
        "SERVICE": "WMS",
        "REQUEST": "GetMap",
        "VERSION": version,
        "LAYERS": layer,
        "STYLES": "",
        "FORMAT": image_format,
        crs_param: srs,
        "BBOX": bbox_str,
        "WIDTH": str(int(width)),
        "HEIGHT": str(int(height)),
        "TIME": str(int(year)),
    }
    sep = "&" if "?" in base_url.rstrip("?") else "?"
    return f"{base_url.rstrip('?')}{sep}{urlencode(params)}"
