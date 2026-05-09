"""Annual WMS fallback for Lantmäteriet years missing from STAC.

Builds a single GetMap URL per requested year. Geo-tagging the response is
left to the existing `pipeline.downloader` helpers — we only return the URL.
"""

from __future__ import annotations

from urllib.parse import urlencode


def build_get_map_url(
    base_url: str,
    *,
    layer: str,
    bbox: tuple[float, float, float, float],
    srs: str,
    width: int,
    height: int,
    year: int,
    version: str = "1.3.0",
    image_format: str = "image/tiff",
) -> str:
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
        "TIME": f"{year}-01-01/{year}-12-31",
    }
    return f"{base_url}?{urlencode(params)}"
