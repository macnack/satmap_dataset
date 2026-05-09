from __future__ import annotations

from urllib.parse import urlencode


DEFAULT_WCS_URL = (
    "https://avoin-karttakuva.maanmittauslaitos.fi/ortokuvat-ja-korkeusmallit/wcs/v2"
)
DEFAULT_COVERAGE_ID = "ortokuva_vari"
EPSG_3067_URI = "http://www.opengis.net/def/crs/EPSG/0/3067"


def build_describe_coverage_url(
    base_url: str,
    *,
    coverage_id: str = DEFAULT_COVERAGE_ID,
) -> str:
    params = [
        ("service", "WCS"),
        ("version", "2.0.1"),
        ("request", "DescribeCoverage"),
        ("coverageID", coverage_id),
    ]
    return f"{base_url}?{urlencode(params)}"


def build_get_coverage_url(
    base_url: str,
    *,
    coverage_id: str,
    bbox: tuple[float, float, float, float],
    year: int,
    output_format: str = "image/tiff",
    tile_size: int = 256,
) -> str:
    xmin, ymin, xmax, ymax = bbox
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("bbox must satisfy xmin<xmax and ymin<ymax")
    params: list[tuple[str, str]] = [
        ("service", "WCS"),
        ("version", "2.0.1"),
        ("request", "GetCoverage"),
        ("CoverageID", coverage_id),
        ("SUBSET", f"E({_fmt(xmin)},{_fmt(xmax)})"),
        ("SUBSET", f"N({_fmt(ymin)},{_fmt(ymax)})"),
        ("SUBSET", f'time("{int(year)}-12-31T00:00:00.000Z")'),
        ("SubsettingCRS", EPSG_3067_URI),
        ("OutputCRS", EPSG_3067_URI),
        ("format", output_format),
        ("geotiff:compression", "LZW"),
        ("geotiff:tiling", "true"),
        ("geotiff:tilewidth", str(int(tile_size))),
        ("geotiff:tileheight", str(int(tile_size))),
    ]
    return f"{base_url}?{urlencode(params)}"


def _fmt(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")
