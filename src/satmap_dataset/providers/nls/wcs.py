from __future__ import annotations

import re
import xml.etree.ElementTree as ET
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


_YEAR_PATTERN = re.compile(r"^(\d{4})-\d{2}-\d{2}T")


def parse_describe_coverage_years(xml_bytes: bytes) -> list[int]:
    """Extract the years advertised by the WCS coverage's time axis.

    NLS Finland's GeoServer-backed WCS exposes the temporal axis as a
    sequence of `<gml:TimeInstant><gml:timePosition>YYYY-MM-DDT...` entries
    nested under `<wstxns1:TimeDomain>` inside `<gmlcov:metadata>`. We scan
    every `timePosition` element regardless of namespace prefix.
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return []
    years: set[int] = set()
    for element in root.iter():
        local = element.tag.split("}", 1)[-1]
        if local != "timePosition":
            continue
        text = (element.text or "").strip()
        match = _YEAR_PATTERN.match(text)
        if match:
            years.add(int(match.group(1)))
    return sorted(years)
