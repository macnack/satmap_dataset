"""Parse available Zeitreise years from swisstopo WMTS GetCapabilities."""

from __future__ import annotations

import re
from typing import Iterable

DEFAULT_WMTS_CAPABILITIES_URL = "https://wmts.geo.admin.ch/1.0.0/WMTSCapabilities.xml"
DEFAULT_LAYER_ID = "ch.swisstopo.swissimage-product"

_YEAR_RE = re.compile(r"^\d{4}$")


class CapabilitiesParseError(ValueError):
    pass


def parse_time_years(
    capabilities_xml: str | bytes,
    *,
    layer_id: str = DEFAULT_LAYER_ID,
) -> list[int]:
    """Return sorted YYYY years advertised for ``layer_id``'s Time dimension.

    Ignores non-year tokens such as ``current``. Raises if the layer or its
    Time dimension cannot be found.
    """
    text = (
        capabilities_xml.decode("utf-8", errors="replace")
        if isinstance(capabilities_xml, (bytes, bytearray))
        else capabilities_xml
    )
    escaped = re.escape(layer_id)
    match = re.search(
        rf"<ows:Identifier>{escaped}</ows:Identifier>(.*?)</Layer>",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        match = re.search(
            rf"<Identifier>{escaped}</Identifier>(.*?)</Layer>",
            text,
            flags=re.DOTALL,
        )
    if match is None:
        raise CapabilitiesParseError(f"WMTS layer {layer_id!r} not found in GetCapabilities")

    block = match.group(1)
    dim = re.search(r"<Dimension>(.*?)</Dimension>", block, flags=re.DOTALL)
    if dim is None:
        raise CapabilitiesParseError(f"No Dimension block for layer {layer_id!r}")

    dim_body = dim.group(1)
    ident = re.search(
        r"<ows:Identifier>\s*([^<\s]+)\s*</ows:Identifier>",
        dim_body,
    ) or re.search(r"<Identifier>\s*([^<\s]+)\s*</Identifier>", dim_body)
    if ident is not None and ident.group(1).strip().lower() != "time":
        raise CapabilitiesParseError(
            f"Expected Time dimension for {layer_id!r}, got {ident.group(1)!r}"
        )

    values = re.findall(r"<Value>\s*([^<\s]+)\s*</Value>", dim_body)
    years = sorted({int(v) for v in values if _YEAR_RE.fullmatch(v)})
    if not years:
        raise CapabilitiesParseError(f"No YYYY Time values for layer {layer_id!r}")
    return years


def expand_wms_time_dimension(raw: str) -> list[int]:
    """Expand a WMS 1.3 Dimension/time string into concrete years.

    Supports comma lists and ISO8601 intervals like ``1949/2025/P1Y``.
    """
    years: set[int] = set()
    for token in (raw or "").split(","):
        part = token.strip()
        if not part or part.lower() == "current" or part == "9999":
            continue
        if _YEAR_RE.fullmatch(part):
            years.add(int(part))
            continue
        if "/" in part:
            bits = part.split("/")
            if len(bits) >= 2 and _YEAR_RE.fullmatch(bits[0]) and _YEAR_RE.fullmatch(bits[1]):
                start, end = int(bits[0]), int(bits[1])
                step = 1
                if len(bits) >= 3 and bits[2].upper().startswith("P") and bits[2].upper().endswith("Y"):
                    try:
                        step = max(1, int(bits[2][1:-1] or "1"))
                    except ValueError:
                        step = 1
                years.update(range(start, end + 1, step))
    return sorted(years)


def intersect_years(requested: Iterable[int], available: Iterable[int]) -> list[int]:
    available_set = set(available)
    return sorted(year for year in requested if year in available_set)
