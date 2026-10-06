"""Esri World Imagery Wayback release catalog.

Each Wayback release is one WMTS layer in the public GetCapabilities document
(``WB_<YYYY>_R<NN>``, titled ``World Imagery (Wayback YYYY-MM-DD)``) whose
``ResourceURL`` embeds the release number ``M`` used by tile and tilemap URLs.
Release numbers are **not** chronological, so ordering always uses the release
date. The companion ``waybackconfig.json`` maps each release number to its
per-release metadata MapServer (capture date / source / resolution).
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Iterable

DEFAULT_CAPABILITIES_URL = (
    "https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/"
    "WMTSCapabilities.xml"
)
DEFAULT_CONFIG_URL = (
    "https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json"
)
DEFAULT_TILE_MATRIX_SET = "default028mm"
DEFAULT_TILE_URL_TEMPLATE = (
    "https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/"
    "default028mm/MapServer/tile/{release}/{z}/{y}/{x}"
)
DEFAULT_TILEMAP_URL_TEMPLATE = (
    "https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/"
    "tilemap/{release}/{z}/{y}/{x}"
)
METADATA_URL_TEMPLATE = (
    "https://metadata.maptiles.arcgis.com/arcgis/rest/services/"
    "World_Imagery_Metadata_{year}_r{rev}/MapServer"
)

_TITLE_DATE_RE = re.compile(r"Wayback\s+(\d{4}-\d{2}-\d{2})")
_RELEASE_NUM_RE = re.compile(r"/tile/(\d+)/", re.IGNORECASE)
_IDENTIFIER_RE = re.compile(r"^WB_(\d{4})_R(\d+)$", re.IGNORECASE)


class CatalogParseError(ValueError):
    pass


@dataclass(frozen=True)
class WaybackRelease:
    release_num: int
    identifier: str
    release_date: str  # ISO YYYY-MM-DD
    title: str = ""
    tile_url_template: str = DEFAULT_TILE_URL_TEMPLATE
    metadata_layer_url: str | None = None

    @property
    def sort_key(self) -> tuple[str, int]:
        return (self.release_date, self.release_num)

    def tile_url(self, z: int, x: int, y: int) -> str:
        return self.tile_url_template.format(release=self.release_num, z=z, y=y, x=x)


@dataclass
class ReleaseFilter:
    release_date_start: str | None = None
    release_date_end: str | None = None
    include_release_nums: set[int] = field(default_factory=set)
    exclude_release_nums: set[int] = field(default_factory=set)

    @classmethod
    def from_options(cls, options: dict[str, Any]) -> "ReleaseFilter":
        def _ints(key: str) -> set[int]:
            raw = options.get(key) or []
            if isinstance(raw, (int, str)):
                raw = [raw]
            return {int(v) for v in raw}

        def _date(key: str) -> str | None:
            value = options.get(key)
            if value in (None, ""):
                return None
            text = str(value)
            if not re.fullmatch(r"\d{4}(-\d{2}(-\d{2})?)?", text):
                raise ValueError(f"{key} must be YYYY, YYYY-MM or YYYY-MM-DD; got {value!r}")
            return text

        return cls(
            release_date_start=_date("release_date_start"),
            release_date_end=_date("release_date_end"),
            include_release_nums=_ints("release_numbers"),
            exclude_release_nums=_ints("exclude_release_numbers"),
        )

    def accepts(self, release: WaybackRelease) -> bool:
        if self.include_release_nums and release.release_num not in self.include_release_nums:
            return False
        if release.release_num in self.exclude_release_nums:
            return False
        if self.release_date_start and release.release_date < self.release_date_start:
            return False
        if self.release_date_end:
            # Prefix compare so "2020" / "2020-06" bound the whole period inclusively.
            if release.release_date[: len(self.release_date_end)] > self.release_date_end:
                return False
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "release_date_start": self.release_date_start,
            "release_date_end": self.release_date_end,
            "release_numbers": sorted(self.include_release_nums),
            "exclude_release_numbers": sorted(self.exclude_release_nums),
        }


def derive_metadata_layer_url(identifier: str) -> str | None:
    match = _IDENTIFIER_RE.match(identifier.strip())
    if match is None:
        return None
    return METADATA_URL_TEMPLATE.format(year=match.group(1), rev=match.group(2).zfill(2))


def _local(tag: str) -> str:
    # Esri publishes non-standard ``https://www.opengis.net/...`` namespace
    # URIs, so match on local names rather than fixed namespaces.
    return tag.rsplit("}", 1)[-1]


def _normalise_template(template: str, tile_matrix_set: str) -> str:
    return (
        template.replace("{TileMatrixSet}", tile_matrix_set)
        .replace("{TileMatrix}", "{z}")
        .replace("{TileRow}", "{y}")
        .replace("{TileCol}", "{x}")
    )


def parse_capabilities(
    capabilities_xml: str | bytes,
    *,
    tile_matrix_set: str = DEFAULT_TILE_MATRIX_SET,
) -> list[WaybackRelease]:
    """Return every Wayback release advertised in WMTS GetCapabilities, newest first."""
    try:
        root = ET.fromstring(
            capabilities_xml if isinstance(capabilities_xml, (bytes, bytearray)) else capabilities_xml.encode("utf-8")
        )
    except ET.ParseError as exc:
        raise CatalogParseError(f"Wayback GetCapabilities is not valid XML: {exc}") from exc

    releases: dict[int, WaybackRelease] = {}
    for layer in (el for el in root.iter() if _local(el.tag) == "Layer"):
        children = {_local(child.tag): child for child in layer}
        title = (children["Title"].text or "").strip() if "Title" in children else ""
        identifier = (children["Identifier"].text or "").strip() if "Identifier" in children else ""
        resource = children.get("ResourceURL")
        template = resource.get("template", "") if resource is not None else ""
        num_match = _RELEASE_NUM_RE.search(template)
        date_match = _TITLE_DATE_RE.search(title)
        if num_match is None or date_match is None or not identifier:
            continue
        release_num = int(num_match.group(1))
        releases[release_num] = WaybackRelease(
            release_num=release_num,
            identifier=identifier,
            release_date=date_match.group(1),
            title=title,
            tile_url_template=_normalise_template(template, tile_matrix_set),
            metadata_layer_url=derive_metadata_layer_url(identifier),
        )
    if not releases:
        raise CatalogParseError("No Wayback release layers found in GetCapabilities")
    return sort_newest_first(releases.values())


def apply_config(
    releases: Iterable[WaybackRelease], config_json: str | bytes | dict[str, Any]
) -> list[WaybackRelease]:
    """Overlay ``metadataLayerUrl`` from ``waybackconfig.json`` onto parsed releases."""
    payload = json.loads(config_json) if isinstance(config_json, (str, bytes, bytearray)) else config_json
    out: list[WaybackRelease] = []
    for release in releases:
        entry = payload.get(str(release.release_num)) if isinstance(payload, dict) else None
        url = entry.get("metadataLayerUrl") if isinstance(entry, dict) else None
        if url:
            release = WaybackRelease(
                release_num=release.release_num,
                identifier=release.identifier,
                release_date=release.release_date,
                title=release.title,
                tile_url_template=release.tile_url_template,
                metadata_layer_url=str(url).rstrip("/"),
            )
        out.append(release)
    return sort_newest_first(out)


def sort_newest_first(releases: Iterable[WaybackRelease]) -> list[WaybackRelease]:
    return sorted(releases, key=lambda r: r.sort_key, reverse=True)
