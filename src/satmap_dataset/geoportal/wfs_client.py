from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from satmap_dataset.geoportal.http import RetryPolicy, request_with_retry
from satmap_dataset.models import YearStatus

DEFAULT_WFS_URL = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WFS/Skorowidze"


def _local_name(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def _iter_featuretype_names(root: ET.Element) -> list[str]:
    names: list[str] = []
    for node in root.iter():
        if _local_name(node.tag) != "FeatureType":
            continue
        for child in node:
            if _local_name(child.tag) == "Name" and child.text:
                names.append(child.text.strip())
                break
    return names


def _extract_year_typenames(cap_root: ET.Element) -> dict[int, str]:
    year_to_typename: dict[int, str] = {}
    pattern = re.compile(r"SkorowidzOrtof\w*?(\d{4})$", re.IGNORECASE)
    for name in _iter_featuretype_names(cap_root):
        match = pattern.search(name)
        if not match:
            continue
        year = int(match.group(1))
        year_to_typename.setdefault(year, name)
    return year_to_typename


def _parse_feature_count(root: ET.Element) -> int:
    # WFS 2.0
    for attr_name in ("numberMatched", "numberOfFeatures"):
        value = root.attrib.get(attr_name)
        if value is None:
            continue
        stripped = value.strip().lower()
        if stripped in {"unknown", ""}:
            continue
        try:
            return max(0, int(float(stripped)))
        except ValueError:
            continue

    # Fallback: count feature members.
    count = 0
    for node in root.iter():
        local = _local_name(node.tag)
        if local in {"member", "featureMember"}:
            count += 1
    return count


def _iter_features(root: ET.Element) -> list[ET.Element]:
    features: list[ET.Element] = []
    for node in root.iter():
        local = _local_name(node.tag)
        if local in {"member", "featureMember"}:
            for child in node:
                features.append(child)
    return features


def _find_attr_value(feature: ET.Element, attr_name: str) -> str | None:
    target = attr_name.lower()
    for node in feature.iter():
        if _local_name(node.tag).lower() != target:
            continue
        if node.text and node.text.strip():
            return node.text.strip()
    return None


def _tile_id_from_url(url: str) -> str:
    name = Path(url).name
    stem = name.rsplit(".", 1)[0]
    parts = stem.split("_")
    if len(parts) >= 3:
        return "_".join(parts[2:])
    return stem


async def get_capabilities(
    base_url: str = DEFAULT_WFS_URL,
    *,
    timeout: float = 20.0,
    retry_policy: RetryPolicy | None = None,
) -> tuple[ET.Element, dict[int, str]]:
    response = await request_with_retry(
        "GET",
        base_url,
        params={"service": "WFS", "request": "GetCapabilities", "version": "2.0.0"},
        timeout=timeout,
        retry_policy=retry_policy,
    )
    root = ET.fromstring(response.text)
    return root, _extract_year_typenames(root)


async def get_feature_count(
    typename: str,
    bbox: str,
    srs: str,
    *,
    base_url: str = DEFAULT_WFS_URL,
    timeout: float = 20.0,
    retry_policy: RetryPolicy | None = None,
) -> int:
    response = await request_with_retry(
        "GET",
        base_url,
        params={
            "SERVICE": "WFS",
            "REQUEST": "GetFeature",
            "VERSION": "2.0.0",
            "TYPENAMES": typename,
            "BBOX": f"{bbox},{srs}",
            "COUNT": "10000",
        },
        timeout=timeout,
        retry_policy=retry_policy,
    )
    root = ET.fromstring(response.text)
    return _parse_feature_count(root)


async def get_year_tiles(
    year: int,
    bbox: str,
    srs: str,
    *,
    base_url: str = DEFAULT_WFS_URL,
    timeout: float = 20.0,
    retry_policy: RetryPolicy | None = None,
    year_to_typename: dict[int, str] | None = None,
) -> tuple[YearStatus, dict[str, str]]:
    mapping = year_to_typename
    if mapping is None:
        _, mapping = await get_capabilities(
            base_url=base_url,
            timeout=timeout,
            retry_policy=retry_policy,
        )

    typename = mapping.get(year)
    if not typename:
        return (
            YearStatus(
                year=year,
                typename_exists=False,
                feature_count=0,
                status="no_typename",
                reason=f"No WFS typename found for year {year}.",
            ),
            {},
        )

    response = await request_with_retry(
        "GET",
        base_url,
        params={
            "SERVICE": "WFS",
            "REQUEST": "GetFeature",
            "VERSION": "2.0.0",
            "TYPENAMES": typename,
            "BBOX": f"{bbox},{srs}",
            "COUNT": "10000",
        },
        timeout=timeout,
        retry_policy=retry_policy,
    )
    root = ET.fromstring(response.text)
    feature_count = _parse_feature_count(root)

    tiles: dict[str, str] = {}
    for feature in _iter_features(root):
        url_value = _find_attr_value(feature, "url_do_pobrania")
        if not url_value:
            continue
        tile_id = _tile_id_from_url(url_value)
        tiles.setdefault(tile_id, url_value)

    if feature_count <= 0:
        return (
            YearStatus(
                year=year,
                typename_exists=True,
                feature_count=0,
                status="zero_features",
                reason="Typename exists but no features found for AOI.",
            ),
            {},
        )

    if not tiles:
        return (
            YearStatus(
                year=year,
                typename_exists=True,
                feature_count=feature_count,
                status="zero_features",
                reason="Features found but url_do_pobrania is missing in AOI response.",
            ),
            {},
        )

    return (
        YearStatus(
            year=year,
            typename_exists=True,
            feature_count=feature_count,
            status="has_features",
            reason=None,
        ),
        tiles,
    )


async def probe_year(
    year: int,
    bbox: str,
    srs: str,
    *,
    base_url: str = DEFAULT_WFS_URL,
    timeout: float = 20.0,
    retry_policy: RetryPolicy | None = None,
    year_to_typename: dict[int, str] | None = None,
) -> YearStatus:
    status, _ = await get_year_tiles(
        year=year,
        bbox=bbox,
        srs=srs,
        base_url=base_url,
        timeout=timeout,
        retry_policy=retry_policy,
        year_to_typename=year_to_typename,
    )
    return status
