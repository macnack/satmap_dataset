from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from satmap_dataset.geoportal.http import RetryPolicy, request_with_retry
from satmap_dataset.models import YearStatus

DEFAULT_WFS_URL = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WFS/Skorowidze"
DEFAULT_PAGE_SIZE = 10000


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


def _color_priority(color: str | None) -> int:
    if not color:
        return 0
    value = color.strip().upper()
    if value == "RGB":
        return 2
    if value == "CIR":
        return 1
    return 0


def _is_grid_compatible_with_srs(uklad_xy: str | None, srs: str) -> bool:
    if not srs:
        return True
    srs_upper = srs.strip().upper()
    if srs_upper == "EPSG:2180":
        if not uklad_xy:
            return False
        return "PL-1992" in uklad_xy.strip().upper()
    return True


def _parse_bbox_str(value: str) -> tuple[float, float, float, float]:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must have format xmin,ymin,xmax,ymax")
    min_x, min_y, max_x, max_y = parts
    if min_x >= max_x or min_y >= max_y:
        raise ValueError("bbox must satisfy xmin<xmax and ymin<ymax")
    return min_x, min_y, max_x, max_y


def _bbox_overlap_area(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    min_x = max(a[0], b[0])
    min_y = max(a[1], b[1])
    max_x = min(a[2], b[2])
    max_y = min(a[3], b[3])
    if min_x >= max_x or min_y >= max_y:
        return 0.0
    return (max_x - min_x) * (max_y - min_y)


def _swap_bbox_axes(
    bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    min_x, min_y, max_x, max_y = bbox
    return min_y, min_x, max_y, max_x


def _extract_feature_bbox(feature: ET.Element) -> tuple[float, float, float, float] | None:
    values: list[float] = []
    for node in feature.iter():
        local = _local_name(node.tag)
        if local not in {"posList", "pos", "coordinates", "lowerCorner", "upperCorner"}:
            continue
        if not node.text:
            continue
        numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", node.text)
        values.extend(float(number) for number in numbers)

    if len(values) < 4:
        return None
    xs = values[0::2]
    ys = values[1::2]
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


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
            "SRSNAME": srs,
            "BBOX": f"{bbox},{srs}",
            "COUNT": "1",
            "RESULTTYPE": "hits",
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
) -> tuple[YearStatus, dict[str, str], dict[str, list[float]]]:
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
            {},
        )

    tiles: dict[str, str] = {}
    tile_bboxes: dict[str, list[float]] = {}
    request_bbox = _parse_bbox_str(bbox)
    tile_overlap_by_id: dict[str, float] = {}
    tile_color_priority_by_id: dict[str, int] = {}
    incompatible_grid_count = 0
    missing_url_count = 0
    feature_count = 0
    start_index = 0
    page_size = DEFAULT_PAGE_SIZE
    pages_seen = 0
    max_pages = 1000
    last_page_signature: tuple[int, str | None, str | None] | None = None

    while True:
        response = await request_with_retry(
            "GET",
            base_url,
            params={
                "SERVICE": "WFS",
                "REQUEST": "GetFeature",
                "VERSION": "2.0.0",
                "TYPENAMES": typename,
                "SRSNAME": srs,
                "BBOX": f"{bbox},{srs}",
                "COUNT": str(page_size),
                "STARTINDEX": str(start_index),
            },
            timeout=timeout,
            retry_policy=retry_policy,
        )
        root = ET.fromstring(response.text)
        matched = _parse_feature_count(root)
        if pages_seen == 0:
            feature_count = matched

        features = _iter_features(root)
        returned = len(features)
        first_url = _find_attr_value(features[0], "url_do_pobrania") if features else None
        last_url = _find_attr_value(features[-1], "url_do_pobrania") if features else None
        page_signature = (returned, first_url, last_url)
        if page_signature == last_page_signature:
            break
        last_page_signature = page_signature

        for feature in features:
            url_value = _find_attr_value(feature, "url_do_pobrania")
            if not url_value:
                missing_url_count += 1
                continue
            uklad_xy = _find_attr_value(feature, "uklad_xy")
            if not _is_grid_compatible_with_srs(uklad_xy, srs):
                incompatible_grid_count += 1
                continue
            tile_id = _tile_id_from_url(url_value)
            color_priority = _color_priority(_find_attr_value(feature, "kolor"))
            feature_bbox = _extract_feature_bbox(feature)
            normalized_bbox = feature_bbox
            overlap = -1.0
            if feature_bbox is not None:
                normal_overlap = _bbox_overlap_area(feature_bbox, request_bbox)
                swapped_bbox = _swap_bbox_axes(feature_bbox)
                swapped_overlap = _bbox_overlap_area(swapped_bbox, request_bbox)
                if swapped_overlap > normal_overlap:
                    normalized_bbox = swapped_bbox
                    overlap = swapped_overlap
                else:
                    overlap = normal_overlap

            existing = tiles.get(tile_id)
            if existing is None:
                tiles[tile_id] = url_value
                if normalized_bbox is not None:
                    tile_bboxes[tile_id] = [float(v) for v in normalized_bbox]
                tile_overlap_by_id[tile_id] = overlap
                tile_color_priority_by_id[tile_id] = color_priority
                continue

            current_overlap = tile_overlap_by_id.get(tile_id, -1.0)
            current_color_priority = tile_color_priority_by_id.get(tile_id, 0)
            should_replace = False
            if color_priority > current_color_priority:
                should_replace = True
            elif color_priority == current_color_priority and overlap > current_overlap:
                should_replace = True
            elif color_priority == current_color_priority and overlap == current_overlap and url_value < existing:
                should_replace = True

            if should_replace:
                tiles[tile_id] = url_value
                if normalized_bbox is not None:
                    tile_bboxes[tile_id] = [float(v) for v in normalized_bbox]
                elif tile_id in tile_bboxes:
                    tile_bboxes.pop(tile_id, None)
                tile_overlap_by_id[tile_id] = overlap
                tile_color_priority_by_id[tile_id] = color_priority

        pages_seen += 1
        if returned == 0 or returned < page_size:
            break
        if feature_count > 0 and (start_index + returned) >= feature_count:
            break
        if pages_seen >= max_pages:
            break
        start_index += returned

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
            {},
        )

    if not tiles:
        if incompatible_grid_count > 0 and missing_url_count == 0:
            return (
                YearStatus(
                    year=year,
                    typename_exists=True,
                    feature_count=feature_count,
                    status="zero_features",
                    reason=(
                        f"Features found but none are compatible with requested SRS {srs} "
                        "(e.g. uklad_xy mismatch)."
                    ),
                ),
                {},
                {},
            )
        return (
            YearStatus(
                year=year,
                typename_exists=True,
                feature_count=feature_count,
                status="zero_features",
                reason="Features found but url_do_pobrania is missing in AOI response.",
            ),
            {},
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
        tile_bboxes,
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
    status, _, _ = await get_year_tiles(
        year=year,
        bbox=bbox,
        srs=srs,
        base_url=base_url,
        timeout=timeout,
        retry_policy=retry_policy,
        year_to_typename=year_to_typename,
    )
    return status
