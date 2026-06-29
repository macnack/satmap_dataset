from __future__ import annotations

from typing import Any

from satmap_dataset.geoportal.http import RetryPolicy, request_with_retry

# Each clause carries a {bbox} placeholder applied per-statement. The bbox area
# filter cannot be applied to a union group as a whole (e.g. `(a; b;)(bbox)` is a
# parse error), so it must sit on every individual way/relation statement.
#
# Area categories (green, water) also query `relation[...]` because large lakes
# and parks are commonly mapped as multipolygon relations rather than single
# closed ways; `out geom` returns their member geometries, which we assemble into
# polygons. Line/point categories (roads, paths) and buildings stay way-only.
CATEGORY_QUERIES: dict[str, str] = {
    "buildings": 'way["building"]({bbox})',
    "roads":     'way["highway"~"motorway|trunk|primary|secondary|tertiary|residential|service|living_street|unclassified"]({bbox})',
    "paths":     'way["highway"~"footway|cycleway|path|steps|pedestrian|track"]({bbox})',
    "green":     '(way["leisure"~"park|garden|pitch|playground|golf_course"]({bbox}); way["natural"~"wood|scrub|grass|meadow"]({bbox}); way["landuse"~"forest|meadow|grass|recreation_ground"]({bbox}); relation["leisure"~"park|garden|pitch|playground|golf_course"]({bbox}); relation["natural"~"wood|scrub|grass|meadow"]({bbox}); relation["landuse"~"forest|meadow|grass|recreation_ground"]({bbox});)',
    # NB: only `relation["natural"="water"]` (lakes/reservoirs/basins) — NOT
    # `relation["waterway"]`: rivers are mapped as waterway relations spanning the
    # whole river (hundreds of km), and `out geom` would return their entire
    # geometry, blowing the timeout and silently yielding nothing. River segments
    # inside the AOI are still captured as `way["waterway"]`.
    "water":     '(way["natural"="water"]({bbox}); way["waterway"]({bbox}); relation["natural"="water"]({bbox});)',
}

_DEFAULT_OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def _bbox_to_overpass(bbox_wgs84: str) -> str:
    """Convert lon_min,lat_min,lon_max,lat_max → Overpass S,W,N,E."""
    lon_min, lat_min, lon_max, lat_max = (float(x) for x in bbox_wgs84.split(","))
    return f"{lat_min},{lon_min},{lat_max},{lon_max}"


def _build_query(category: str, bbox_overpass: str, snapshot_date: str) -> str:
    clause = CATEGORY_QUERIES[category].format(bbox=bbox_overpass)
    date_tag = f'[date:"{snapshot_date}"]'
    return f'[out:json][timeout:55]{date_tag};{clause};out geom;'


def _assemble_rings(segments: list[list[list[float]]]) -> list[list[list[float]]]:
    """Stitch member-way segments into closed rings by matching shared endpoints.

    A multipolygon ring is often split across several member ways; `out geom`
    returns each member's geometry separately, so we join segments end-to-end
    until each ring closes. Open leftovers (never closed) are dropped.
    """
    segs = [[tuple(p) for p in s] for s in segments if len(s) >= 2]
    rings: list[list[list[float]]] = []
    while segs:
        ring = segs.pop(0)
        changed = True
        while ring[0] != ring[-1] and changed:
            changed = False
            for i, s in enumerate(segs):
                if s[0] == ring[-1]:
                    ring = ring + s[1:]
                elif s[-1] == ring[-1]:
                    ring = ring + s[-2::-1]
                elif s[-1] == ring[0]:
                    ring = s[:-1] + ring
                elif s[0] == ring[0]:
                    ring = s[:0:-1] + ring
                else:
                    continue
                segs.pop(i)
                changed = True
                break
        if len(ring) >= 4 and ring[0] == ring[-1]:
            rings.append([list(p) for p in ring])
    return rings


def _point_in_ring(pt: list[float], ring: list[list[float]]) -> bool:
    """Ray-casting point-in-polygon for assigning holes to their outer ring."""
    x, y = pt[0], pt[1]
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _relation_to_features(el: dict) -> list[dict[str, Any]]:
    """Assemble a multipolygon relation's member ways into Polygon features."""
    def members(role_filter) -> list[list[list[float]]]:
        out = []
        for m in el.get("members", []):
            if m.get("type") != "way":
                continue
            role = m.get("role", "")
            if not role_filter(role):
                continue
            geom = m.get("geometry") or []
            coords = [[n["lon"], n["lat"]] for n in geom if n]
            if len(coords) >= 2:
                out.append(coords)
        return out

    outers = _assemble_rings(members(lambda r: r in ("outer", "")))
    inners = _assemble_rings(members(lambda r: r == "inner"))
    if not outers:
        return []
    tags = el.get("tags", {})
    features: list[dict[str, Any]] = []
    for outer in outers:
        holes = [inner for inner in inners if _point_in_ring(inner[0], outer)]
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [outer, *holes]},
            "properties": tags,
        })
    return features


def _elements_to_geojson(overpass_result: dict) -> dict[str, Any]:
    """Convert Overpass JSON (out geom) → GeoJSON FeatureCollection.

    Ways become Polygons (closed) or LineStrings (open); relations are assembled
    into multipolygon Polygon features (one per outer ring, with holes).
    """
    features = []
    for el in overpass_result.get("elements", []):
        etype = el.get("type")
        if etype == "way":
            nodes = el.get("geometry", [])
            if len(nodes) < 2:
                continue
            coords = [[n["lon"], n["lat"]] for n in nodes]
            if coords[0] == coords[-1] and len(coords) >= 4:
                geometry: dict[str, Any] = {"type": "Polygon", "coordinates": [coords]}
            else:
                geometry = {"type": "LineString", "coordinates": coords}
            features.append({
                "type": "Feature",
                "geometry": geometry,
                "properties": el.get("tags", {}),
            })
        elif etype == "relation":
            features.extend(_relation_to_features(el))
    return {"type": "FeatureCollection", "features": features}


# Backwards-compatible alias (kept for any external importers).
_ways_to_geojson = _elements_to_geojson


async def get_elements_geometry(
    bbox_wgs84: str,
    category: str,
    snapshot_date: str,
    *,
    overpass_url: str = _DEFAULT_OVERPASS_URL,
    timeout: float = 60.0,
    retry_policy: RetryPolicy | None = None,
) -> dict[str, Any]:
    """Query Overpass API at a historical snapshot and return GeoJSON FeatureCollection."""
    normalized = snapshot_date if snapshot_date.endswith("Z") else snapshot_date + "T00:00:00Z"
    bbox_overpass = _bbox_to_overpass(bbox_wgs84)
    query = _build_query(category, bbox_overpass, normalized)
    response = await request_with_retry(
        "POST",
        overpass_url,
        data={"data": query},
        timeout=timeout,
        retry_policy=retry_policy,
        # overpass-api.de returns 406 Not Acceptable unless an explicit JSON
        # Accept header is sent (the httpx default `*/*` is rejected).
        headers={
            "Accept": "application/json",
            "User-Agent": "satmap_dataset/0.1 (OSM historical labels; +https://github.com)",
        },
    )
    return _elements_to_geojson(response.json())
