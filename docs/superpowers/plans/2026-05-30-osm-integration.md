# OSM Stage Integration into Location Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make OSM semantic-label masks a reliable, automatic part of `run-all-location-json` by replacing the non-working Overpass backend with the proven OSM-API (`/api/0.6/map` + timestamp heuristic) backend behind a config seam, and wiring an isolated OSM step into the location orchestrator.

**Architecture:** A new `osm/osm_api_client.py` downloads the OSM extent once per location (adaptive quadrant split on the node-limit), parses ways with `timestamp`/`version`, and reconstructs per-year historical state via a timestamp heuristic. `pipeline/osm.py` branches on `OsmConfig.backend` (`"osm_api"` now, `"overpass"` raises `NotImplementedError` as a future extension) and fetches once per location, slicing per (year, category). `run-all-location-json` calls `osm_pipeline.run()` as an isolated step after the orthophoto run succeeds — an OSM failure is recorded but never invalidates the orthophoto artifacts.

**Tech Stack:** Python ≥3.10, Pydantic v2, httpx (async, via existing `request_with_retry`), pyproj, GDAL CLI (`ogr2ogr`/`gdal_rasterize`), Pillow + tifffile (optional preview), Typer CLI, pytest.

---

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/satmap_dataset/osm/osm_api_client.py` | OSM-API fetch (adaptive split), parse ways, timestamp historical filter, category→GeoJSON |
| Create | `tests/test_osm_api_client.py` | unit tests for the new client (HTTP mocked) |
| Modify | `src/satmap_dataset/config.py` | add `backend` + `osm_render_preview` to `OsmConfig` |
| Modify | `src/satmap_dataset/pipeline/osm.py` | backend branch; fetch-once-per-location; optional preview hook |
| Create | `src/satmap_dataset/osm/preview.py` | render per-year overlay PNG of masks on the ortho |
| Modify | `src/satmap_dataset/cli.py` | isolated OSM step in `run-all-location-json` + `--osm/--no-osm` gate |
| Modify | `tests/test_osm_pipeline.py` | swap mock seam to `osm_api_client`; backend-branch tests |
| Modify | `tests/test_osm_cli.py` | `backend` validator; run-all OSM-step gating |
| Modify | `configs/run/base.json` | add `"fetch_osm": true` |
| Create | `tests/test_osm_preview.py` | preview PNG creation test (tifffile/PIL mocked) |

`overpass_client.py` is retained untouched as the future backend skeleton. `ohsome_client.py`'s only remaining use (`bbox_epsg2180_to_wgs84`) moves to `osm_api_client.py`; the old module is left in place (still imported by tests) but no longer used by the pipeline.

---

## Task 1: `OsmConfig.backend` + `osm_render_preview`

**Files:**
- Modify: `src/satmap_dataset/config.py`
- Test: `tests/test_osm_cli.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_osm_cli.py`:

```python
def test_osm_config_backend_default_and_validation():
    cfg = OsmConfig(bbox="0,0,10,10")
    assert cfg.backend == "osm_api"
    assert cfg.osm_render_preview is True
    ok = OsmConfig(bbox="0,0,10,10", backend="overpass")
    assert ok.backend == "overpass"
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        OsmConfig(bbox="0,0,10,10", backend="nonsense")
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_osm_cli.py -k backend -q`
Expected: FAIL — `AttributeError`/`ValidationError` (no `backend` field).

- [ ] **Step 3: Implement the fields**

In `src/satmap_dataset/config.py`, inside `class OsmConfig`, add after the `overpass_url` line (currently line 487):

```python
    backend: str = "osm_api"
    osm_render_preview: bool = True
```

And add this validator alongside the existing `OsmConfig` validators (after `validate_categories`):

```python
    @field_validator("backend")
    @classmethod
    def validate_backend(cls, value: str) -> str:
        allowed = {"osm_api", "overpass"}
        normalized = str(value).strip().lower()
        if normalized not in allowed:
            raise ValueError(f"backend must be one of {sorted(allowed)}")
        return normalized
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_osm_cli.py -k backend -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/config.py tests/test_osm_cli.py
git commit -m "feat(osm): add backend + osm_render_preview to OsmConfig"
```

---

## Task 2: `osm_api_client.py` — pure functions

**Files:**
- Create: `src/satmap_dataset/osm/osm_api_client.py`
- Test: `tests/test_osm_api_client.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_osm_api_client.py`:

```python
from datetime import datetime, timezone

from satmap_dataset.osm import osm_api_client as c


SAMPLE_XML = b"""<?xml version="1.0"?>
<osm version="0.6">
  <node id="1" lon="16.7790" lat="52.4220"/>
  <node id="2" lon="16.7800" lat="52.4220"/>
  <node id="3" lon="16.7800" lat="52.4230"/>
  <node id="4" lon="16.7790" lat="52.4230"/>
  <way id="100" version="2" timestamp="2019-01-01T00:00:00Z">
    <nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="1"/>
    <tag k="building" v="yes"/>
  </way>
  <way id="200" version="1" timestamp="2024-03-01T00:00:00Z">
    <nd ref="1"/><nd ref="2"/>
    <tag k="highway" v="residential"/>
  </way>
</osm>
"""


def test_bbox_epsg2180_to_wgs84_known_values():
    out = c.bbox_epsg2180_to_wgs84("348967.353,508503.706,349967.353,509503.706")
    lon_min, lat_min, lon_max, lat_max = (float(x) for x in out.split(","))
    assert abs(lon_min - 16.778248) < 0.001
    assert abs(lat_min - 52.421547) < 0.001
    assert abs(lon_max - 16.792497) < 0.001
    assert abs(lat_max - 52.430809) < 0.001


def test_parse_ways_builds_geometry_and_metadata():
    ways = c.parse_ways([SAMPLE_XML])
    assert len(ways) == 2
    by_id = {len(w.coords): w for w in ways}
    building = next(w for w in ways if w.tags.get("building") == "yes")
    assert building.ver == 2
    assert building.coords[0] == building.coords[-1]  # closed ring
    assert building.ts == datetime(2019, 1, 1, tzinfo=timezone.utc)


def test_parse_ways_dedupes_across_chunks():
    ways = c.parse_ways([SAMPLE_XML, SAMPLE_XML])
    assert len(ways) == 2  # same ids deduped


def test_existed_at_truth_table():
    old = c.Way(tags={}, coords=[(0,0),(1,1)], ts=datetime(2019,1,1,tzinfo=timezone.utc), ver=1)
    new_v1 = c.Way(tags={}, coords=[(0,0),(1,1)], ts=datetime(2024,3,1,tzinfo=timezone.utc), ver=1)
    new_v2 = c.Way(tags={}, coords=[(0,0),(1,1)], ts=datetime(2024,3,1,tzinfo=timezone.utc), ver=2)
    assert c.existed_at(old, "2022-04-29") is True       # predates target
    assert c.existed_at(new_v1, "2022-04-29") is False   # born after target, never edited
    assert c.existed_at(new_v2, "2022-04-29") is True    # edited after target but pre-existed


def test_category_tags_cover_five_classes():
    assert set(c.CATEGORY_TAGS.keys()) == {"buildings", "roads", "paths", "green", "water"}
    assert c.CATEGORY_TAGS["buildings"]({"building": "yes"}) is True
    assert c.CATEGORY_TAGS["roads"]({"highway": "residential"}) is True
    assert c.CATEGORY_TAGS["paths"]({"highway": "footway"}) is True
    assert c.CATEGORY_TAGS["green"]({"leisure": "park"}) is True
    assert c.CATEGORY_TAGS["water"]({"natural": "water"}) is True
    assert c.CATEGORY_TAGS["roads"]({"highway": "footway"}) is False


def test_features_for_filters_by_category_and_date():
    ways = c.parse_ways([SAMPLE_XML])
    # 2022: building (v2, pre-existed) present; road (v1, born 2024) absent
    gj_2022 = c.features_for(ways, "buildings", "2022-04-29")
    assert len(gj_2022["features"]) == 1
    assert gj_2022["features"][0]["geometry"]["type"] == "Polygon"
    roads_2022 = c.features_for(ways, "roads", "2022-04-29")
    assert len(roads_2022["features"]) == 0
    roads_2024 = c.features_for(ways, "roads", "2024-09-17")
    assert len(roads_2024["features"]) == 1
    assert roads_2024["features"][0]["geometry"]["type"] == "LineString"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_osm_api_client.py -q`
Expected: FAIL — `ModuleNotFoundError: ...osm_api_client`.

- [ ] **Step 3: Implement the pure functions**

Create `src/satmap_dataset/osm/osm_api_client.py`:

```python
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from pyproj import Transformer

OSM_MAP_URL = "https://api.openstreetmap.org/api/0.6/map"

CATEGORY_TAGS: dict[str, Callable[[dict], bool]] = {
    "buildings": lambda t: "building" in t,
    "roads": lambda t: t.get("highway") in {
        "motorway", "trunk", "primary", "secondary", "tertiary",
        "residential", "service", "living_street", "unclassified",
    },
    "paths": lambda t: t.get("highway") in {
        "footway", "cycleway", "path", "steps", "pedestrian", "track",
    },
    "green": lambda t: (
        t.get("leisure") in {"park", "garden", "pitch", "playground", "golf_course"}
        or t.get("natural") in {"wood", "scrub", "grass", "meadow"}
        or t.get("landuse") in {"forest", "meadow", "grass", "recreation_ground"}
    ),
    "water": lambda t: t.get("natural") == "water" or "waterway" in t,
}


@dataclass
class Way:
    tags: dict[str, str]
    coords: list[tuple[float, float]]
    ts: datetime
    ver: int = 1


def bbox_epsg2180_to_wgs84(bbox_2180: str) -> str:
    """Return OSM-format bbox: lon_min,lat_min,lon_max,lat_max in WGS84."""
    xmin, ymin, xmax, ymax = (float(x) for x in bbox_2180.split(","))
    t = Transformer.from_crs("EPSG:2180", "EPSG:4326", always_xy=True)
    lon_min, lat_min = t.transform(xmin, ymin)
    lon_max, lat_max = t.transform(xmax, ymax)
    return f"{lon_min:.6f},{lat_min:.6f},{lon_max:.6f},{lat_max:.6f}"


def _parse_ts(ts_str: str) -> datetime:
    if not ts_str:
        return datetime(2000, 1, 1, tzinfo=timezone.utc)
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))


def parse_ways(xml_chunks: list[bytes]) -> list[Way]:
    """Parse OSM XML chunks into deduplicated Way objects (nodes resolved to coords)."""
    nodes: dict[str, tuple[float, float]] = {}
    raw: list[tuple[str, dict, list[str], str, int]] = []
    for chunk in xml_chunks:
        root = ET.fromstring(chunk)
        for n in root.findall("node"):
            nodes[n.get("id")] = (float(n.get("lon")), float(n.get("lat")))
        for w in root.findall("way"):
            tags = {t.get("k"): t.get("v") for t in w.findall("tag")}
            refs = [nd.get("ref") for nd in w.findall("nd")]
            raw.append((w.get("id"), tags, refs, w.get("timestamp", ""), int(w.get("version", "1"))))
    seen: set[str] = set()
    ways: list[Way] = []
    for wid, tags, refs, ts_str, ver in raw:
        if wid in seen:
            continue
        seen.add(wid)
        coords = [nodes[r] for r in refs if r in nodes]
        if len(coords) < 2:
            continue
        ways.append(Way(tags=tags, coords=coords, ts=_parse_ts(ts_str), ver=ver))
    return ways


def existed_at(way: Way, date_str: str) -> bool:
    """Historical heuristic: way existed at date if its current version predates
    the date, or it has been edited (version>1) and thus pre-existed."""
    target = datetime.fromisoformat(date_str[:10] + "T00:00:00+00:00")
    return way.ts <= target or way.ver > 1


def ways_to_geojson(ways: list[Way]) -> dict[str, Any]:
    features = []
    for w in ways:
        c = w.coords
        if c[0] == c[-1] and len(c) >= 4:
            geom: dict[str, Any] = {"type": "Polygon", "coordinates": [[list(p) for p in c]]}
        else:
            geom = {"type": "LineString", "coordinates": [list(p) for p in c]}
        features.append({"type": "Feature", "geometry": geom, "properties": w.tags})
    return {"type": "FeatureCollection", "features": features}


def features_for(ways: list[Way], category: str, snapshot_date: str) -> dict[str, Any]:
    """Filter parsed ways by category tags and historical existence, return GeoJSON."""
    pred = CATEGORY_TAGS[category]
    selected = [w for w in ways if pred(w.tags) and existed_at(w, snapshot_date)]
    return ways_to_geojson(selected)
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_osm_api_client.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/osm/osm_api_client.py tests/test_osm_api_client.py
git commit -m "feat(osm): osm_api_client pure functions (parse/filter/geojson)"
```

---

## Task 3: `osm_api_client` — async fetch with adaptive quadrant split

**Files:**
- Modify: `src/satmap_dataset/osm/osm_api_client.py`
- Test: `tests/test_osm_api_client.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_osm_api_client.py`:

```python
import asyncio
import httpx


def test_quadrants_splits_bbox_into_four():
    quads = c._quadrants("16.0,52.0,16.4,52.2")
    assert len(quads) == 4
    # union covers original extent
    xs0 = [float(q.split(",")[0]) for q in quads]
    ys3 = [float(q.split(",")[3]) for q in quads]
    assert min(xs0) == 16.0
    assert max(ys3) == 52.2


def test_fetch_and_parse_single_request(monkeypatch):
    calls = []

    async def fake_req(method, url, *, params, timeout, retry_policy, **kw):
        calls.append(params["bbox"])
        class R:
            content = SAMPLE_XML
        return R()

    monkeypatch.setattr(c, "request_with_retry", fake_req)
    ways = asyncio.run(c.fetch_and_parse("16.778,52.421,16.792,52.430", timeout=5, retry_policy=None))
    assert len(ways) == 2
    assert len(calls) == 1


def test_fetch_and_parse_splits_on_http_400(monkeypatch):
    calls = []

    async def fake_req(method, url, *, params, timeout, retry_policy, **kw):
        bbox = params["bbox"]
        calls.append(bbox)
        # The full bbox 400s; quadrants succeed.
        if len(calls) == 1:
            req = httpx.Request("GET", url)
            resp = httpx.Response(400, request=req)
            raise httpx.HTTPStatusError("bad", request=req, response=resp)
        class R:
            content = SAMPLE_XML
        return R()

    monkeypatch.setattr(c, "request_with_retry", fake_req)
    ways = asyncio.run(c.fetch_and_parse("16.0,52.0,16.4,52.2", timeout=5, retry_policy=None))
    # 1 failed full + 4 quadrant requests
    assert len(calls) == 5
    assert len(ways) == 2  # deduped across quadrant chunks
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_osm_api_client.py -k "fetch or quadrant" -q`
Expected: FAIL — `AttributeError` (`_quadrants`/`fetch_and_parse` missing).

- [ ] **Step 3: Implement fetch + split**

Append to `src/satmap_dataset/osm/osm_api_client.py`:

```python
import httpx  # noqa: E402  (kept near async section for clarity)

from satmap_dataset.geoportal.http import RetryPolicy, request_with_retry  # noqa: E402

_MAX_SPLIT_DEPTH = 4


def _quadrants(bbox_wgs84: str) -> list[str]:
    lon_min, lat_min, lon_max, lat_max = (float(x) for x in bbox_wgs84.split(","))
    lon_mid = (lon_min + lon_max) / 2
    lat_mid = (lat_min + lat_max) / 2
    return [
        f"{lon_min},{lat_min},{lon_mid},{lat_mid}",
        f"{lon_mid},{lat_min},{lon_max},{lat_mid}",
        f"{lon_min},{lat_mid},{lon_mid},{lat_max}",
        f"{lon_mid},{lat_mid},{lon_max},{lat_max}",
    ]


async def fetch_osm_xml(
    bbox_wgs84: str,
    *,
    timeout: float,
    retry_policy: RetryPolicy | None,
    _depth: int = 0,
) -> list[bytes]:
    """GET the OSM /map extent. On HTTP 400 (node-limit), recursively split into
    quadrants and concatenate the resulting XML chunks."""
    try:
        resp = await request_with_retry(
            "GET", OSM_MAP_URL,
            params={"bbox": bbox_wgs84},
            timeout=timeout, retry_policy=retry_policy,
        )
        return [resp.content]
    except httpx.HTTPStatusError as exc:
        if exc.response is not None and exc.response.status_code == 400 and _depth < _MAX_SPLIT_DEPTH:
            chunks: list[bytes] = []
            for sub in _quadrants(bbox_wgs84):
                chunks.extend(await fetch_osm_xml(
                    sub, timeout=timeout, retry_policy=retry_policy, _depth=_depth + 1,
                ))
            return chunks
        raise


async def fetch_and_parse(
    bbox_wgs84: str,
    *,
    timeout: float,
    retry_policy: RetryPolicy | None,
) -> list[Way]:
    """Download (adaptive quadrants) and parse the OSM extent once per location."""
    chunks = await fetch_osm_xml(bbox_wgs84, timeout=timeout, retry_policy=retry_policy)
    return parse_ways(chunks)
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_osm_api_client.py -q`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/osm/osm_api_client.py tests/test_osm_api_client.py
git commit -m "feat(osm): osm_api_client adaptive-split fetch_and_parse"
```

---

## Task 4: Rewire `pipeline/osm.py` to the osm_api backend (fetch once per location)

**Files:**
- Modify: `src/satmap_dataset/pipeline/osm.py`
- Test: `tests/test_osm_pipeline.py`

- [ ] **Step 1: Update the test mock seam + add backend tests**

In `tests/test_osm_pipeline.py`, replace the `_patch_seams` function (currently lines 47-62) with:

```python
def _patch_seams(monkeypatch, *, features_by_cat=None):
    counts = features_by_cat or {"buildings": 5, "roads": 3, "paths": 2, "green": 2, "water": 1}

    async def _fake_fetch_and_parse(bbox_wgs84, *, timeout, retry_policy):
        return ["WAYS_SENTINEL"]  # opaque; features_for is also patched

    def _fake_features_for(ways, category, snapshot_date):
        n = counts.get(category, 0)
        return {
            "type": "FeatureCollection",
            "features": [{"type": "Feature", "geometry": None, "properties": {}} for _ in range(n)],
        }

    def _fake_rasterize(geojson, out_path, *, target_bbox, target_width, target_height, target_srs="EPSG:2180"):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"RASTER")

    monkeypatch.setattr(osm_pipeline.osm_api_client, "fetch_and_parse", _fake_fetch_and_parse)
    monkeypatch.setattr(osm_pipeline.osm_api_client, "features_for", _fake_features_for)
    monkeypatch.setattr(osm_pipeline.rasterize, "rasterize_geojson_to_file", _fake_rasterize)
    monkeypatch.setattr(osm_pipeline, "_maybe_render_preview", lambda *a, **k: None)
```

Then append a backend test:

```python
def test_run_overpass_backend_not_implemented(tmp_path, monkeypatch):
    _patch_seams(monkeypatch)
    render = _make_render_manifest(tmp_path, years_dates={2022: "2022-04-29"})
    cfg = OsmConfig(
        bbox="0,0,100,100",
        osm_root=tmp_path / "osm_x",
        output_json=tmp_path / "osm_x" / "osm_manifest.json",
        render_manifest=render,
        categories=["buildings"],
        target_width=10, target_height=10,
        backend="overpass",
        sleep_min=0.0, sleep_max=0.0,
    )
    code, path = osm_pipeline.run(cfg)
    assert code == 1
    from satmap_dataset.models import OsmManifest
    manifest = OsmManifest.model_validate_json(Path(path).read_text())
    assert manifest.passed is False
    assert any("future extension" in e or "overpass" in e for e in manifest.errors)
```

Note: existing tests (`test_run_writes_rasters_and_manifest`, `test_run_zero_features_no_raster`, `test_run_uses_acquisition_date_not_jan1`, `test_run_reuses_existing_raster`) keep working through the new seam unchanged.

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_osm_pipeline.py -q`
Expected: FAIL — `AttributeError: module ... has no attribute 'osm_api_client'` and the new test errors.

- [ ] **Step 3: Rewrite `pipeline/osm.py`**

Replace the entire body of `src/satmap_dataset/pipeline/osm.py` with:

```python
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from satmap_dataset.config import OsmConfig
from satmap_dataset.models import OsmCategoryAsset, OsmManifest, OsmYearAsset
from satmap_dataset.osm import osm_api_client, rasterize
from satmap_dataset.osm.osm_api_client import bbox_epsg2180_to_wgs84

logger = logging.getLogger("satmap_dataset.osm")


def _read_year_date_map(config: OsmConfig) -> dict[int, str]:
    if config.year_date_map is not None:
        return dict(config.year_date_map)
    if config.render_manifest is not None and Path(config.render_manifest).exists():
        from satmap_dataset.models import DatasetManifest

        manifest = DatasetManifest.model_validate_json(
            Path(config.render_manifest).read_text(encoding="utf-8")
        )
        result: dict[int, str] = {}
        for year, tile_acq in manifest.tile_acquisition_by_year.items():
            dates = [v.acquisition_date for v in tile_acq.values() if v.acquisition_date is not None]
            if dates:
                result[int(year)] = max(dates)
        return result
    raise ValueError(
        "OsmConfig requires either render_manifest (with tile_acquisition_by_year) "
        "or year_date_map to determine per-year snapshot dates."
    )


def _read_grid(config: OsmConfig) -> tuple[int, int]:
    if config.target_width is not None and config.target_height is not None:
        return config.target_width, config.target_height
    if config.render_manifest is not None and Path(config.render_manifest).exists():
        from satmap_dataset.models import DatasetManifest

        manifest = DatasetManifest.model_validate_json(
            Path(config.render_manifest).read_text(encoding="utf-8")
        )
        if manifest.target_width is not None and manifest.target_height is not None:
            return int(manifest.target_width), int(manifest.target_height)
    xmin, ymin, xmax, ymax = (float(x) for x in config.bbox.split(","))
    return max(1, round(xmax - xmin)), max(1, round(ymax - ymin))


def _maybe_render_preview(config: OsmConfig, year: int, year_asset: OsmYearAsset) -> None:
    """Render an optional per-year overlay PNG; never raises."""
    if not config.osm_render_preview:
        return
    try:
        from satmap_dataset.osm.preview import render_year_preview

        render_year_preview(config, year, year_asset)
    except Exception as exc:  # noqa: BLE001 - preview is best-effort
        logger.warning("OSM preview failed year=%s: %s", year, exc)


def _write_manifest(config, bbox_wgs84, target_width, target_height, year_assets, errors, passed):
    manifest = OsmManifest(
        bbox=config.bbox,
        bbox_wgs84=bbox_wgs84,
        srs=config.srs,
        target_width=target_width,
        target_height=target_height,
        categories=list(config.categories),
        years=year_assets,
        passed=passed,
        errors=errors,
        run_parameters=config.model_dump(mode="json"),
    )
    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    config.output_json.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


async def _run_async(config: OsmConfig) -> tuple[int, Path]:
    from satmap_dataset.geoportal.http import RetryPolicy

    errors: list[str] = []
    year_assets: list[OsmYearAsset] = []

    if config.backend != "osm_api":
        errors.append(
            f"backend={config.backend!r} is a future extension; only 'osm_api' is implemented"
        )
        _write_manifest(config, "", config.target_width, config.target_height, [], errors, False)
        return 1, config.output_json

    try:
        year_date_map = _read_year_date_map(config)
    except ValueError as exc:
        errors.append(str(exc))
        _write_manifest(config, "", config.target_width, config.target_height, [], errors, False)
        return 1, config.output_json

    target_width, target_height = _read_grid(config)
    target_bbox = tuple(float(x) for x in config.bbox.split(","))
    bbox_wgs84 = bbox_epsg2180_to_wgs84(config.bbox)
    retry_policy = RetryPolicy(max_attempts=config.retries, backoff_seconds=config.retry_delay)

    # Download + parse the OSM extent ONCE for the whole location.
    try:
        ways = await osm_api_client.fetch_and_parse(
            bbox_wgs84, timeout=config.timeout, retry_policy=retry_policy,
        )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"fetch_and_parse failed: {exc}")
        _write_manifest(config, bbox_wgs84, target_width, target_height, [], errors, False)
        return 1, config.output_json

    for year in sorted(year_date_map.keys()):
        snapshot_date = year_date_map[year]
        normalized = snapshot_date if snapshot_date.endswith("Z") else snapshot_date + "T00:00:00Z"
        year_asset = OsmYearAsset(year=year, snapshot_date=normalized)

        for category in config.categories:
            raster_path = config.osm_root / f"year_{year}_{category}.tif"
            if raster_path.exists() and not config.overwrite:
                year_asset.categories[category] = OsmCategoryAsset(
                    feature_count=0, raster_path=str(raster_path)
                )
                continue
            try:
                geojson = osm_api_client.features_for(ways, category, normalized)
                count = len(geojson.get("features") or [])
                if count == 0:
                    year_asset.categories[category] = OsmCategoryAsset(
                        feature_count=0, raster_path=None
                    )
                else:
                    rasterize.rasterize_geojson_to_file(
                        geojson, raster_path,
                        target_bbox=target_bbox,
                        target_width=target_width, target_height=target_height,
                        target_srs=config.srs,
                    )
                    year_asset.categories[category] = OsmCategoryAsset(
                        feature_count=count, raster_path=str(raster_path)
                    )
            except Exception as exc:  # noqa: BLE001
                year_asset.errors.append(f"{category}: {exc}")
                errors.append(f"year={year} category={category}: {exc}")

        year_asset.passed = not year_asset.errors
        year_assets.append(year_asset)
        _maybe_render_preview(config, year, year_asset)

    passed = bool(year_assets) and all(a.passed for a in year_assets)
    _write_manifest(config, bbox_wgs84, target_width, target_height, year_assets, errors, passed)
    logger.info(
        "OSM run: years=%s passed=%s errors=%s",
        [a.year for a in year_assets], passed, len(errors),
    )
    return (0 if passed else 1), config.output_json


def run(config: OsmConfig) -> tuple[int, Path]:
    return asyncio.run(_run_async(config))
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_osm_pipeline.py -q`
Expected: PASS (all, including the new overpass-backend test).

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/pipeline/osm.py tests/test_osm_pipeline.py
git commit -m "feat(osm): pipeline uses osm_api backend, fetch once per location"
```

---

## Task 5: Per-year preview PNG

**Files:**
- Create: `src/satmap_dataset/osm/preview.py`
- Test: `tests/test_osm_preview.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_osm_preview.py`:

```python
import numpy as np

from satmap_dataset.config import OsmConfig
from satmap_dataset.models import OsmCategoryAsset, OsmYearAsset
from satmap_dataset.osm import preview


def test_render_year_preview_writes_png(tmp_path, monkeypatch):
    osm_root = tmp_path / "osm_x"
    render_root = tmp_path / "rendered_x"
    osm_root.mkdir(); render_root.mkdir()
    # build mask + ortho stand-ins
    (osm_root / "year_2022_buildings.tif").write_bytes(b"x")
    ortho = render_root / "year_2022.tiff"
    ortho.write_bytes(b"x")

    def fake_imread(path):
        p = str(path)
        if p.endswith("year_2022.tiff"):
            return np.zeros((40, 40, 3), dtype=np.uint8)
        return (np.ones((40, 40), dtype=np.uint8) * 255)

    monkeypatch.setattr(preview.tifffile, "imread", fake_imread)

    cfg = OsmConfig(
        bbox="0,0,40,40",
        osm_root=osm_root,
        output_json=osm_root / "osm_manifest.json",
        render_root=render_root,
        categories=["buildings"],
        target_width=40, target_height=40,
        osm_render_preview=True,
    )
    year_asset = OsmYearAsset(
        year=2022, snapshot_date="2022-04-29T00:00:00Z",
        categories={"buildings": OsmCategoryAsset(feature_count=3, raster_path=str(osm_root / "year_2022_buildings.tif"))},
        passed=True,
    )
    preview.render_year_preview(cfg, 2022, year_asset)
    assert (osm_root / "viz_2022.png").exists()


def test_render_year_preview_skips_when_ortho_missing(tmp_path):
    osm_root = tmp_path / "osm_x"; osm_root.mkdir()
    cfg = OsmConfig(
        bbox="0,0,40,40", osm_root=osm_root,
        output_json=osm_root / "osm_manifest.json",
        render_root=tmp_path / "rendered_x",
        categories=["buildings"], target_width=40, target_height=40,
    )
    ya = OsmYearAsset(year=2099, snapshot_date="2099-01-01T00:00:00Z", categories={}, passed=True)
    # No ortho for 2099 → silent no-op, no exception, no file
    preview.render_year_preview(cfg, 2099, ya)
    assert not (osm_root / "viz_2099.png").exists()
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_osm_preview.py -q`
Expected: FAIL — `ModuleNotFoundError: ...osm.preview` and `OsmConfig` has no `render_root`.

- [ ] **Step 3a: Add `render_root` to `OsmConfig`**

In `src/satmap_dataset/config.py`, inside `class OsmConfig`, add after the `osm_root` line:

```python
    render_root: Path | None = None
```

- [ ] **Step 3b: Implement the preview module**

Create `src/satmap_dataset/osm/preview.py`:

```python
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from satmap_dataset.config import OsmConfig
from satmap_dataset.models import OsmYearAsset

logger = logging.getLogger("satmap_dataset.osm.preview")

_COLORS = {
    "buildings": (220, 60, 60, 175),
    "roads": (255, 200, 50, 200),
    "paths": (255, 130, 30, 160),
    "green": (60, 180, 60, 130),
    "water": (30, 120, 255, 160),
}
_PREVIEW_MAX = 1000  # longest output side in px


def render_year_preview(config: OsmConfig, year: int, year_asset: OsmYearAsset) -> None:
    """Render a downscaled overlay of OSM masks on the ortho for `year`.
    Silent no-op when the ortho for the year is absent."""
    if config.render_root is None:
        return
    ortho_path = Path(config.render_root) / f"year_{year}.tiff"
    if not ortho_path.exists():
        return

    ortho = tifffile.imread(str(ortho_path))
    h = ortho.shape[0]
    step = max(1, h // _PREVIEW_MAX)
    thumb = ortho[::step, ::step]
    base = Image.fromarray(thumb.astype(np.uint8)).convert("RGBA")

    overlay = np.zeros((*thumb.shape[:2], 4), dtype=np.uint8)
    # buildings drawn last (on top): iterate reversed so first category ends on top
    for category in reversed(list(config.categories)):
        asset = year_asset.categories.get(category)
        if asset is None or asset.raster_path is None:
            continue
        mask = tifffile.imread(asset.raster_path)[::step, ::step]
        overlay[mask > 0] = _COLORS.get(category, (255, 255, 255, 150))

    composite = Image.alpha_composite(base, Image.fromarray(overlay, "RGBA")).convert("RGB")
    out = Path(config.osm_root) / f"viz_{year}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    composite.save(out)
    logger.info("OSM preview written: %s", out)
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_osm_preview.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/osm/preview.py src/satmap_dataset/config.py tests/test_osm_preview.py
git commit -m "feat(osm): per-year overlay preview PNG"
```

---

## Task 6: Guard test — `render_root` flows into `OsmConfig`

**Files:**
- Test: `tests/test_osm_cli.py`

No production code change is needed: `_apply_location_paths_policy` already injects `render_root` = `rendered_<slug>` into the merged dict (`cli.py:272`), and `_build_osm_config_from_base_and_location` calls that policy. Once Task 5 added the `render_root` field to `OsmConfig`, `OsmConfig.model_validate(merged)` carries it automatically. This task adds a regression guard so a future edit can't silently drop it.

- [ ] **Step 1: Write the guard test**

Append to `tests/test_osm_cli.py`:

```python
def test_build_osm_config_sets_render_root(tmp_path):
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"categories": ["buildings"]}))
    loc = tmp_path / "loc_poznan.json"
    loc.write_text(json.dumps({
        "location_name": "Poznań", "center_lat": 52.4, "center_lon": 16.9, "square_km": 1.0,
    }))
    cfg = _build_osm_config_from_base_and_location(base_json=base, location_json=loc)
    assert cfg.render_root is not None
    assert str(cfg.render_root).endswith("rendered_poznan")
```

- [ ] **Step 2: Run to verify it passes**

Run: `pytest tests/test_osm_cli.py -k render_root -q`
Expected: PASS immediately (the field added in Task 5 + the existing path policy make this work with no new production code). If it FAILS, Task 5's `render_root` field is missing — fix Task 5 before continuing.

- [ ] **Step 3: Commit**

```bash
git add tests/test_osm_cli.py
git commit -m "test(osm): guard that render_root reaches OsmConfig for previews"
```

---

## Task 7: Isolated OSM step in `run-all-location-json`

**Files:**
- Modify: `src/satmap_dataset/cli.py`
- Test: `tests/test_osm_cli.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_osm_cli.py`:

```python
def _write_base_loc(tmp_path, fetch_osm=None):
    base_obj = {"year_start": 2022, "year_end": 2022, "mode": "hybrid", "profile": "train", "area_km2": 1.0}
    if fetch_osm is not None:
        base_obj["fetch_osm"] = fetch_osm
    base = tmp_path / "base.json"
    base.write_text(json.dumps(base_obj))
    locdir = tmp_path / "locations"; locdir.mkdir()
    (locdir / "poznan.json").write_text(json.dumps({
        "location_name": "Poznań", "center_lat": 52.4, "center_lon": 16.9,
    }))
    return base, locdir


def test_run_all_invokes_osm_step_by_default(tmp_path, monkeypatch):
    base, locdir = _write_base_loc(tmp_path)
    osm_calls = []

    def fake_run_all(config):
        config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        rep = config.artifacts_dir / "validation_report.json"
        rep.write_text('{"passed": true}')
        return (0, rep)

    def fake_osm_run(config):
        osm_calls.append(config)
        return (0, Path(config.output_json))

    monkeypatch.setattr("satmap_dataset.pipeline.run_all.run", fake_run_all)
    monkeypatch.setattr("satmap_dataset.pipeline.osm.run", fake_osm_run)
    monkeypatch.setattr("satmap_dataset.cli._has_successful_validation_artifact", lambda d: False)

    result = runner.invoke(app, ["run-all-location-json", "--locations-dir", str(locdir), "--base-json", str(base)])
    assert result.exit_code == 0
    assert len(osm_calls) == 1


def test_run_all_no_osm_flag_skips_step(tmp_path, monkeypatch):
    base, locdir = _write_base_loc(tmp_path)
    osm_calls = []

    def fake_run_all(config):
        config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        rep = config.artifacts_dir / "validation_report.json"
        rep.write_text('{"passed": true}')
        return (0, rep)

    monkeypatch.setattr("satmap_dataset.pipeline.run_all.run", fake_run_all)
    monkeypatch.setattr("satmap_dataset.pipeline.osm.run", lambda c: osm_calls.append(c) or (0, Path(c.output_json)))
    monkeypatch.setattr("satmap_dataset.cli._has_successful_validation_artifact", lambda d: False)

    result = runner.invoke(app, ["run-all-location-json", "--locations-dir", str(locdir), "--base-json", str(base), "--no-osm"])
    assert result.exit_code == 0
    assert len(osm_calls) == 0


def test_run_all_osm_failure_does_not_break_orthos(tmp_path, monkeypatch):
    base, locdir = _write_base_loc(tmp_path)

    def fake_run_all(config):
        config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        rep = config.artifacts_dir / "validation_report.json"
        rep.write_text('{"passed": true}')
        return (0, rep)

    def boom(config):
        raise RuntimeError("osm network down")

    monkeypatch.setattr("satmap_dataset.pipeline.run_all.run", fake_run_all)
    monkeypatch.setattr("satmap_dataset.pipeline.osm.run", boom)
    monkeypatch.setattr("satmap_dataset.cli._has_successful_validation_artifact", lambda d: False)

    result = runner.invoke(app, ["run-all-location-json", "--locations-dir", str(locdir), "--base-json", str(base)])
    # OSM failure → overall exit 1 (recorded failure) but orthos completed
    assert result.exit_code == 1
    assert "osm" in result.stdout.lower()
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_osm_cli.py -k "run_all_invokes_osm or no_osm_flag or osm_failure" -q`
Expected: FAIL — `--osm/--no-osm` option does not exist; no OSM step runs.

- [ ] **Step 3: Add the `--osm/--no-osm` option and isolated step**

In `src/satmap_dataset/cli.py`, modify `run_all_location_json_command`. Add a new option parameter after `continue_on_error` (around line 1200):

```python
    osm: bool = typer.Option(
        True,
        "--osm/--no-osm",
        help="After orthophoto run, fetch OSM semantic-label masks for the location.",
    ),
```

Then, inside the per-location loop, after the existing orthophoto exit-code handling block (currently lines 1229-1234):

```python
        exit_code, artifact_path = run_all.run(config)
        console.print(str(artifact_path))
        if exit_code != 0:
            failures.append(f"{location_json}: exit={exit_code}")
            if not continue_on_error:
                raise typer.Exit(code=exit_code)
            continue
```

add the isolated OSM step (note the added `continue` above so OSM only runs on orthophoto success):

```python
        # Isolated OSM step — never invalidates the orthophoto artifacts.
        merged_dict = {
            **_load_params_json_dict(base_json),
            **_load_params_json_dict(location_json),
        }
        fetch_osm = bool(merged_dict.get("fetch_osm", True)) and osm
        if fetch_osm:
            try:
                osm_cfg = _build_osm_config_from_base_and_location(
                    base_json=base_json, location_json=location_json,
                )
                osm_code, osm_path = osm_pipeline.run(osm_cfg)
                console.print(str(osm_path))
                if osm_code != 0:
                    failures.append(f"{location_json}: osm exit={osm_code}")
            except Exception as osm_exc:  # noqa: BLE001 - OSM must not break orthos
                console.print(f"[yellow]osm step failed:[/yellow] {osm_exc}")
                failures.append(f"{location_json}: osm {osm_exc}")
```

IMPORTANT: the existing code currently does NOT `continue` after a successful orthophoto run (it falls through to the next iteration). Adding the `continue` inside the `if exit_code != 0` block changes only the failure path — confirm the success path now falls through into the OSM step. The block above is placed immediately after, in the same loop body.

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_osm_cli.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/cli.py tests/test_osm_cli.py
git commit -m "feat(osm): isolated OSM step in run-all-location-json with --osm/--no-osm"
```

---

## Task 8: `fetch_osm` default in base.json + full suite

**Files:**
- Modify: `configs/run/base.json`

- [ ] **Step 1: Add the flag**

In `configs/run/base.json`, add a top-level key (after `"overwrite": false`):

```json
  "fetch_osm": true
```

(Add a comma to the preceding line as needed so the JSON stays valid.)

- [ ] **Step 2: Verify base.json parses**

Run: `python -c "import json; json.load(open('configs/run/base.json')); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Run the full suite**

Run: `pytest -q`
Expected: PASS (all tests, including OSM client/pipeline/preview/cli).

- [ ] **Step 4: Commit**

```bash
git add configs/run/base.json
git commit -m "feat(osm): enable fetch_osm by default in base.json"
```

---

## Task 9: Manual smoke (outside sandbox)

**Files:** none (live verification — sandbox rate-limits OSM API on repeated calls).

- [ ] **Step 1: Run a single location end-to-end**

```bash
python -m satmap_dataset.cli run-all-location-json \
  --locations-dir configs/run/locations \
  --base-json configs/run/base.json
```

Expected: orthophoto artifacts produced, then for each location `osm_<slug>/osm_manifest.json` plus `year_<YYYY>_<cat>.tif` (burn 255) and `viz_<YYYY>.png`. OSM masks differ across years (historical filter).

- [ ] **Step 2: Confirm OSM isolation**

Temporarily point `overpass`/network off (or run an AOI where OSM API 400s without recovery) and confirm orthophoto artifacts still complete and only an OSM failure is listed.

- [ ] **Step 3: Confirm `--no-osm`**

```bash
python -m satmap_dataset.cli run-all-location-json --no-osm \
  --locations-dir configs/run/locations --base-json configs/run/base.json
```

Expected: no `osm_<slug>/` outputs; orthophoto artifacts unchanged.

---

## Self-Review

**Spec coverage:**
- Orchestration placement (isolated step after run_all.run) → Task 7. ✓
- Backend now = osm_api (`/map` + timestamp), future = overpass seam → Tasks 2–4 (`backend` branch, `NotImplementedError`). ✓
- Adaptive quadrant split on node-limit → Task 3 (`fetch_osm_xml`/`_quadrants`). ✓
- Download once per location, filter per (year, category) → Task 4 (`fetch_and_parse` once, `features_for` in loop). ✓
- `fetch_osm` default True + `--osm/--no-osm` → Tasks 7, 8. ✓
- Reuse-existing-raster, zero-features→null, burn 255 → Task 4 (preserved) + already-committed burn=255. ✓
- Per-year preview PNG, `osm_render_preview` → Tasks 1, 5. ✓
- `bbox_epsg2180_to_wgs84` moved to osm_api_client → Task 2. ✓
- Tests: client, pipeline backend branch, config validator, cli gating, preview → Tasks 2,3,4,5,6,7. ✓

**Placeholder scan:** No TBD/TODO/placeholder snippets. (Task 6 was rewritten to a no-code-change guard test once it was confirmed `_apply_location_paths_policy` already injects `render_root`.)

**Type consistency:** `Way(tags, coords, ts, ver)`, `parse_ways(list[bytes]) -> list[Way]`, `existed_at(Way, str) -> bool`, `features_for(list[Way], str, str) -> dict`, `fetch_and_parse(bbox, *, timeout, retry_policy) -> list[Way]`, `render_year_preview(config, year, year_asset)` — used identically across Tasks 2–7. Pipeline mock seam (`osm_api_client.fetch_and_parse`, `osm_api_client.features_for`, `_maybe_render_preview`) matches Task 4 implementation. ✓
