# ISOK Elevation (NMT/NMPT) Download Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Polish ISOK 1 m elevation download (NMT/DTM + NMPT/DSM) to the geoportal side of `satmap_dataset` via WCS GetCoverage, producing native-1 m and ortho-render-aligned float32 EPSG:2180 GeoTIFFs.

**Architecture:** A new `geoportal/wcs_client.py` builds coverage ids/endpoints, tiles the AOI, and fetches GeoTIFF bytes through the existing `geoportal/http.py` retry stack. A new `pipeline/dem.py` orchestrates fetch → merge (GDAL) → optional align-to-render-grid (GDAL) → `dem_manifest.json`. A new `DemConfig`/`DemManifest` pair carries the contract, `GeoportalProvider.dem()` delegates to the stage, and four CLI commands (`dem`, `dem-json`, `dem-location-json`, `dem-all-location-json`) mirror the ortho command flavors.

**Tech Stack:** Python ≥3.10, Pydantic v2, httpx (async), GDAL CLI (`gdalbuildvrt`/`gdal_translate`/`gdalwarp`, shelled out as `render.py` already does), tifffile + numpy (raster dims / emptiness check), Typer CLI, pytest.

**Reference spec:** `docs/superpowers/specs/2026-05-30-geoportal-elevation-nmt-nmpt-design.md`

**Deviation from spec (intentional):** GDAL embeds CRS + geotransform directly into the output GeoTIFFs, so we do **not** write separate `.prj` sidecars for DEM assets (unlike the pyvips render path, which needs them). Task 9 amends the spec line accordingly.

---

## File Structure

- Create `src/satmap_dataset/geoportal/wcs_client.py` — coverage-id/endpoint mapping, bbox tiling, async `get_coverage`.
- Create `src/satmap_dataset/pipeline/dem.py` — `run(DemConfig) -> (exit_code, Path)` orchestration + GDAL merge/align seams.
- Modify `src/satmap_dataset/config.py` — add `DemConfig`.
- Modify `src/satmap_dataset/models.py` — add `DemProductAsset`, `DemManifest`.
- Modify `src/satmap_dataset/providers/geoportal.py` — add `GeoportalProvider.dem()`.
- Modify `src/satmap_dataset/cli.py` — add 4 commands + `_build_dem_config_from_base_and_location`, extend `_apply_location_paths_policy`.
- Modify `scripts/manage_location_roots.py` — add `dem` root kind.
- Modify `.gitignore` — ignore `dem_*/`.
- Modify `justfile` — add `dem-location-json` / `dem-all-json` tasks.
- Modify the spec doc — amend the `.prj` sidecar line.
- Tests: `tests/test_wcs_client.py`, `tests/test_dem_models.py`, `tests/test_dem_config.py`, `tests/test_dem_pipeline.py`, `tests/test_dem_cli.py`.

---

## Task 1: `DemConfig` in config.py

**Files:**
- Modify: `src/satmap_dataset/config.py`
- Test: `tests/test_dem_config.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dem_config.py`:

```python
import pytest
from pydantic import ValidationError

from satmap_dataset.config import DemConfig


def test_defaults_are_both_products_evrf2007():
    cfg = DemConfig(bbox="210300,521900,210500,522100")
    assert cfg.products == ["nmt", "nmpt"]
    assert cfg.vertical_datum == "evrf2007"
    assert cfg.srs == "EPSG:2180"
    assert cfg.align_to_render is True
    assert cfg.max_request_px == 2048


def test_products_normalized_and_validated():
    cfg = DemConfig(bbox="0,0,10,10", products=["NMT"])
    assert cfg.products == ["nmt"]
    with pytest.raises(ValidationError):
        DemConfig(bbox="0,0,10,10", products=["foo"])
    with pytest.raises(ValidationError):
        DemConfig(bbox="0,0,10,10", products=[])


def test_vertical_datum_enum():
    assert DemConfig(bbox="0,0,10,10", vertical_datum="kron86").vertical_datum == "kron86"
    with pytest.raises(ValidationError):
        DemConfig(bbox="0,0,10,10", vertical_datum="nonsense")


def test_bbox_order_validated():
    with pytest.raises(ValidationError):
        DemConfig(bbox="10,10,0,0")


def test_sleep_and_paired_target_dims():
    with pytest.raises(ValidationError):
        DemConfig(bbox="0,0,10,10", sleep_min=2.0, sleep_max=1.0)
    with pytest.raises(ValidationError):
        DemConfig(bbox="0,0,10,10", target_width=100)  # height missing
    ok = DemConfig(bbox="0,0,10,10", target_width=100, target_height=200)
    assert ok.target_width == 100 and ok.target_height == 200
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_dem_config.py -q`
Expected: FAIL with `ImportError: cannot import name 'DemConfig'`.

- [ ] **Step 3: Implement `DemConfig`**

Append to `src/satmap_dataset/config.py` (after `RunConfig`):

```python
class DemConfig(BaseModel):
    bbox: str
    srs: str = "EPSG:2180"
    products: list[str] = Field(default_factory=lambda: ["nmt", "nmpt"])
    vertical_datum: str = "evrf2007"
    dem_root: Path = Path("dem")
    align_to_render: bool = True
    render_manifest: Path | None = None
    target_bbox: str | None = None
    target_width: int | None = Field(default=None, ge=1)
    target_height: int | None = Field(default=None, ge=1)
    px_per_meter: float = Field(default=1.0, gt=0.0)
    max_request_px: int = Field(default=2048, ge=1)
    overwrite: bool = False
    timeout: float = Field(default=120.0, gt=0.0)
    retries: int = Field(default=6, ge=1, le=20)
    retry_delay: float = Field(default=1.0, gt=0.0)
    sleep_min: float = Field(default=0.6, ge=0.0)
    sleep_max: float = Field(default=2.2, ge=0.0)
    location_name: str | None = None
    output_json: Path = Path("dem/dem_manifest.json")
    provider: str = PROVIDER_GEOPORTAL
    provider_options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value: str) -> str:
        return _validate_bbox(value)

    @field_validator("target_bbox")
    @classmethod
    def validate_target_bbox(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return _validate_bbox(value)

    @field_validator("products")
    @classmethod
    def validate_products(cls, value: list[str]) -> list[str]:
        allowed = {"nmt", "nmpt"}
        normalized = [str(item).strip().lower() for item in value]
        if not normalized:
            raise ValueError("products must not be empty")
        bad = [item for item in normalized if item not in allowed]
        if bad:
            raise ValueError(f"products must be a subset of {sorted(allowed)}; got {bad}")
        # preserve order, drop duplicates
        seen: list[str] = []
        for item in normalized:
            if item not in seen:
                seen.append(item)
        return seen

    @field_validator("vertical_datum")
    @classmethod
    def validate_vertical_datum(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        allowed = {"evrf2007", "kron86"}
        if normalized not in allowed:
            raise ValueError(f"vertical_datum must be one of {sorted(allowed)}")
        return normalized

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        if value not in ALLOWED_PROVIDERS:
            raise ValueError(f"provider must be one of {sorted(ALLOWED_PROVIDERS)}")
        return value

    @model_validator(mode="after")
    def validate_invariants(self) -> "DemConfig":
        if self.sleep_max < self.sleep_min:
            raise ValueError("sleep_max must be >= sleep_min")
        if (self.target_width is None) != (self.target_height is None):
            raise ValueError("target_width and target_height must be set together")
        return self
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_dem_config.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/config.py tests/test_dem_config.py
git commit -m "feat(dem): add DemConfig for ISOK elevation download"
```

---

## Task 2: `DemManifest` / `DemProductAsset` in models.py

**Files:**
- Modify: `src/satmap_dataset/models.py`
- Test: `tests/test_dem_models.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_dem_models.py`:

```python
from satmap_dataset.models import DemManifest, DemProductAsset


def test_dem_manifest_round_trip():
    manifest = DemManifest(
        bbox="0,0,10,10",
        srs="EPSG:2180",
        vertical_datum="evrf2007",
        products=[
            DemProductAsset(
                product="nmt",
                coverage_id="DTM_PL-EVRF2007-NH_TIFF",
                endpoint="https://example/wcs",
                native_path="dem_x/native/nmt_evrf2007.tif",
                native_width=10,
                native_height=10,
                tile_count=1,
                passed=True,
            )
        ],
        passed=True,
    )
    blob = manifest.model_dump_json()
    restored = DemManifest.model_validate_json(blob)
    assert restored.kind == "dem_manifest"
    assert restored.stage == "dem"
    assert restored.products[0].product == "nmt"
    assert restored.products[0].coverage_id == "DTM_PL-EVRF2007-NH_TIFF"
    assert restored.passed is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_dem_models.py -q`
Expected: FAIL with `ImportError: cannot import name 'DemManifest'`.

- [ ] **Step 3: Implement the models**

Append to `src/satmap_dataset/models.py` (end of file):

```python
class DemProductAsset(BaseModel):
    product: Literal["nmt", "nmpt"]
    coverage_id: str
    endpoint: str
    native_path: str | None = None
    native_width: int | None = None
    native_height: int | None = None
    aligned_path: str | None = None
    aligned_width: int | None = None
    aligned_height: int | None = None
    tile_count: int = Field(default=0, ge=0)
    nodata: float | None = None
    passed: bool = False
    errors: list[str] = Field(default_factory=list)


class DemManifest(BaseModel):
    kind: Literal["dem_manifest"] = "dem_manifest"
    stage: Literal["dem"] = "dem"
    generated_at: datetime = Field(default_factory=_utc_now)
    provider: str = "geoportal"
    bbox: str
    srs: str
    vertical_datum: str
    products: list[DemProductAsset] = Field(default_factory=list)
    align_to_render: bool = True
    passed: bool = False
    notes: str | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    run_parameters: dict[str, Any] = Field(default_factory=dict)
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_dem_models.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/models.py tests/test_dem_models.py
git commit -m "feat(dem): add DemManifest + DemProductAsset schema"
```

---

## Task 3: WCS client — coverage-id + endpoint mapping

**Files:**
- Create: `src/satmap_dataset/geoportal/wcs_client.py`
- Test: `tests/test_wcs_client.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_wcs_client.py`:

```python
import pytest

from satmap_dataset.geoportal import wcs_client


def test_coverage_id_all_combinations():
    assert wcs_client.coverage_id("nmt", "evrf2007") == "DTM_PL-EVRF2007-NH_TIFF"
    assert wcs_client.coverage_id("nmt", "kron86") == "DTM_PL-KRON86-NH_TIFF"
    assert wcs_client.coverage_id("nmpt", "evrf2007") == "DSM_PL-EVRF2007-NH_TIFF"
    assert wcs_client.coverage_id("nmpt", "kron86") == "DSM_PL-KRON86-NH_TIFF"


def test_coverage_id_rejects_unknown():
    with pytest.raises(ValueError):
        wcs_client.coverage_id("foo", "evrf2007")
    with pytest.raises(ValueError):
        wcs_client.coverage_id("nmt", "wgs84")


def test_endpoint_url_default_and_override():
    assert "NMT/GRID1/WCS" in wcs_client.endpoint_url("nmt")
    assert "NMPT/GRID1/WCS" in wcs_client.endpoint_url("nmpt")
    custom = {"endpoints": {"nmt": "https://example/custom"}}
    assert wcs_client.endpoint_url("nmt", custom) == "https://example/custom"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_wcs_client.py -q`
Expected: FAIL with `ModuleNotFoundError: ...wcs_client`.

- [ ] **Step 3: Implement the mapping (initial file)**

Create `src/satmap_dataset/geoportal/wcs_client.py`:

```python
from __future__ import annotations

import math
from typing import Any

from satmap_dataset.geoportal.http import RetryPolicy, request_with_retry

DEFAULT_ENDPOINTS = {
    "nmt": "https://mapy.geoportal.gov.pl/wss/service/PZGIK/NMT/GRID1/WCS/DigitalTerrainModelFormatTIFF",
    "nmpt": "https://mapy.geoportal.gov.pl/wss/service/PZGIK/NMPT/GRID1/WCS/DigitalSurfaceModelFormatTIFF",
}
_PRODUCT_PREFIX = {"nmt": "DTM", "nmpt": "DSM"}
_DATUM_TOKEN = {"evrf2007": "PL-EVRF2007-NH", "kron86": "PL-KRON86-NH"}


def endpoint_url(product: str, options: dict[str, Any] | None = None) -> str:
    options = options or {}
    overrides = options.get("endpoints") or {}
    if product in overrides:
        return str(overrides[product])
    if product not in DEFAULT_ENDPOINTS:
        raise ValueError(f"Unknown product {product!r}; expected one of {sorted(DEFAULT_ENDPOINTS)}")
    return DEFAULT_ENDPOINTS[product]


def coverage_id(product: str, datum: str, options: dict[str, Any] | None = None) -> str:
    options = options or {}
    template = str(options.get("coverage_id_template", "{prefix}_{datum}_TIFF"))
    if product not in _PRODUCT_PREFIX:
        raise ValueError(f"Unknown product {product!r}; expected one of {sorted(_PRODUCT_PREFIX)}")
    if datum not in _DATUM_TOKEN:
        raise ValueError(f"Unknown datum {datum!r}; expected one of {sorted(_DATUM_TOKEN)}")
    return template.format(prefix=_PRODUCT_PREFIX[product], datum=_DATUM_TOKEN[datum])
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_wcs_client.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/geoportal/wcs_client.py tests/test_wcs_client.py
git commit -m "feat(dem): WCS coverage-id and endpoint resolution"
```

---

## Task 4: WCS client — `split_bbox` tiling

**Files:**
- Modify: `src/satmap_dataset/geoportal/wcs_client.py`
- Test: `tests/test_wcs_client.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_wcs_client.py`:

```python
def test_split_bbox_single_tile_when_within_cap():
    tiles = wcs_client.split_bbox((0.0, 0.0, 100.0, 100.0), max_request_px=2048, gsd_m=1.0)
    assert tiles == [(0.0, 0.0, 100.0, 100.0)]


def test_split_bbox_tiles_and_covers_exactly():
    bbox = (0.0, 0.0, 250.0, 150.0)
    tiles = wcs_client.split_bbox(bbox, max_request_px=100, gsd_m=1.0)
    # 250m/100m -> 3 cols, 150m/100m -> 2 rows
    assert len(tiles) == 6
    # union of tiles equals original bbox extent
    assert min(t[0] for t in tiles) == 0.0
    assert min(t[1] for t in tiles) == 0.0
    assert max(t[2] for t in tiles) == 250.0
    assert max(t[3] for t in tiles) == 150.0
    # no tile exceeds the cap span
    for x0, y0, x1, y1 in tiles:
        assert x1 - x0 <= 100.0 + 1e-9
        assert y1 - y0 <= 100.0 + 1e-9


def test_split_bbox_rejects_bad_cap():
    with pytest.raises(ValueError):
        wcs_client.split_bbox((0, 0, 10, 10), max_request_px=0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_wcs_client.py -k split_bbox -q`
Expected: FAIL with `AttributeError: module ... has no attribute 'split_bbox'`.

- [ ] **Step 3: Implement `split_bbox`**

Append to `src/satmap_dataset/geoportal/wcs_client.py`:

```python
def split_bbox(
    bbox: tuple[float, float, float, float],
    max_request_px: int,
    gsd_m: float = 1.0,
) -> list[tuple[float, float, float, float]]:
    """Split an AOI bbox into non-overlapping sub-bboxes, each at most
    ``max_request_px`` pixels per side at the given ground sample distance."""
    if max_request_px < 1:
        raise ValueError("max_request_px must be >= 1")
    if gsd_m <= 0:
        raise ValueError("gsd_m must be > 0")
    xmin, ymin, xmax, ymax = bbox
    span_m = max_request_px * gsd_m
    nx = max(1, math.ceil((xmax - xmin) / span_m))
    ny = max(1, math.ceil((ymax - ymin) / span_m))
    tiles: list[tuple[float, float, float, float]] = []
    for iy in range(ny):
        y0 = ymin + iy * span_m
        y1 = min(ymax, y0 + span_m)
        for ix in range(nx):
            x0 = xmin + ix * span_m
            x1 = min(xmax, x0 + span_m)
            tiles.append((x0, y0, x1, y1))
    return tiles
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_wcs_client.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/geoportal/wcs_client.py tests/test_wcs_client.py
git commit -m "feat(dem): WCS AOI bbox tiling"
```

---

## Task 5: WCS client — async `get_coverage`

**Files:**
- Modify: `src/satmap_dataset/geoportal/wcs_client.py`
- Test: `tests/test_wcs_client.py`

- [ ] **Step 1: Add failing test (mocked HTTP)**

Append to `tests/test_wcs_client.py`:

```python
import asyncio


def test_get_coverage_builds_wcs_params(monkeypatch):
    captured = {}

    class _FakeResponse:
        content = b"GEOTIFF-BYTES"

    async def _fake_request(method, url, *, params, timeout, retry_policy, client=None):
        captured["method"] = method
        captured["url"] = url
        captured["params"] = params
        return _FakeResponse()

    monkeypatch.setattr(wcs_client, "request_with_retry", _fake_request)

    data = asyncio.run(
        wcs_client.get_coverage(
            "https://example/wcs",
            "DTM_PL-EVRF2007-NH_TIFF",
            (10.0, 20.0, 30.0, 40.0),
            "EPSG:2180",
        )
    )
    assert data == b"GEOTIFF-BYTES"
    params = captured["params"]
    assert params["SERVICE"] == "WCS"
    assert params["REQUEST"] == "GetCoverage"
    assert params["COVERAGEID"] == "DTM_PL-EVRF2007-NH_TIFF"
    assert params["VERSION"] == "2.0.1"
    assert params["SUBSET"] == ["x(10.0,30.0)", "y(20.0,40.0)"]
    assert params["SUBSETTINGCRS"].endswith("/2180")
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_wcs_client.py -k get_coverage -q`
Expected: FAIL with `AttributeError: ... 'get_coverage'`.

- [ ] **Step 3: Implement `get_coverage`**

Append to `src/satmap_dataset/geoportal/wcs_client.py`:

```python
async def get_coverage(
    endpoint: str,
    coverage_id_value: str,
    sub_bbox: tuple[float, float, float, float],
    srs: str,
    *,
    options: dict[str, Any] | None = None,
    timeout: float = 120.0,
    retry_policy: RetryPolicy | None = None,
    client: Any | None = None,
) -> bytes:
    """Issue a WCS 2.0.1 GetCoverage and return the GeoTIFF bytes."""
    options = options or {}
    axis_x = str(options.get("axis_label_x", "x"))
    axis_y = str(options.get("axis_label_y", "y"))
    fmt = str(options.get("format", "image/tiff"))
    epsg = srs.split(":")[-1]
    subsetting_crs = str(
        options.get("subsetting_crs", f"http://www.opengis.net/def/crs/EPSG/0/{epsg}")
    )
    xmin, ymin, xmax, ymax = sub_bbox
    params: dict[str, Any] = {
        "SERVICE": "WCS",
        "VERSION": str(options.get("wcs_version", "2.0.1")),
        "REQUEST": "GetCoverage",
        "COVERAGEID": coverage_id_value,
        "FORMAT": fmt,
        "SUBSETTINGCRS": subsetting_crs,
        "SUBSET": [f"{axis_x}({xmin},{xmax})", f"{axis_y}({ymin},{ymax})"],
    }
    response = await request_with_retry(
        "GET",
        endpoint,
        params=params,
        timeout=timeout,
        retry_policy=retry_policy,
        client=client,
    )
    return response.content
```

Note: httpx encodes the list-valued `SUBSET` as two repeated `SUBSET=` query params, which is what WCS 2.0.1 expects.

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_wcs_client.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/geoportal/wcs_client.py tests/test_wcs_client.py
git commit -m "feat(dem): async WCS GetCoverage fetch"
```

---

## Task 6: `pipeline/dem.py` — orchestration + GDAL seams

**Files:**
- Create: `src/satmap_dataset/pipeline/dem.py`
- Test: `tests/test_dem_pipeline.py`

- [ ] **Step 1: Write the failing tests (fetch + GDAL fully mocked)**

Create `tests/test_dem_pipeline.py`:

```python
import json
from pathlib import Path

from satmap_dataset.config import DemConfig
from satmap_dataset.models import DemManifest
from satmap_dataset.pipeline import dem


def _patch_seams(monkeypatch, *, empty=False):
    """Replace network + GDAL + raster IO with deterministic fakes."""

    async def _fake_fetch(config, product, dest_dir, *, retry_policy):
        tile = Path(dest_dir) / f"{product}_0000.tif"
        tile.write_bytes(b"TILE")
        return [tile]

    def _fake_merge(tiles, out_path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"NATIVE")

    def _fake_align(native, out_path, **kwargs):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"ALIGNED")

    monkeypatch.setattr(dem, "_fetch_tiles_for_product", _fake_fetch)
    monkeypatch.setattr(dem, "_merge_tiles", _fake_merge)
    monkeypatch.setattr(dem, "_align_to_grid", _fake_align)
    monkeypatch.setattr(dem, "_coverage_is_empty", lambda path: empty)
    monkeypatch.setattr(dem, "_raster_dims", lambda path: (10, 10))


def test_run_writes_native_and_aligned_for_both_products(tmp_path, monkeypatch):
    _patch_seams(monkeypatch)
    cfg = DemConfig(
        bbox="0,0,100,100",
        dem_root=tmp_path / "dem_x",
        output_json=tmp_path / "dem_x" / "dem_manifest.json",
        target_bbox="0,0,100,100",
        target_width=100,
        target_height=100,
    )
    code, path = dem.run(cfg)
    assert code == 0
    manifest = DemManifest.model_validate_json(Path(path).read_text())
    assert manifest.passed is True
    assert {p.product for p in manifest.products} == {"nmt", "nmpt"}
    for p in manifest.products:
        assert p.passed is True
        assert Path(p.native_path).exists()
        assert Path(p.aligned_path).exists()
        assert "native" in p.native_path and "aligned" in p.aligned_path


def test_run_no_align_when_disabled(tmp_path, monkeypatch):
    _patch_seams(monkeypatch)
    cfg = DemConfig(
        bbox="0,0,100,100",
        products=["nmt"],
        align_to_render=False,
        dem_root=tmp_path / "dem_x",
        output_json=tmp_path / "dem_x" / "dem_manifest.json",
    )
    code, path = dem.run(cfg)
    assert code == 0
    manifest = DemManifest.model_validate_json(Path(path).read_text())
    assert manifest.products[0].aligned_path is None


def test_run_fails_on_empty_coverage(tmp_path, monkeypatch):
    _patch_seams(monkeypatch, empty=True)
    cfg = DemConfig(
        bbox="0,0,100,100",
        products=["nmt"],
        align_to_render=False,
        dem_root=tmp_path / "dem_x",
        output_json=tmp_path / "dem_x" / "dem_manifest.json",
    )
    code, path = dem.run(cfg)
    assert code == 1
    manifest = DemManifest.model_validate_json(Path(path).read_text())
    assert manifest.passed is False
    assert manifest.products[0].passed is False


def test_resolve_align_grid_prefers_render_manifest(tmp_path):
    render_manifest = tmp_path / "dataset_manifest_render.json"
    render_manifest.write_text(json.dumps({
        "kind": "dataset_manifest", "stage": "render",
        "target_bbox": "5,5,55,55", "target_width": 500, "target_height": 500,
    }))
    cfg = DemConfig(
        bbox="0,0,100,100",
        render_manifest=render_manifest,
        target_bbox="0,0,100,100", target_width=100, target_height=100,
    )
    grid = dem._resolve_align_grid(cfg)
    assert grid == ((5.0, 5.0, 55.0, 55.0), 500, 500)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_dem_pipeline.py -q`
Expected: FAIL with `ModuleNotFoundError: ...pipeline.dem`.

- [ ] **Step 3: Implement `pipeline/dem.py`**

Create `src/satmap_dataset/pipeline/dem.py`:

```python
from __future__ import annotations

import asyncio
import logging
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

from satmap_dataset.config import DemConfig
from satmap_dataset.geoportal import wcs_client
from satmap_dataset.geoportal.http import RetryPolicy
from satmap_dataset.models import DemManifest, DemProductAsset

logger = logging.getLogger("satmap_dataset.dem")


def _parse_bbox(value: str) -> tuple[float, float, float, float]:
    parts = [float(p.strip()) for p in value.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must have format xmin,ymin,xmax,ymax")
    return (parts[0], parts[1], parts[2], parts[3])


def _tool_path(name: str) -> str | None:
    return shutil.which(name)


async def _fetch_tiles_for_product(
    config: DemConfig, product: str, dest_dir: Path, *, retry_policy: RetryPolicy
) -> list[Path]:
    options = dict(config.provider_options)
    endpoint = wcs_client.endpoint_url(product, options)
    cov = wcs_client.coverage_id(product, config.vertical_datum, options)
    sub_bboxes = wcs_client.split_bbox(_parse_bbox(config.bbox), config.max_request_px)
    tiles: list[Path] = []
    for i, sub in enumerate(sub_bboxes):
        if config.sleep_max > 0:
            await asyncio.sleep(random.uniform(config.sleep_min, config.sleep_max))
        data = await wcs_client.get_coverage(
            endpoint, cov, sub, config.srs,
            options=options, timeout=config.timeout, retry_policy=retry_policy,
        )
        out = dest_dir / f"{product}_{i:04d}.tif"
        out.write_bytes(data)
        tiles.append(out)
    return tiles


def _merge_tiles(tiles: list[Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(tiles) == 1:
        shutil.copyfile(tiles[0], out_path)
        return
    gdalbuildvrt = _tool_path("gdalbuildvrt")
    gdal_translate = _tool_path("gdal_translate")
    if not gdalbuildvrt or not gdal_translate:
        raise RuntimeError(
            "Merging tiled WCS output requires the GDAL CLI (gdalbuildvrt, "
            "gdal_translate). Install GDAL or reduce the AOI below max_request_px."
        )
    vrt_path = out_path.with_suffix(".vrt")
    subprocess.run([gdalbuildvrt, str(vrt_path), *[str(t) for t in tiles]], check=True)
    subprocess.run(
        [gdal_translate, "-co", "COMPRESS=DEFLATE", str(vrt_path), str(out_path)],
        check=True,
    )
    vrt_path.unlink(missing_ok=True)


def _align_to_grid(
    native: Path, out_path: Path, *,
    target_bbox: tuple[float, float, float, float],
    target_width: int, target_height: int, srs: str, resample: str = "bilinear",
) -> None:
    gdalwarp = _tool_path("gdalwarp")
    if not gdalwarp:
        raise RuntimeError(
            "Aligning the DEM to the render grid requires the GDAL CLI (gdalwarp). "
            "Install GDAL or set align_to_render=false."
        )
    xmin, ymin, xmax, ymax = target_bbox
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            gdalwarp, "-t_srs", srs,
            "-te", str(xmin), str(ymin), str(xmax), str(ymax),
            "-ts", str(target_width), str(target_height),
            "-r", resample, "-co", "COMPRESS=DEFLATE", "-overwrite",
            str(native), str(out_path),
        ],
        check=True,
    )


def _raster_dims(path: Path) -> tuple[int | None, int | None]:
    try:
        import tifffile

        with tifffile.TiffFile(str(path)) as tif:
            page = tif.pages[0]
            return int(page.imagewidth), int(page.imagelength)
    except Exception:  # best-effort; dims are informational
        return (None, None)


def _coverage_is_empty(path: Path) -> bool:
    try:
        import numpy as np
        import tifffile

        arr = np.asarray(tifffile.imread(str(path)), dtype="float64")
    except Exception:  # cannot read -> don't block
        return False
    if arr.size == 0:
        return True
    finite = np.isfinite(arr)
    if not finite.any():
        return True
    return False


def _resolve_align_grid(
    config: DemConfig,
) -> tuple[tuple[float, float, float, float], int, int]:
    if config.render_manifest and Path(config.render_manifest).exists():
        from satmap_dataset.models import DatasetManifest

        manifest = DatasetManifest.model_validate_json(
            Path(config.render_manifest).read_text(encoding="utf-8")
        )
        if manifest.target_bbox and manifest.target_width and manifest.target_height:
            return (
                _parse_bbox(manifest.target_bbox),
                int(manifest.target_width),
                int(manifest.target_height),
            )
    bbox = _parse_bbox(config.target_bbox or config.bbox)
    if config.target_width and config.target_height:
        return (bbox, int(config.target_width), int(config.target_height))
    xmin, ymin, xmax, ymax = bbox
    width = max(1, round((xmax - xmin) * config.px_per_meter))
    height = max(1, round((ymax - ymin) * config.px_per_meter))
    return (bbox, width, height)


async def _run_async(config: DemConfig) -> tuple[int, Path]:
    retry_policy = RetryPolicy(max_attempts=config.retries)
    grid = _resolve_align_grid(config) if config.align_to_render else None
    resample = str(config.provider_options.get("resample", "bilinear"))
    product_assets: list[DemProductAsset] = []
    errors: list[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for product in config.products:
            options = dict(config.provider_options)
            asset = DemProductAsset(
                product=product,
                coverage_id=wcs_client.coverage_id(product, config.vertical_datum, options),
                endpoint=wcs_client.endpoint_url(product, options),
            )
            native_path = config.dem_root / "native" / f"{product}_{config.vertical_datum}.tif"
            try:
                if not (native_path.exists() and not config.overwrite):
                    tiles = await _fetch_tiles_for_product(
                        config, product, tmp_dir, retry_policy=retry_policy
                    )
                    asset.tile_count = len(tiles)
                    _merge_tiles(tiles, native_path)
                if _coverage_is_empty(native_path):
                    asset.errors.append("coverage empty / nodata-only for AOI")
                    errors.append(f"{product}: empty coverage")
                else:
                    asset.native_path = str(native_path)
                    asset.native_width, asset.native_height = _raster_dims(native_path)
                    if grid is not None:
                        aligned_path = (
                            config.dem_root / "aligned" / f"{product}_{config.vertical_datum}.tif"
                        )
                        target_bbox, gw, gh = grid
                        _align_to_grid(
                            native_path, aligned_path,
                            target_bbox=target_bbox, target_width=gw, target_height=gh,
                            srs=config.srs, resample=resample,
                        )
                        asset.aligned_path = str(aligned_path)
                        asset.aligned_width, asset.aligned_height = gw, gh
                    asset.passed = True
            except Exception as exc:  # noqa: BLE001 - record and continue per-product
                asset.errors.append(str(exc))
                errors.append(f"{product}: {exc}")
            product_assets.append(asset)

    passed = bool(product_assets) and all(a.passed for a in product_assets)
    manifest = DemManifest(
        provider="geoportal",
        bbox=config.bbox,
        srs=config.srs,
        vertical_datum=config.vertical_datum,
        products=product_assets,
        align_to_render=config.align_to_render,
        passed=passed,
        notes="WCS GRID1 serves a current-best 1 m composite; not year-aware.",
        errors=errors,
        run_parameters=config.model_dump(mode="json"),
    )
    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    config.output_json.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    logger.info(
        "DEM run: products=%s passed=%s errors=%s",
        [a.product for a in product_assets], passed, len(errors),
    )
    return (0 if passed else 1), config.output_json


def run(config: DemConfig) -> tuple[int, Path]:
    return asyncio.run(_run_async(config))
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_dem_pipeline.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/pipeline/dem.py tests/test_dem_pipeline.py
git commit -m "feat(dem): WCS elevation pipeline stage (fetch/merge/align)"
```

---

## Task 7: `GeoportalProvider.dem()` delegator

**Files:**
- Modify: `src/satmap_dataset/providers/geoportal.py`
- Test: `tests/test_dem_pipeline.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_dem_pipeline.py`:

```python
def test_geoportal_provider_dem_delegates(tmp_path, monkeypatch):
    from satmap_dataset.providers.geoportal import GeoportalProvider

    called = {}

    def _fake_run(config):
        called["config"] = config
        return (0, tmp_path / "dem_manifest.json")

    monkeypatch.setattr("satmap_dataset.pipeline.dem.run", _fake_run)
    cfg = DemConfig(bbox="0,0,10,10", dem_root=tmp_path / "dem_x")
    code, path = GeoportalProvider().dem(cfg)
    assert code == 0
    assert called["config"] is cfg
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_dem_pipeline.py -k provider_dem -q`
Expected: FAIL with `AttributeError: 'GeoportalProvider' object has no attribute 'dem'`.

- [ ] **Step 3: Implement the method**

Replace the body of `src/satmap_dataset/providers/geoportal.py` with:

```python
from __future__ import annotations

from pathlib import Path

from satmap_dataset.config import DemConfig, DownloadConfig, IndexConfig
from satmap_dataset.pipeline import dem, downloader, index_builder
from satmap_dataset.providers.base import Provider


class GeoportalProvider(Provider):
    name = "geoportal"
    default_target_srs = "EPSG:2180"

    def index(self, config: IndexConfig) -> tuple[int, Path]:
        return index_builder.run(config)

    def download(self, config: DownloadConfig) -> tuple[int, Path]:
        return downloader.run(config)

    def dem(self, config: DemConfig) -> tuple[int, Path]:
        return dem.run(config)
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_dem_pipeline.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/providers/geoportal.py tests/test_dem_pipeline.py
git commit -m "feat(dem): GeoportalProvider.dem delegator"
```

---

## Task 8: CLI commands + base/location builder

**Files:**
- Modify: `src/satmap_dataset/cli.py`
- Test: `tests/test_dem_cli.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dem_cli.py`:

```python
import json
from pathlib import Path

from typer.testing import CliRunner

from satmap_dataset.cli import (
    app,
    _apply_location_paths_policy,
    _build_dem_config_from_base_and_location,
)

runner = CliRunner()


def test_apply_location_paths_policy_adds_dem_root(tmp_path):
    out = _apply_location_paths_policy({"location_name": "Poznań"}, tmp_path)
    assert out["dem_root"].endswith("dem_poznan")


def test_build_dem_config_from_base_and_location(tmp_path):
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"vertical_datum": "kron86", "max_request_px": 1024}))
    loc = tmp_path / "location_poznan.json"
    loc.write_text(json.dumps({
        "location_name": "Poznań", "center_lat": 52.4, "center_lon": 16.9, "square_km": 4.0,
    }))
    cfg = _build_dem_config_from_base_and_location(base_json=base, location_json=loc)
    assert cfg.vertical_datum == "kron86"
    assert cfg.max_request_px == 1024
    assert str(cfg.dem_root).endswith("dem_poznan")
    assert str(cfg.output_json).endswith("dem_manifest.json")
    assert cfg.bbox  # center resolved to a concrete bbox


def test_dem_json_command_invokes_pipeline(tmp_path, monkeypatch):
    captured = {}

    def _fake_run(config):
        captured["config"] = config
        out = tmp_path / "dem_manifest.json"
        out.write_text("{}")
        return (0, out)

    monkeypatch.setattr("satmap_dataset.pipeline.dem.run", _fake_run)
    params = tmp_path / "params.json"
    params.write_text(json.dumps({
        "bbox": "210300,521900,210500,522100", "products": ["nmt"], "align_to_render": False,
        "dem_root": str(tmp_path / "dem_x"), "output_json": str(tmp_path / "dem_manifest.json"),
    }))
    result = runner.invoke(app, ["dem-json", str(params)])
    assert result.exit_code == 0
    assert captured["config"].products == ["nmt"]
    # last stdout line is the artifact path
    assert result.stdout.strip().splitlines()[-1].endswith("dem_manifest.json")


def test_dem_json_command_bad_config_exit_2(tmp_path):
    params = tmp_path / "params.json"
    params.write_text(json.dumps({"bbox": "10,10,0,0"}))  # invalid order
    result = runner.invoke(app, ["dem-json", str(params)])
    assert result.exit_code == 2
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_dem_cli.py -q`
Expected: FAIL with `ImportError: cannot import name '_build_dem_config_from_base_and_location'`.

- [ ] **Step 3a: Extend `_apply_location_paths_policy`**

In `src/satmap_dataset/cli.py`, find (around line 244):

```python
    normalized.setdefault("download_root", str(repo_root / f"downloads_{slug}"))
    normalized.setdefault("render_root", str(repo_root / f"rendered_{slug}"))
    normalized.setdefault("artifacts_dir", str(repo_root / f"artifacts_{slug}"))
    return normalized
```

Replace with (add the `dem_root` line):

```python
    normalized.setdefault("download_root", str(repo_root / f"downloads_{slug}"))
    normalized.setdefault("render_root", str(repo_root / f"rendered_{slug}"))
    normalized.setdefault("artifacts_dir", str(repo_root / f"artifacts_{slug}"))
    normalized.setdefault("dem_root", str(repo_root / f"dem_{slug}"))
    return normalized
```

(Extra `dem_root` key is harmless for the ortho configs — Pydantic ignores unknown fields.)

- [ ] **Step 3b: Add the import**

At the top of `cli.py`, find the config import line (it imports `RunConfig`, `IndexConfig`, etc.) and add `DemConfig` to it. For example if it reads:

```python
from satmap_dataset.config import (
    DownloadConfig,
    IndexConfig,
    RenderConfig,
    RunConfig,
    ValidateConfig,
)
```

add `DemConfig,` alphabetically:

```python
from satmap_dataset.config import (
    DemConfig,
    DownloadConfig,
    IndexConfig,
    RenderConfig,
    RunConfig,
    ValidateConfig,
)
```

If the import is a single line, append `, DemConfig` to it. Also ensure `dem` stage is importable; add near the other pipeline imports (e.g. `from satmap_dataset.pipeline import ... render, run_all`) — add `dem`:

```python
from satmap_dataset.pipeline import dem
```

(Place it with the existing `from satmap_dataset.pipeline import ...` imports.)

- [ ] **Step 3c: Add the builder helper**

In `cli.py`, after `_build_render_config_from_base_and_location` (search for `def _build_render_config_from_base_and_location`), add:

```python
def _build_dem_config_from_base_and_location(*, base_json: Path, location_json: Path) -> DemConfig:
    base_payload = _load_params_json_dict(base_json)
    location_payload = _load_params_json_dict(location_json)
    merged: dict[str, object] = dict(base_payload)
    merged.update(location_payload)
    repo_root = base_json.resolve().parents[2] if len(base_json.resolve().parents) >= 3 else Path.cwd().resolve()
    merged = _apply_location_paths_policy(merged, repo_root)
    merged = _resolve_json_center_bbox(merged, required=True)
    dem_root = Path(str(merged.get("dem_root", "dem")))
    merged.setdefault("output_json", str(dem_root / "dem_manifest.json"))
    artifacts_dir = merged.get("artifacts_dir")
    if artifacts_dir is not None and merged.get("align_to_render", True):
        merged.setdefault("render_manifest", str(Path(str(artifacts_dir)) / "dataset_manifest_render.json"))
    return DemConfig.model_validate(merged)
```

- [ ] **Step 3d: Add the four commands**

At the end of `cli.py` (after the last `@app.command(...)`), add:

```python
@app.command("dem-json")
def dem_json_command(
    params_json: Path = typer.Argument(
        ...,
        help="Path to JSON file with DemConfig fields. Supports center_lat/center_lon + square_km|area_km2.",
    ),
) -> None:
    try:
        payload = _load_params_json_dict(params_json)
        payload = _resolve_json_center_bbox(payload, required=True)
        config = DemConfig.model_validate(payload)
    except typer.BadParameter as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=2) from error
    except ValidationError as error:
        _print_validation_error(error)
        raise typer.Exit(code=2) from error

    exit_code, artifact_path = dem.run(config)
    _finish(exit_code, artifact_path)


@app.command("dem")
def dem_command(
    bbox: str = typer.Option(None, "--bbox", help="xmin,ymin,xmax,ymax in --srs."),
    srs: str = typer.Option("EPSG:2180", "--srs"),
    center_lat: float = typer.Option(None, "--center-lat"),
    center_lon: float = typer.Option(None, "--center-lon"),
    square_km: float = typer.Option(None, "--square-km"),
    products: str = typer.Option("nmt,nmpt", "--products", help="Comma-separated subset of nmt,nmpt."),
    vertical_datum: str = typer.Option("evrf2007", "--vertical-datum", help="evrf2007 or kron86."),
    dem_root: Path = typer.Option(Path("dem"), "--dem-root"),
    align_to_render: bool = typer.Option(True, "--align/--no-align"),
    render_manifest: Path = typer.Option(None, "--render-manifest"),
    max_request_px: int = typer.Option(2048, "--max-request-px"),
    overwrite: bool = typer.Option(False, "--overwrite"),
    output_json: Path = typer.Option(None, "--output-json"),
) -> None:
    try:
        payload: dict[str, object] = {
            "bbox": bbox,
            "srs": srs,
            "center_lat": center_lat,
            "center_lon": center_lon,
            "square_km": square_km,
            "products": [p.strip() for p in products.split(",") if p.strip()],
            "vertical_datum": vertical_datum,
            "dem_root": str(dem_root),
            "align_to_render": align_to_render,
            "max_request_px": max_request_px,
            "overwrite": overwrite,
        }
        if render_manifest is not None:
            payload["render_manifest"] = str(render_manifest)
        payload["output_json"] = str(output_json) if output_json is not None else str(dem_root / "dem_manifest.json")
        payload = _resolve_json_center_bbox(payload, required=True)
        config = DemConfig.model_validate(payload)
    except typer.BadParameter as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=2) from error
    except ValidationError as error:
        _print_validation_error(error)
        raise typer.Exit(code=2) from error

    exit_code, artifact_path = dem.run(config)
    _finish(exit_code, artifact_path)


@app.command("dem-location-json")
def dem_location_json_command(
    location_json: Path = typer.Argument(..., help="Path to location JSON (location_name, center_lat, center_lon)."),
    base_json: Path = typer.Option(
        Path("configs/run/base.json"), "--base-json",
        help="Path to base JSON with shared parameters.",
    ),
) -> None:
    try:
        config = _build_dem_config_from_base_and_location(base_json=base_json, location_json=location_json)
    except typer.BadParameter as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=2) from error
    except ValidationError as error:
        _print_validation_error(error)
        raise typer.Exit(code=2) from error

    exit_code, artifact_path = dem.run(config)
    _finish(exit_code, artifact_path)


@app.command("dem-all-location-json")
def dem_all_location_json_command(
    locations_dir: Path = typer.Option(
        Path("configs/run/locations"), "--locations-dir",
        help="Directory with location JSON files.",
    ),
    base_json: Path = typer.Option(
        Path("configs/run/base.json"), "--base-json",
        help="Path to base JSON with shared parameters.",
    ),
    continue_on_error: bool = typer.Option(
        False, "--continue-on-error/--no-continue-on-error",
        help="Continue with remaining locations when one fails.",
    ),
) -> None:
    location_files = _location_files_or_exit(locations_dir)
    failures: list[str] = []
    for location_json in location_files:
        console.print(f"[cyan]dem-all-location-json:[/cyan] {location_json}")
        try:
            config = _build_dem_config_from_base_and_location(
                base_json=base_json, location_json=location_json,
            )
        except typer.BadParameter as error:
            console.print(f"[red]{error}[/red]")
            failures.append(f"{location_json}: {error}")
            if not continue_on_error:
                raise typer.Exit(code=2) from error
            continue
        except ValidationError as error:
            _print_validation_error(error)
            failures.append(f"{location_json}: validation_error")
            if not continue_on_error:
                raise typer.Exit(code=2) from error
            continue

        exit_code, artifact_path = dem.run(config)
        console.print(str(artifact_path))
        if exit_code != 0:
            failures.append(f"{location_json}: exit={exit_code}")
            if not continue_on_error:
                raise typer.Exit(code=exit_code)

    if failures:
        console.print("[yellow]dem-all-location-json finished with failures:[/yellow]")
        for entry in failures:
            console.print(f"- {entry}")
        raise typer.Exit(code=1)
    raise typer.Exit(code=0)
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_dem_cli.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/cli.py tests/test_dem_cli.py
git commit -m "feat(dem): CLI commands dem / dem-json / dem-location-json / dem-all-location-json"
```

---

## Task 9: roots management, gitignore, just tasks, spec amendment

**Files:**
- Modify: `scripts/manage_location_roots.py`
- Modify: `.gitignore`
- Modify: `justfile`
- Modify: `docs/superpowers/specs/2026-05-30-geoportal-elevation-nmt-nmpt-design.md`
- Test: `tests/test_dem_cli.py`

- [ ] **Step 1: Add failing test for the root-kind mapping**

Append to `tests/test_dem_cli.py`:

```python
def test_manage_roots_knows_dem_kind(tmp_path):
    import scripts.manage_location_roots as mlr

    assert "dem" in mlr.KINDS
    payload = {"location_name": "Poznań"}
    path = mlr._path_for_kind(payload, "dem", tmp_path)
    assert str(path).endswith("dem_poznan")
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_dem_cli.py -k manage_roots -q`
Expected: FAIL with `AssertionError` (`"dem" not in KINDS`) or `KeyError: 'dem'`.

- [ ] **Step 3a: Add the `dem` kind to `manage_location_roots.py`**

In `scripts/manage_location_roots.py`, change line 14:

```python
KINDS = ("downloads", "rendered", "artifacts")
```

to:

```python
KINDS = ("downloads", "rendered", "artifacts", "dem")
```

and in `_path_for_kind`, change the mapping:

```python
    mapping = {
        "downloads": "download_root",
        "rendered": "render_root",
        "artifacts": "artifacts_dir",
    }
```

to:

```python
    mapping = {
        "downloads": "download_root",
        "rendered": "render_root",
        "artifacts": "artifacts_dir",
        "dem": "dem_root",
    }
```

- [ ] **Step 3b: Ignore `dem_*/` dirs**

In `.gitignore`, after line 3 (`download*/`), add:

```
dem_*/
```

- [ ] **Step 3c: Add just tasks**

Append to `justfile`:

```just
# Download ISOK elevation (NMT/NMPT) for a single location
dem-location-json location_json:
    python -m satmap_dataset.cli dem-location-json {{location_json}}

# Download ISOK elevation for all locations in the default dir
dem-all-json:
    python -m satmap_dataset.cli dem-all-location-json
```

- [ ] **Step 3d: Amend the spec's `.prj` sidecar line**

In `docs/superpowers/specs/2026-05-30-geoportal-elevation-nmt-nmpt-design.md`, find the output-convention block line:

```
  dem_manifest.json
```

and the two asset lines that say `+ .prj`. Replace the comment about `+ .prj sidecars` in the **Output directory convention** section: change

```
  native/   {nmt,nmpt}_{evrf2007|kron86}.tif   + .prj   # authoritative 1 m
  aligned/  {nmt,nmpt}_{evrf2007|kron86}.tif   + .prj   # matches render grid
```

to

```
  native/   {nmt,nmpt}_{evrf2007|kron86}.tif            # authoritative 1 m (CRS embedded by GDAL)
  aligned/  {nmt,nmpt}_{evrf2007|kron86}.tif            # matches render grid (CRS embedded by GDAL)
```

Also remove `+ .prj sidecars` from the Components §2 line so the spec matches the implementation.

- [ ] **Step 4: Run the targeted test + full suite**

Run: `pytest tests/test_dem_cli.py -k manage_roots -q`
Expected: PASS.

Run: `pytest -q`
Expected: PASS (entire suite, including all new DEM tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/manage_location_roots.py .gitignore justfile docs/superpowers/specs/2026-05-30-geoportal-elevation-nmt-nmpt-design.md tests/test_dem_cli.py
git commit -m "feat(dem): wire dem root into roots mgmt, gitignore, just tasks; sync spec"
```

---

## Task 10: End-to-end smoke (manual, outside sandbox)

**Files:** none (manual verification — the dev sandbox blocks geoportal).

- [ ] **Step 1: Single-location run against the live WCS**

Run (outside the sandbox, with GDAL installed):

```bash
python -m satmap_dataset.cli dem \
  --center-lat 52.40 --center-lon 16.90 --square-km 1.0 \
  --products nmt,nmpt --vertical-datum evrf2007 --no-align \
  --dem-root ./dem_smoke
```

Expected: exit code `0`; last stdout line is an absolute path to `dem_smoke/dem_manifest.json`; `dem_smoke/native/nmt_evrf2007.tif` and `dem_smoke/native/nmpt_evrf2007.tif` exist and open in QGIS as single-band float32 EPSG:2180 rasters.

- [ ] **Step 2: Verify alignment against a rendered ortho (optional)**

If a `dataset_manifest_render.json` exists for the same AOI, run `dem-location-json` and confirm `dem_<slug>/aligned/nmt_evrf2007.tif` has the same width/height/extent as the rendered `year_YYYY.tif`.

- [ ] **Step 3: Record results**

Note the actual GUGiK WCS per-request pixel cap and axis-label behavior observed; if the default `max_request_px=2048` or `axis_label_x/y="x"/"y"` needed adjusting, update `DemConfig` defaults / `provider_options` docs accordingly.

---

## Self-Review

**Spec coverage:**
- Extend geoportal provider → Task 7 (`GeoportalProvider.dem`). ✓
- WCS GetCoverage + tiling/merge → Tasks 3–6. ✓
- Both NMT + NMPT → `DemConfig.products` default, iterated in `dem.run`. ✓
- Both datums, default EVRF2007 → `DemConfig.vertical_datum` + `coverage_id`. ✓
- Native 1 m + ortho-aligned → `dem.run` writes `native/` always, `aligned/` when `align_to_render`. ✓
- Output dir convention + manifest → Task 6 paths + Task 2 model. ✓
- CLI three+batch flavors, exit codes, last-line artifact path → Task 8. ✓
- Roots mgmt / gitignore / just → Task 9. ✓
- Known limitations recorded → `DemManifest.notes`. ✓
- Tests for tiling math, coverage-id mapping, config validation, base+location merge, manifest round-trip → Tasks 1–8. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code. ✓

**Type consistency:** `coverage_id(product, datum, options)`, `endpoint_url(product, options)`, `split_bbox(bbox, max_request_px, gsd_m)`, `get_coverage(endpoint, coverage_id_value, sub_bbox, srs, *, options, timeout, retry_policy, client)`, `_resolve_align_grid -> (bbox_tuple, w, h)`, `DemProductAsset`/`DemManifest` field names are used identically across Tasks 3–8. ✓

**Deviation:** `.prj` sidecars dropped (GDAL embeds CRS) — reconciled in Task 9 Step 3d. ✓
