# Historical NMT/NMPT Skorowidz Downloader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a year-aware `transport: "skorowidz"` mode to the existing `dem` stage that downloads historical NMT/NMPT `.asc` tiles from the GUGiK WFS skorowidz, mosaics them per acquisition year into float32 EPSG:2180 GeoTIFFs (native + ortho-grid-aligned).

**Architecture:** `DemConfig` gains `transport` + `year_start`/`year_end`. `pipeline/dem.py:run()` dispatches `skorowidz` to a new `pipeline/dem_skorowidz.py`, which reuses the ORTO `wfs_client` (generalized with a `typename_pattern`) for per-year indexing, the lantmäteriet async `_download_asset_with_retry` for `.asc` fetch, and `dem.py`'s GDAL `_align_to_grid`/`_resolve_align_grid`/`_raster_dims` helpers. A new `geoportal/dem_skorowidz_client.py` maps `(product, datum)` to the long Polish service URLs.

**Tech Stack:** Python ≥3.10, Pydantic v2, httpx (async), GDAL CLI (`gdalbuildvrt`/`gdal_translate`/`gdalwarp`), tifffile, Typer, pytest.

**Reference spec:** `docs/superpowers/specs/2026-05-30-geoportal-dem-skorowidz-historical-design.md`

**Scoping note (intentional):** `wfs_client.get_year_tiles` returns tile-id→url (tile-id ≈ godło for NMT) and acquisition metadata, but not `blad_sr_wys`. To avoid forking shared ORTO code, `DemYearAsset.godla` and `tile_count` are populated, and `mean_height_error` stays `None` in this version (documented in the manifest model + spec). Task 7 reconciles the spec line.

---

## File Structure

- Modify `src/satmap_dataset/geoportal/wfs_client.py` — add `typename_pattern` param to `get_capabilities` + `_extract_year_typenames` (default = current ORTO regex; no behavior change).
- Create `src/satmap_dataset/geoportal/dem_skorowidz_client.py` — endpoint + typename-pattern mapping; async `year_typenames`/`tiles_for_year` wrappers over `wfs_client`.
- Modify `src/satmap_dataset/models.py` — `DemYearAsset`; `DemProductAsset.years`; `DemManifest.transport`/`years_requested`/`years_skipped`.
- Modify `src/satmap_dataset/config.py` — `DemConfig.transport`/`year_start`/`year_end` + validators + `requested_years`.
- Create `src/satmap_dataset/pipeline/dem_skorowidz.py` — the skorowidz stage (`run`).
- Modify `src/satmap_dataset/pipeline/dem.py` — dispatch on `config.transport`.
- Modify `src/satmap_dataset/cli.py` — `--transport`/`--year-start`/`--year-end` on the `dem` flag command.
- Tests: `tests/test_wfs_client_typename_pattern.py`, `tests/test_dem_skorowidz_client.py`, `tests/test_dem_skorowidz_pipeline.py`, plus additions to `tests/test_dem_models.py`, `tests/test_dem_config.py`, `tests/test_dem_cli.py`.

---

## Task 1: Generalize `wfs_client` typename pattern

**Files:**
- Modify: `src/satmap_dataset/geoportal/wfs_client.py`
- Test: `tests/test_wfs_client_typename_pattern.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_wfs_client_typename_pattern.py`:

```python
import re
import xml.etree.ElementTree as ET

from satmap_dataset.geoportal import wfs_client

CAPS = """<?xml version='1.0'?>
<wfs:WFS_Capabilities xmlns:wfs="http://www.opengis.net/wfs/2.0">
  <wfs:FeatureTypeList>
    <wfs:FeatureType><wfs:Name>gugik:SkorowidzNMT2012</wfs:Name></wfs:FeatureType>
    <wfs:FeatureType><wfs:Name>gugik:SkorowidzNMT2019</wfs:Name></wfs:FeatureType>
    <wfs:FeatureType><wfs:Name>gugik:SkorowidzOrtofoto2021</wfs:Name></wfs:FeatureType>
  </wfs:FeatureTypeList>
</wfs:WFS_Capabilities>"""


def test_default_pattern_extracts_orto_years():
    root = ET.fromstring(CAPS)
    mapping = wfs_client._extract_year_typenames(root)
    assert mapping == {2021: "gugik:SkorowidzOrtofoto2021"}


def test_custom_pattern_extracts_nmt_years():
    root = ET.fromstring(CAPS)
    pattern = re.compile(r"SkorowidzNMT(\d{4})", re.IGNORECASE)
    mapping = wfs_client._extract_year_typenames(root, pattern)
    assert mapping == {2012: "gugik:SkorowidzNMT2012", 2019: "gugik:SkorowidzNMT2019"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_wfs_client_typename_pattern.py -q`
Expected: FAIL — `test_custom_pattern_extracts_nmt_years` errors because `_extract_year_typenames` takes only one argument.

- [ ] **Step 3: Implement the generalization**

In `src/satmap_dataset/geoportal/wfs_client.py`, replace the function:

```python
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
```

with:

```python
DEFAULT_TYPENAME_PATTERN = re.compile(r"SkorowidzOrtof\w*?(\d{4})$", re.IGNORECASE)


def _extract_year_typenames(
    cap_root: ET.Element, pattern: "re.Pattern[str] | None" = None
) -> dict[int, str]:
    regex = pattern or DEFAULT_TYPENAME_PATTERN
    year_to_typename: dict[int, str] = {}
    for name in _iter_featuretype_names(cap_root):
        match = regex.search(name)
        if not match:
            continue
        year = int(match.group(1))
        year_to_typename.setdefault(year, name)
    return year_to_typename
```

Then update `get_capabilities` to accept and forward the pattern. Replace:

```python
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
```

with:

```python
async def get_capabilities(
    base_url: str = DEFAULT_WFS_URL,
    *,
    timeout: float = 20.0,
    retry_policy: RetryPolicy | None = None,
    typename_pattern: "re.Pattern[str] | None" = None,
) -> tuple[ET.Element, dict[int, str]]:
    response = await request_with_retry(
        "GET",
        base_url,
        params={"service": "WFS", "request": "GetCapabilities", "version": "2.0.0"},
        timeout=timeout,
        retry_policy=retry_policy,
    )
    root = ET.fromstring(response.text)
    return root, _extract_year_typenames(root, typename_pattern)
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_wfs_client_typename_pattern.py -q` (2 pass). Then `pytest -q` (full suite stays green; this must not break the ORTO index tests).

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/geoportal/wfs_client.py tests/test_wfs_client_typename_pattern.py
git commit -m "feat(dem): parameterize wfs_client typename pattern (default unchanged)"
```

---

## Task 2: `dem_skorowidz_client.py` — endpoints + WFS wrappers

**Files:**
- Create: `src/satmap_dataset/geoportal/dem_skorowidz_client.py`
- Test: `tests/test_dem_skorowidz_client.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dem_skorowidz_client.py`:

```python
import asyncio

import pytest

from satmap_dataset.geoportal import dem_skorowidz_client as dsc


def test_endpoint_all_combinations():
    assert dsc.endpoint("nmt", "kron86").endswith("NumerycznyModelTerenuKRON86/WFS/Skorowidze")
    assert dsc.endpoint("nmt", "evrf2007").endswith("NumerycznyModelTerenuEVRF2007/WFS/Skorowidze")
    assert dsc.endpoint("nmpt", "kron86").endswith("NumerycznyModelPokryciaTerenuKRON86/WFS/Skorowidze")
    assert dsc.endpoint("nmpt", "evrf2007").endswith("NumerycznyModelPokryciaTerenuEVRF2007/WFS/Skorowidze")


def test_endpoint_override_and_unknown():
    opts = {"skorowidz_endpoints": {"nmt|kron86": "https://example/custom"}}
    assert dsc.endpoint("nmt", "kron86", opts) == "https://example/custom"
    with pytest.raises(ValueError):
        dsc.endpoint("foo", "kron86")
    with pytest.raises(ValueError):
        dsc.endpoint("nmt", "baddatum")


def test_typename_pattern_matches_product_years():
    pat = dsc.typename_pattern("nmt")
    assert pat.search("gugik:SkorowidzNMT2019").group(1) == "2019"
    assert pat.search("gugik:SkorowidzNMPT2019") is None
    patp = dsc.typename_pattern("nmpt")
    assert patp.search("gugik:SkorowidzNMPT2018").group(1) == "2018"


def test_year_typenames_uses_wfs_client(monkeypatch):
    async def _fake_caps(base_url, *, timeout, retry_policy, typename_pattern):
        assert "NumerycznyModelTerenuKRON86" in base_url
        assert typename_pattern.search("gugik:SkorowidzNMT2012")
        return (None, {2012: "gugik:SkorowidzNMT2012"})

    monkeypatch.setattr(dsc.wfs_client, "get_capabilities", _fake_caps)
    out = asyncio.run(dsc.year_typenames("nmt", "kron86"))
    assert out == {2012: "gugik:SkorowidzNMT2012"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_dem_skorowidz_client.py -q`
Expected: FAIL — `ModuleNotFoundError: ...dem_skorowidz_client`.

- [ ] **Step 3: Implement the client**

Create `src/satmap_dataset/geoportal/dem_skorowidz_client.py`:

```python
from __future__ import annotations

import re
from typing import Any

from satmap_dataset.geoportal import wfs_client
from satmap_dataset.geoportal.http import RetryPolicy
from satmap_dataset.models import YearStatus

_BASE = "https://mapy.geoportal.gov.pl/wss/service/PZGIK"
_SERVICE = {
    ("nmt", "kron86"): "NumerycznyModelTerenuKRON86",
    ("nmt", "evrf2007"): "NumerycznyModelTerenuEVRF2007",
    ("nmpt", "kron86"): "NumerycznyModelPokryciaTerenuKRON86",
    ("nmpt", "evrf2007"): "NumerycznyModelPokryciaTerenuEVRF2007",
}
_TYPENAME_TOKEN = {"nmt": "NMT", "nmpt": "NMPT"}


def endpoint(product: str, datum: str, options: dict[str, Any] | None = None) -> str:
    options = options or {}
    overrides = options.get("skorowidz_endpoints") or {}
    key = f"{product}|{datum}"
    if key in overrides:
        return str(overrides[key])
    if (product, datum) not in _SERVICE:
        raise ValueError(
            f"Unknown (product, datum)=({product!r}, {datum!r}); "
            f"expected product in {sorted(_TYPENAME_TOKEN)} and datum in {{evrf2007, kron86}}."
        )
    return f"{_BASE}/{_SERVICE[(product, datum)]}/WFS/Skorowidze"


def typename_pattern(product: str) -> "re.Pattern[str]":
    if product not in _TYPENAME_TOKEN:
        raise ValueError(f"Unknown product {product!r}; expected one of {sorted(_TYPENAME_TOKEN)}")
    # NMT must not also match NMPT: require the token followed immediately by 4 digits.
    return re.compile(rf"Skorowidz{_TYPENAME_TOKEN[product]}(\d{{4}})", re.IGNORECASE)


async def year_typenames(
    product: str,
    datum: str,
    options: dict[str, Any] | None = None,
    *,
    timeout: float = 45.0,
    retry_policy: RetryPolicy | None = None,
) -> dict[int, str]:
    _root, mapping = await wfs_client.get_capabilities(
        base_url=endpoint(product, datum, options),
        timeout=timeout,
        retry_policy=retry_policy,
        typename_pattern=typename_pattern(product),
    )
    return mapping


async def tiles_for_year(
    product: str,
    datum: str,
    year: int,
    bbox: str,
    srs: str,
    *,
    year_to_typename: dict[int, str],
    options: dict[str, Any] | None = None,
    timeout: float = 45.0,
    retry_policy: RetryPolicy | None = None,
) -> tuple[YearStatus, dict[str, str], dict[str, list[float]], dict[str, dict[str, int | str | None]]]:
    return await wfs_client.get_year_tiles(
        year=year,
        bbox=bbox,
        srs=srs,
        base_url=endpoint(product, datum, options),
        timeout=timeout,
        retry_policy=retry_policy,
        year_to_typename=year_to_typename,
    )
```

Note: `typename_pattern("nmt")` uses `SkorowidzNMT(\d{4})`. Because "NMPT" has a letter ("P") between "NM" and "T", `SkorowidzNMT\d{4}` will NOT match `SkorowidzNMPT2019` — the test asserts this.

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_dem_skorowidz_client.py -q` (4 pass). Then `pytest -q` (green).

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/geoportal/dem_skorowidz_client.py tests/test_dem_skorowidz_client.py
git commit -m "feat(dem): skorowidz WFS client (endpoints + year/tile wrappers)"
```

---

## Task 3: Manifest models for year-keyed DEM

**Files:**
- Modify: `src/satmap_dataset/models.py`
- Test: `tests/test_dem_models.py`

- [ ] **Step 1: Add failing test**

APPEND to `tests/test_dem_models.py`:

```python
def test_dem_manifest_skorowidz_round_trip():
    from satmap_dataset.models import DemYearAsset

    manifest = DemManifest(
        bbox="0,0,10,10", srs="EPSG:2180", vertical_datum="kron86",
        transport="skorowidz", years_requested=[2012, 2019], years_skipped={2015: "no tiles in AOI"},
        products=[
            DemProductAsset(
                product="nmt", coverage_id="skorowidz:nmt:kron86", endpoint="https://example/wfs",
                years=[
                    DemYearAsset(
                        year=2012, native_path="dem_x/skorowidz/nmt_kron86/native/year_2012.tif",
                        native_width=10, native_height=10, tile_count=2,
                        godla=["N-33-141-C-a-3-4"], passed=True,
                    )
                ],
                passed=True,
            )
        ],
        passed=True,
    )
    restored = DemManifest.model_validate_json(manifest.model_dump_json())
    assert restored.transport == "skorowidz"
    assert restored.years_requested == [2012, 2019]
    assert restored.years_skipped == {2015: "no tiles in AOI"}
    assert restored.products[0].years[0].year == 2012
    assert restored.products[0].years[0].godla == ["N-33-141-C-a-3-4"]
    assert restored.products[0].years[0].mean_height_error is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_dem_models.py -k skorowidz -q`
Expected: FAIL — `ImportError: cannot import name 'DemYearAsset'`.

- [ ] **Step 3: Implement the model changes**

In `src/satmap_dataset/models.py`, ADD `DemYearAsset` immediately before `class DemProductAsset`:

```python
class DemYearAsset(BaseModel):
    year: int = Field(..., ge=1900)
    native_path: str | None = None
    native_width: int | None = None
    native_height: int | None = None
    aligned_path: str | None = None
    aligned_width: int | None = None
    aligned_height: int | None = None
    tile_count: int = Field(default=0, ge=0)
    mean_height_error: float | None = None  # not populated yet (would require extending wfs_client return)
    godla: list[str] = Field(default_factory=list)
    passed: bool = False
    errors: list[str] = Field(default_factory=list)
```

In `class DemProductAsset`, ADD this field (after `errors`):

```python
    years: list[DemYearAsset] = Field(default_factory=list)
```

In `class DemManifest`, ADD these fields (after `vertical_datum`):

```python
    transport: Literal["wcs", "skorowidz"] = "wcs"
    years_requested: list[int] = Field(default_factory=list)
    years_skipped: dict[int, str] = Field(default_factory=dict)
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_dem_models.py -q` (all pass). Then `pytest -q` (green).

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/models.py tests/test_dem_models.py
git commit -m "feat(dem): year-keyed manifest (DemYearAsset, transport, years_skipped)"
```

---

## Task 4: `DemConfig` transport + year range

**Files:**
- Modify: `src/satmap_dataset/config.py`
- Test: `tests/test_dem_config.py`

- [ ] **Step 1: Add failing tests**

APPEND to `tests/test_dem_config.py`:

```python
def test_transport_default_and_enum():
    assert DemConfig(bbox="0,0,10,10").transport == "wcs"
    assert DemConfig(bbox="0,0,10,10", transport="skorowidz", year_start=2012, year_end=2019).transport == "skorowidz"
    with pytest.raises(ValidationError):
        DemConfig(bbox="0,0,10,10", transport="ftp")


def test_skorowidz_requires_valid_year_range():
    with pytest.raises(ValidationError):
        DemConfig(bbox="0,0,10,10", transport="skorowidz")  # years missing
    with pytest.raises(ValidationError):
        DemConfig(bbox="0,0,10,10", transport="skorowidz", year_start=2019, year_end=2012)  # reversed
    cfg = DemConfig(bbox="0,0,10,10", transport="skorowidz", year_start=2012, year_end=2014)
    assert cfg.requested_years == [2012, 2013, 2014]


def test_wcs_ignores_year_range():
    cfg = DemConfig(bbox="0,0,10,10")  # wcs, no years
    assert cfg.requested_years == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_dem_config.py -k "transport or year" -q`
Expected: FAIL — `transport` is not a field / no `requested_years`.

- [ ] **Step 3: Implement the config changes**

In `src/satmap_dataset/config.py`, inside `class DemConfig`, ADD these fields (place after `srs`):

```python
    transport: str = "wcs"
    year_start: int | None = Field(default=None, ge=1900)
    year_end: int | None = Field(default=None, ge=1900)
```

ADD a field validator for `transport` (next to the other `@field_validator`s in `DemConfig`):

```python
    @field_validator("transport")
    @classmethod
    def validate_transport(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized not in {"wcs", "skorowidz"}:
            raise ValueError("transport must be 'wcs' or 'skorowidz'")
        return normalized
```

In the existing `@model_validator(mode="after") def validate_invariants`, ADD before `return self`:

```python
        if self.transport == "skorowidz":
            if self.year_start is None or self.year_end is None:
                raise ValueError("year_start and year_end are required when transport='skorowidz'")
            if self.year_end < self.year_start:
                raise ValueError("year_end must be >= year_start")
```

ADD a `requested_years` property at the end of `class DemConfig`:

```python
    @property
    def requested_years(self) -> list[int]:
        if self.year_start is None or self.year_end is None:
            return []
        return list(range(self.year_start, self.year_end + 1))
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_dem_config.py -q` (all pass). Then `pytest -q` (green).

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/config.py tests/test_dem_config.py
git commit -m "feat(dem): DemConfig transport + year range (skorowidz)"
```

---

## Task 5: `pipeline/dem_skorowidz.py` — the stage

**Files:**
- Create: `src/satmap_dataset/pipeline/dem_skorowidz.py`
- Test: `tests/test_dem_skorowidz_pipeline.py`

- [ ] **Step 1: Write the failing tests (network + GDAL mocked)**

Create `tests/test_dem_skorowidz_pipeline.py`:

```python
from pathlib import Path

from satmap_dataset.config import DemConfig
from satmap_dataset.models import DemManifest, YearStatus
from satmap_dataset.pipeline import dem_skorowidz


def _patch(monkeypatch, *, tiles_by_year):
    """tiles_by_year: {year: {godlo: url}} ; empty dict -> skipped year."""

    async def _fake_year_typenames(product, datum, options=None, *, timeout=45.0, retry_policy=None):
        return {y: f"gugik:SkorowidzNMT{y}" for y in tiles_by_year}

    async def _fake_tiles_for_year(product, datum, year, bbox, srs, *, year_to_typename, options=None, timeout=45.0, retry_policy=None):
        tiles = tiles_by_year[year]
        status = YearStatus(year=year, typename_exists=True, feature_count=len(tiles),
                            status="has_features" if tiles else "zero_features")
        return status, dict(tiles), {}, {}

    async def _fake_download(urls, dest_dir, config, retry_policy):
        out = []
        for i, _u in enumerate(urls):
            p = Path(dest_dir) / f"t{i}.asc"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("ncols 1\n")
            out.append(p)
        return out

    def _fake_mosaic(tiles, out_path, bbox):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"NATIVE")

    def _fake_align(native, out_path, **kwargs):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"ALIGNED")

    monkeypatch.setattr(dem_skorowidz.dem_skorowidz_client, "year_typenames", _fake_year_typenames)
    monkeypatch.setattr(dem_skorowidz.dem_skorowidz_client, "tiles_for_year", _fake_tiles_for_year)
    monkeypatch.setattr(dem_skorowidz, "_download_tiles", _fake_download)
    monkeypatch.setattr(dem_skorowidz, "_mosaic_asc_to_native", _fake_mosaic)
    monkeypatch.setattr(dem_skorowidz.dem, "_align_to_grid", _fake_align)
    monkeypatch.setattr(dem_skorowidz.dem, "_raster_dims", lambda path: (10, 10))


def _cfg(tmp_path, **kw):
    base = dict(
        bbox="0,0,100,100", transport="skorowidz", year_start=2012, year_end=2019,
        products=["nmt"], vertical_datum="kron86", align_to_render=False,
        dem_root=tmp_path / "dem_x", output_json=tmp_path / "dem_x" / "dem_manifest.json",
    )
    base.update(kw)
    return DemConfig(**base)


def test_run_year_keyed_native(tmp_path, monkeypatch):
    _patch(monkeypatch, tiles_by_year={2012: {"g1": "u1", "g2": "u2"}, 2019: {"g3": "u3"}})
    code, path = dem_skorowidz.run(_cfg(tmp_path))
    assert code == 0
    m = DemManifest.model_validate_json(Path(path).read_text())
    assert m.transport == "skorowidz"
    nmt = m.products[0]
    years = {y.year: y for y in nmt.years}
    assert set(years) == {2012, 2019}
    assert years[2012].tile_count == 2 and years[2012].passed
    assert years[2012].godla == ["g1", "g2"]
    assert Path(years[2012].native_path).exists()
    assert years[2019].aligned_path is None  # align disabled


def test_run_skips_empty_year(tmp_path, monkeypatch):
    _patch(monkeypatch, tiles_by_year={2012: {"g1": "u1"}, 2015: {}})
    code, path = dem_skorowidz.run(_cfg(tmp_path))
    assert code == 0
    m = DemManifest.model_validate_json(Path(path).read_text())
    assert 2015 in m.years_skipped
    assert {y.year for y in m.products[0].years} == {2012}


def test_run_aligns_when_enabled(tmp_path, monkeypatch):
    _patch(monkeypatch, tiles_by_year={2012: {"g1": "u1"}})
    cfg = _cfg(tmp_path, align_to_render=True, target_bbox="0,0,100,100", target_width=100, target_height=100)
    code, path = dem_skorowidz.run(cfg)
    assert code == 0
    m = DemManifest.model_validate_json(Path(path).read_text())
    y = m.products[0].years[0]
    assert Path(y.aligned_path).exists() and y.aligned_width == 100


def test_run_fails_when_no_years_available(tmp_path, monkeypatch):
    _patch(monkeypatch, tiles_by_year={})  # no typenames in range
    code, path = dem_skorowidz.run(_cfg(tmp_path, year_start=2012, year_end=2012))
    assert code == 1
    m = DemManifest.model_validate_json(Path(path).read_text())
    assert m.passed is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_dem_skorowidz_pipeline.py -q`
Expected: FAIL — `ModuleNotFoundError: ...pipeline.dem_skorowidz`.

- [ ] **Step 3: Implement the stage**

Create `src/satmap_dataset/pipeline/dem_skorowidz.py`:

```python
from __future__ import annotations

import asyncio
import logging
import subprocess
import tempfile
from pathlib import Path

import httpx

from satmap_dataset.config import DemConfig
from satmap_dataset.geoportal import dem_skorowidz_client
from satmap_dataset.geoportal.http import RetryPolicy
from satmap_dataset.models import DemManifest, DemProductAsset, DemYearAsset
from satmap_dataset.pipeline import dem
from satmap_dataset.providers.lantmateriet.provider import _download_asset_with_retry

logger = logging.getLogger("satmap_dataset.dem_skorowidz")


async def _download_tiles(
    urls: list[str], dest_dir: Path, config: DemConfig, retry_policy: RetryPolicy
) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    timeout = httpx.Timeout(timeout=config.timeout, connect=min(config.timeout, 20.0))
    headers = {"User-Agent": "satmap_dataset/0.1"}
    paths: list[Path] = []
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout, headers=headers) as client:
        for url in urls:
            out = dest_dir / Path(url).name
            ok = await _download_asset_with_retry(
                client, url, out,
                retries=config.retries, retry_delay=config.retry_delay,
                sleep_min=config.sleep_min, sleep_max=config.sleep_max,
            )
            if not ok:
                raise RuntimeError(f"download failed: {url}")
            paths.append(out)
    return paths


def _mosaic_asc_to_native(tiles: list[Path], out_path: Path, bbox: tuple[float, float, float, float]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    buildvrt = dem._tool_path("gdalbuildvrt")
    translate = dem._tool_path("gdal_translate")
    if not buildvrt or not translate:
        raise RuntimeError(
            "Mosaicking .asc tiles requires the GDAL CLI (gdalbuildvrt, gdal_translate). Install GDAL."
        )
    xmin, ymin, xmax, ymax = bbox
    vrt_path = out_path.with_suffix(".vrt")
    try:
        subprocess.run(
            [buildvrt, "-a_srs", "EPSG:2180", str(vrt_path), *[str(t) for t in tiles]],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(
            [
                translate, "-a_srs", "EPSG:2180",
                "-projwin", str(xmin), str(ymax), str(xmax), str(ymin),  # ulx uly lrx lry
                "-co", "COMPRESS=DEFLATE", str(vrt_path), str(out_path),
            ],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"GDAL .asc mosaic failed: {(exc.stderr or '')[-500:]}") from exc
    finally:
        vrt_path.unlink(missing_ok=True)


async def _run_async(config: DemConfig) -> tuple[int, Path]:
    retry_policy = RetryPolicy(max_attempts=config.retries, backoff_seconds=config.retry_delay)
    grid = dem._resolve_align_grid(config) if config.align_to_render else None
    resample = str(config.provider_options.get("resample", "bilinear"))
    bbox = dem._parse_bbox(config.bbox)
    requested = config.requested_years
    options = dict(config.provider_options)

    product_assets: list[DemProductAsset] = []
    years_skipped: dict[int, str] = {}

    for product in config.products:
        datum = config.vertical_datum
        asset = DemProductAsset(
            product=product,
            coverage_id=f"skorowidz:{product}:{datum}",
            endpoint=dem_skorowidz_client.endpoint(product, datum, options),
        )
        try:
            year_to_typename = await dem_skorowidz_client.year_typenames(
                product, datum, options, timeout=config.timeout, retry_policy=retry_policy
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("skorowidz capabilities failed for %s/%s: %s", product, datum, exc)
            year_to_typename = {}
        available = [y for y in requested if y in year_to_typename]

        year_assets: list[DemYearAsset] = []
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            for year in available:
                ya = DemYearAsset(year=year)
                native = (
                    config.dem_root / "skorowidz" / f"{product}_{datum}" / "native" / f"year_{year}.tif"
                )
                try:
                    _status, tiles, _bb, _acq = await dem_skorowidz_client.tiles_for_year(
                        product, datum, year, config.bbox, config.srs,
                        year_to_typename=year_to_typename, options=options,
                        timeout=config.timeout, retry_policy=retry_policy,
                    )
                    if not tiles:
                        years_skipped[year] = "no tiles in AOI"
                        continue
                    ya.godla = sorted(tiles.keys())
                    if not (native.exists() and not config.overwrite):
                        paths = await _download_tiles(
                            list(tiles.values()), tmp_dir / f"{product}_{year}", config, retry_policy
                        )
                        _mosaic_asc_to_native(paths, native, bbox)
                        ya.tile_count = len(paths)
                    ya.native_path = str(native)
                    ya.native_width, ya.native_height = dem._raster_dims(native)
                    if grid is not None:
                        aligned = (
                            config.dem_root / "skorowidz" / f"{product}_{datum}" / "aligned" / f"year_{year}.tif"
                        )
                        target_bbox, gw, gh = grid
                        dem._align_to_grid(
                            native, aligned, target_bbox=target_bbox,
                            target_width=gw, target_height=gh, srs=config.srs, resample=resample,
                        )
                        ya.aligned_path = str(aligned)
                        ya.aligned_width, ya.aligned_height = gw, gh
                    ya.passed = True
                except Exception as exc:  # noqa: BLE001 - record per-year and continue
                    ya.errors.append(str(exc))
                year_assets.append(ya)

        asset.years = year_assets
        asset.passed = any(y.passed for y in year_assets)
        product_assets.append(asset)

    passed = any(y.passed for a in product_assets for y in a.years)
    manifest = DemManifest(
        provider="geoportal", bbox=config.bbox, srs=config.srs, vertical_datum=config.vertical_datum,
        transport="skorowidz", years_requested=requested, years_skipped=years_skipped,
        products=product_assets, align_to_render=config.align_to_render, passed=passed,
        notes="GUGiK skorowidz (WFS) historical NMT/NMPT; one mosaic per ALS acquisition year.",
        run_parameters=config.model_dump(mode="json"),
    )
    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    config.output_json.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    logger.info(
        "DEM skorowidz: products=%s years_done=%s skipped=%s passed=%s",
        [a.product for a in product_assets],
        sum(1 for a in product_assets for y in a.years if y.passed),
        len(years_skipped), passed,
    )
    return (0 if passed else 1), config.output_json


def run(config: DemConfig) -> tuple[int, Path]:
    return asyncio.run(_run_async(config))
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_dem_skorowidz_pipeline.py -q` (4 pass). Then `pytest -q` (green).

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/pipeline/dem_skorowidz.py tests/test_dem_skorowidz_pipeline.py
git commit -m "feat(dem): skorowidz stage (per-year .asc mosaic, native + aligned)"
```

---

## Task 6: Dispatch `transport` in `pipeline/dem.py`

**Files:**
- Modify: `src/satmap_dataset/pipeline/dem.py`
- Test: `tests/test_dem_pipeline.py`

- [ ] **Step 1: Add failing tests**

APPEND to `tests/test_dem_pipeline.py`:

```python
def test_run_dispatches_skorowidz(tmp_path, monkeypatch):
    called = {}

    def _fake_skoro_run(config):
        called["config"] = config
        return (0, tmp_path / "m.json")

    monkeypatch.setattr("satmap_dataset.pipeline.dem_skorowidz.run", _fake_skoro_run)
    cfg = DemConfig(
        bbox="0,0,10,10", transport="skorowidz", year_start=2012, year_end=2012,
        dem_root=tmp_path / "dem_x", output_json=tmp_path / "dem_x" / "m.json",
    )
    code, _ = dem.run(cfg)
    assert code == 0
    assert called["config"] is cfg


def test_run_wcs_path_not_dispatched_to_skorowidz(tmp_path, monkeypatch):
    _patch_seams(monkeypatch)  # defined earlier in this file
    monkeypatch.setattr(
        "satmap_dataset.pipeline.dem_skorowidz.run",
        lambda config: (_ for _ in ()).throw(AssertionError("must not call skorowidz")),
    )
    cfg = DemConfig(
        bbox="0,0,100,100", products=["nmt"], align_to_render=False,
        dem_root=tmp_path / "dem_x", output_json=tmp_path / "dem_x" / "dem_manifest.json",
    )
    code, _ = dem.run(cfg)
    assert code == 0  # WCS path ran
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_dem_pipeline.py -k "dispatches_skorowidz or wcs_path_not" -q`
Expected: FAIL — `dem.run` always runs the WCS path (no dispatch yet), so `test_run_dispatches_skorowidz` fails (the fake isn't called / config mismatch).

- [ ] **Step 3: Implement the dispatch**

In `src/satmap_dataset/pipeline/dem.py`, replace the existing `run`:

```python
def run(config: DemConfig) -> tuple[int, Path]:
    return asyncio.run(_run_async(config))
```

with:

```python
def run(config: DemConfig) -> tuple[int, Path]:
    if config.transport == "skorowidz":
        from satmap_dataset.pipeline import dem_skorowidz

        return dem_skorowidz.run(config)
    return asyncio.run(_run_async(config))
```

(The import is local to avoid a circular import: `dem_skorowidz` imports `dem`.)

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_dem_pipeline.py -q` (all pass). Then `pytest -q` (green).

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/pipeline/dem.py tests/test_dem_pipeline.py
git commit -m "feat(dem): dispatch transport=skorowidz from dem.run"
```

---

## Task 7: CLI flags + spec reconciliation

**Files:**
- Modify: `src/satmap_dataset/cli.py`
- Modify: `docs/superpowers/specs/2026-05-30-geoportal-dem-skorowidz-historical-design.md`
- Test: `tests/test_dem_cli.py`

- [ ] **Step 1: Add failing tests**

APPEND to `tests/test_dem_cli.py`:

```python
def test_dem_json_skorowidz_dispatches(tmp_path, monkeypatch):
    captured = {}

    def _fake_run(config):
        captured["config"] = config
        out = tmp_path / "m.json"
        out.write_text("{}")
        return (0, out)

    # dem.run dispatches to dem_skorowidz.run; patch the underlying stage
    monkeypatch.setattr("satmap_dataset.pipeline.dem_skorowidz.run", _fake_run)
    params = tmp_path / "p.json"
    params.write_text(json.dumps({
        "bbox": "210300,521900,210500,522100", "transport": "skorowidz",
        "year_start": 2012, "year_end": 2019, "products": ["nmt"], "align_to_render": False,
        "dem_root": str(tmp_path / "dem_x"), "output_json": str(tmp_path / "m.json"),
    }))
    result = runner.invoke(app, ["dem-json", str(params)])
    assert result.exit_code == 0
    assert captured["config"].transport == "skorowidz"
    assert captured["config"].requested_years == list(range(2012, 2020))


def test_dem_flag_transport_year_options(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr("satmap_dataset.pipeline.dem_skorowidz.run",
                        lambda config: (captured.setdefault("c", config), (0, tmp_path / "m.json"))[1])
    result = runner.invoke(app, [
        "dem", "--bbox", "0,0,100,100", "--transport", "skorowidz",
        "--year-start", "2012", "--year-end", "2014", "--products", "nmt", "--no-align",
        "--dem-root", str(tmp_path / "dem_x"), "--output-json", str(tmp_path / "m.json"),
    ])
    assert result.exit_code == 0
    assert captured["c"].transport == "skorowidz"
    assert captured["c"].requested_years == [2012, 2013, 2014]


def test_dem_location_builder_inherits_years_from_base(tmp_path):
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"transport": "skorowidz", "year_start": 2011, "year_end": 2019}))
    loc = tmp_path / "location_x.json"
    loc.write_text(json.dumps({"location_name": "Test", "center_lat": 52.4, "center_lon": 16.9, "square_km": 1.0}))
    cfg = _build_dem_config_from_base_and_location(base_json=base, location_json=loc)
    assert cfg.transport == "skorowidz"
    assert cfg.requested_years == list(range(2011, 2020))
```

(`_build_dem_config_from_base_and_location` is already imported at the top of `tests/test_dem_cli.py` from Task 8 of the prior feature.)

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_dem_cli.py -k "skorowidz or transport or inherits_years" -q`
Expected: FAIL — the `dem` flag command has no `--transport`/`--year-start`/`--year-end`, so `test_dem_flag_transport_year_options` errors on the unknown option.

- [ ] **Step 3: Add the flags to the `dem` command**

In `src/satmap_dataset/cli.py`, find the `dem` command signature (the `def dem_command(...)` parameter list). ADD these three options (place them after the `srs` option):

```python
    transport: str = typer.Option("wcs", "--transport", help="wcs (current composite) or skorowidz (historical per-year)."),
    year_start: int = typer.Option(None, "--year-start", help="First year (skorowidz transport)."),
    year_end: int = typer.Option(None, "--year-end", help="Last year (skorowidz transport)."),
```

Then in the `payload` dict built inside `dem_command`, ADD these keys (alongside `bbox`, `srs`, ...):

```python
            "transport": transport,
            "year_start": year_start,
            "year_end": year_end,
```

No change is needed for `dem-json` (JSON maps straight onto `DemConfig`) or for
`_build_dem_config_from_base_and_location` (it already merges `base.json`, whose
`transport`/`year_start`/`year_end` keys now map onto the new `DemConfig` fields).

- [ ] **Step 4: Reconcile the spec's `mean_height_error` line**

In `docs/superpowers/specs/2026-05-30-geoportal-dem-skorowidz-historical-design.md`, find the `DemYearAsset` bullet in the "models additions" section that reads:

```
  `mean_height_error: float | None` (max `blad_sr_wys` across tiles), `godla: list[str]`,
```

Replace it with:

```
  `mean_height_error: float | None` (reserved; left `None` in this version — populating it
  would require extending the shared `wfs_client.get_year_tiles` return type), `godla: list[str]`,
```

- [ ] **Step 5: Run tests + full suite**

Run: `pytest tests/test_dem_cli.py -q` (all pass). Then `pytest -q` (entire suite green).

- [ ] **Step 6: Commit**

```bash
git add src/satmap_dataset/cli.py docs/superpowers/specs/2026-05-30-geoportal-dem-skorowidz-historical-design.md tests/test_dem_cli.py
git commit -m "feat(dem): CLI --transport/--year-start/--year-end; sync spec"
```

---

## Task 8: Live smoke (manual, outside sandbox)

**Files:** none (manual verification — the dev sandbox blocks geoportal).

- [ ] **Step 1: Run skorowidz for Przeźmierowo over a historical range**

Run (outside the sandbox, GDAL installed):

```bash
python -m satmap_dataset.cli dem \
  --center-lat 52.426178 --center-lon 16.785372 --square-km 1.0 \
  --transport skorowidz --year-start 2011 --year-end 2019 \
  --products nmt --vertical-datum kron86 --no-align \
  --dem-root ./dem_hist_przezmierowo
```

Expected: exit `0`; last stdout line is the manifest path; `dem_hist_przezmierowo/skorowidz/nmt_kron86/native/year_<YYYY>.tif` exist for each ALS year that covers the AOI; each opens as a single-band float32 EPSG:2180 raster; `years_skipped` lists in-range years with no local coverage.

- [ ] **Step 2: Verify NMPT + EVRF2007 reachability**

Run the same with `--products nmpt --vertical-datum evrf2007`. Expected: exit `0` and NMPT mosaics produced (confirms the four-endpoint mapping live).

- [ ] **Step 3: Verify alignment to the ortho grid (optional)**

With an existing `artifacts_<slug>/dataset_manifest_render.json`, run `dem-location-json` (base.json carrying `transport=skorowidz` + a year range) and confirm `aligned/year_<YYYY>.tif` match the rendered `year_YYYY.tiff` width/height/extent.

- [ ] **Step 4: Record results**

Note which years actually cover the AOI and any per-tile quirks; if the WFS BBOX axis order behaves differently than the per-feature auto-swap handles, record it.

---

## Self-Review

**Spec coverage:**
- Approach A `transport` dispatch → Task 4 (config) + Task 6 (dispatch). ✓
- `wfs_client` generalization (default unchanged) → Task 1. ✓
- `dem_skorowidz_client` endpoints (4 combos) + typename patterns → Task 2. ✓
- Per-year `.asc` download → mosaic (`-a_srs EPSG:2180`, AOI clip) → native; optional align → Task 5. ✓
- Year selection = all available in range → Task 5 (`available = [y for y in requested if y in year_to_typename]`). ✓
- Year-keyed output layout `dem_<slug>/skorowidz/<product>_<datum>/{native,aligned}/year_<YYYY>.tif` → Task 5. ✓
- Manifest `transport`/`years_requested`/`years_skipped`/`DemYearAsset` → Task 3. ✓
- Reuse-skip when native exists → Task 5 (`if not (native.exists() and not config.overwrite)`). ✓
- Per-(product,year) isolation; empty year → `years_skipped`; `passed` logic; exit 0/1 → Task 5. ✓
- CLI `--transport`/`--year-start`/`--year-end`; location builder inherits years → Task 7. ✓
- `mean_height_error` deferred → reconciled in Task 7 Step 4. ✓
- Default `transport=wcs` preserved → Task 4 + Task 6 (`test_run_wcs_path_not_dispatched_to_skorowidz`). ✓

**Placeholder scan:** No TBD/TODO; every code step has complete code. ✓

**Type consistency:** `dem_skorowidz_client.endpoint(product, datum, options)`, `.typename_pattern(product)`, `.year_typenames(product, datum, options, *, timeout, retry_policy)`, `.tiles_for_year(...) -> (YearStatus, dict, dict, dict)`; `dem_skorowidz._download_tiles(urls, dest_dir, config, retry_policy)`, `_mosaic_asc_to_native(tiles, out_path, bbox)`; reused `dem._parse_bbox/_resolve_align_grid/_align_to_grid/_raster_dims/_tool_path`; `DemYearAsset`/`DemProductAsset.years`/`DemManifest.transport` consistent across Tasks 2–7. ✓

**Reuse safety:** Task 1 keeps the ORTO default pattern (regression guarded by `test_default_pattern_extracts_orto_years` + full suite). The `dem.run` skorowidz import is local to avoid the `dem ↔ dem_skorowidz` circular import. ✓
