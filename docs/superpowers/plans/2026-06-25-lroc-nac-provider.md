# LROC NAC Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `lroc_nac` orthophoto provider that enumerates multi-temporal lunar LROC NAC observations over a lat/lon bbox + date range via the PDS Orbital Data Explorer (ODE) REST API and downloads the frames.

**Architecture:** A new `src/satmap_dataset/providers/lroc_nac/` package mirroring the existing `lantmateriet/` provider: a pure-parsing ODE client (`ode.py`), a lunar-CRS bbox helper (`crs.py`), and a `LrocNacProvider` (`provider.py`) implementing the `Provider` ABC's `index()`/`download()`. Index + download only; ISIS projection and render are out of scope.

**Tech Stack:** Python ≥3.10, Pydantic v2 (configs/manifests), httpx + aiofiles (async download), pytest. Reuses `geoportal.http.RetryPolicy`.

## Global Constraints

- Stage `run()` methods return `(exit_code, artifact_path)` and write exactly one JSON manifest. Exit codes: `0` success, `1` policy/data failure.
- New config fields must default-resolve cleanly when missing.
- ODE base URL: `https://oderest.rsl.wustl.edu/live2`. Query params (verbatim): `query=product`, `target=moon`, `ihid=LRO`, `iid=LROC`, `pt=<code>`, `output=JSON`, `westernlon`/`easternlon`/`minlat`/`maxlat`, `loc=f`, `minobtime`/`maxobtime`, `results=opmf`.
- Default NAC product type: `CDRNAC4`. Default provider CRS: `IAU_2015:30100` (Moon geographic, ocentric lon/lat degrees).
- ODE/PDS are public services — keep concurrency modest and reuse the existing RetryPolicy + pre-request jitter.
- All new tests must run offline against checked-in fixtures (no network in the test suite).

---

### Task 1: ODE REST client — URL builder + JSON parser (`ode.py`)

Pure functions only (no network) so they test against a fixture. Models ODE's JSON shape: `ODEResults.Products.Product` is a list, OR a single dict when one result; `Product_files.Product_file` likewise.

**Files:**
- Create: `src/satmap_dataset/providers/lroc_nac/__init__.py` (empty for now)
- Create: `src/satmap_dataset/providers/lroc_nac/ode.py`
- Create: `tests/fixtures/lroc_nac/ode_search_two_years.json`
- Test: `tests/test_lroc_nac_ode_parse.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) OdeProduct(pdsid: str, observation_time: str | None, acquisition_year: int | None, incidence_angle: float | None, emission_angle: float | None, map_resolution: float | None, footprint_bbox: tuple[float,float,float,float] | None, file_url: str | None, file_bytes: int | None)`
  - `build_query_url(base: str, *, product_type: str, westlon: float, eastlon: float, minlat: float, maxlat: float, loc: str = "f", min_obtime: str | None = None, max_obtime: str | None = None, results: str = "opmf", limit: int = 100, offset: int = 0) -> str`
  - `parse_products(payload: dict) -> list[OdeProduct]`
  - `group_products_by_year(products: Iterable[OdeProduct]) -> dict[int, list[OdeProduct]]`

- [ ] **Step 1: Write the fixture**

Create `tests/fixtures/lroc_nac/ode_search_two_years.json`:

```json
{
  "ODEResults": {
    "Count": "2",
    "Status": "success",
    "Products": {
      "Product": [
        {
          "pdsid": "M101013931LC",
          "ihid": "LRO", "iid": "LROC", "pt": "CDRNAC4",
          "Observation_time": "2009-09-15T12:00:00",
          "UTC_start_time": "2009-09-15T12:00:01",
          "Incidence_angle": "42.5",
          "Emission_angle": "1.2",
          "Map_resolution": "0.5",
          "Westernmost_longitude": "30.60", "Easternmost_longitude": "30.72",
          "Minimum_latitude": "20.05", "Maximum_latitude": "20.30",
          "Product_files": {
            "Product_file": [
              {"FileName": "M101013931LC.IMG", "URL": "https://pds.example/M101013931LC.IMG", "Type": "Product", "KBytes": "51200"},
              {"FileName": "M101013931LC.browse.jpg", "URL": "https://pds.example/M101013931LC.jpg", "Type": "Browse", "KBytes": "40"}
            ]
          }
        },
        {
          "pdsid": "M198273648LC",
          "ihid": "LRO", "iid": "LROC", "pt": "CDRNAC4",
          "Observation_time": "2012-04-03T08:15:00",
          "Incidence_angle": "44.1",
          "Map_resolution": "0.52",
          "Westernmost_longitude": "30.61", "Easternmost_longitude": "30.73",
          "Minimum_latitude": "20.06", "Maximum_latitude": "20.31",
          "Product_files": {
            "Product_file": {"FileName": "M198273648LC.IMG", "URL": "https://pds.example/M198273648LC.IMG", "Type": "Product", "KBytes": "49000"}
          }
        }
      ]
    }
  }
}
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_lroc_nac_ode_parse.py`:

```python
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.providers.lroc_nac.ode import (
    OdeProduct,
    build_query_url,
    group_products_by_year,
    parse_products,
)

FIXTURES = ROOT / "tests" / "fixtures" / "lroc_nac"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_build_query_url_has_required_params() -> None:
    url = build_query_url(
        "https://oderest.rsl.wustl.edu/live2",
        product_type="CDRNAC4",
        westlon=30.6, eastlon=30.9, minlat=20.0, maxlat=20.35,
        min_obtime="2009-01-01", max_obtime="2026-12-31",
    )
    for fragment in (
        "query=product", "target=moon", "ihid=LRO", "iid=LROC",
        "pt=CDRNAC4", "westernlon=30.6", "easternlon=30.9",
        "minlat=20.0", "maxlat=20.35", "loc=f",
        "minobtime=2009-01-01", "maxobtime=2026-12-31",
        "results=opmf", "output=JSON",
    ):
        assert fragment in url


def test_parse_products_two_years() -> None:
    products = parse_products(_load("ode_search_two_years.json"))
    assert len(products) == 2
    first = products[0]
    assert isinstance(first, OdeProduct)
    assert first.pdsid == "M101013931LC"
    assert first.acquisition_year == 2009
    assert first.incidence_angle == 42.5
    assert first.map_resolution == 0.5
    assert first.file_url == "https://pds.example/M101013931LC.IMG"
    assert first.file_bytes == 51200 * 1024
    assert first.footprint_bbox == (30.60, 20.05, 30.72, 20.30)
    # Single-object Product_file (not a list) still resolves a URL:
    assert products[1].file_url == "https://pds.example/M198273648LC.IMG"


def test_group_products_by_year() -> None:
    grouped = group_products_by_year(parse_products(_load("ode_search_two_years.json")))
    assert sorted(grouped.keys()) == [2009, 2012]
    assert len(grouped[2009]) == 1 and len(grouped[2012]) == 1
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_lroc_nac_ode_parse.py -v`
Expected: FAIL — `ModuleNotFoundError: ... lroc_nac.ode`.

- [ ] **Step 4: Write the implementation**

Create `src/satmap_dataset/providers/lroc_nac/__init__.py`:

```python
from __future__ import annotations
```

Create `src/satmap_dataset/providers/lroc_nac/ode.py`:

```python
"""Minimal ODE (Orbital Data Explorer) REST client for LROC NAC products.

Network code lives in `search_products`; everything else is pure parsing so
tests run against fixture JSON without hitting the network. ODE returns
`ODEResults.Products.Product` as a list, or a bare dict when a single product
matches — the parser normalizes both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import Any, Iterable, Sequence
from urllib.parse import urlencode

import httpx

from satmap_dataset.geoportal.http import RetryPolicy

logger = logging.getLogger("satmap_dataset.lroc_nac.ode")


@dataclass(frozen=True)
class OdeProduct:
    pdsid: str
    observation_time: str | None
    acquisition_year: int | None
    incidence_angle: float | None
    emission_angle: float | None
    map_resolution: float | None
    footprint_bbox: tuple[float, float, float, float] | None
    file_url: str | None
    file_bytes: int | None


@dataclass
class OdeSearchOptions:
    url: str = "https://oderest.rsl.wustl.edu/live2"
    product_type: str = "CDRNAC4"
    loc: str = "f"
    results: str = "opmf"
    limit: int = 100
    max_pages: int = 20


def build_query_url(
    base: str,
    *,
    product_type: str,
    westlon: float,
    eastlon: float,
    minlat: float,
    maxlat: float,
    loc: str = "f",
    min_obtime: str | None = None,
    max_obtime: str | None = None,
    results: str = "opmf",
    limit: int = 100,
    offset: int = 0,
) -> str:
    params: list[tuple[str, str]] = [
        ("query", "product"),
        ("target", "moon"),
        ("ihid", "LRO"),
        ("iid", "LROC"),
        ("pt", product_type),
        ("westernlon", _fmt(westlon)),
        ("easternlon", _fmt(eastlon)),
        ("minlat", _fmt(minlat)),
        ("maxlat", _fmt(maxlat)),
        ("loc", loc),
        ("results", results),
        ("limit", str(limit)),
        ("offset", str(offset)),
        ("output", "JSON"),
    ]
    if min_obtime:
        params.append(("minobtime", min_obtime))
    if max_obtime:
        params.append(("maxobtime", max_obtime))
    return f"{base}?{urlencode(params)}"


def _fmt(value: float) -> str:
    # Avoid trailing ".0" noise but keep fractional precision (e.g. 30.6, 20.0).
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text or "0"


def _to_float(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _parse_year(value: str | None) -> int | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).year
    except ValueError:
        try:
            return datetime.strptime(value[:10], "%Y-%m-%d").year
        except ValueError:
            return None


def _select_file(record: dict[str, Any]) -> tuple[str | None, int | None]:
    files_block = record.get("Product_files") or {}
    raw_files = _as_list(files_block.get("Product_file")) if isinstance(files_block, dict) else []
    # Prefer Type == "Product" with an image extension; fall back to first Product.
    best: dict[str, Any] | None = None
    for entry in raw_files:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("Type", "")).lower() != "product":
            continue
        name = str(entry.get("FileName", "")).lower()
        if name.endswith((".img", ".tif", ".tiff", ".cub")):
            best = entry
            break
        if best is None:
            best = entry
    if best is None:
        return None, None
    url = best.get("URL")
    kbytes = _to_float(best.get("KBytes"))
    file_bytes = int(kbytes * 1024) if kbytes is not None else None
    return (str(url) if url else None), file_bytes


def _footprint_bbox(record: dict[str, Any]) -> tuple[float, float, float, float] | None:
    west = _to_float(record.get("Westernmost_longitude"))
    east = _to_float(record.get("Easternmost_longitude"))
    south = _to_float(record.get("Minimum_latitude"))
    north = _to_float(record.get("Maximum_latitude"))
    if None in (west, east, south, north):
        return None
    return (west, south, east, north)  # type: ignore[return-value]


def parse_product(record: dict[str, Any]) -> OdeProduct:
    obs_time = record.get("Observation_time") or record.get("UTC_start_time")
    obs_time = str(obs_time) if obs_time else None
    url, file_bytes = _select_file(record)
    return OdeProduct(
        pdsid=str(record.get("pdsid") or ""),
        observation_time=obs_time,
        acquisition_year=_parse_year(obs_time),
        incidence_angle=_to_float(record.get("Incidence_angle")),
        emission_angle=_to_float(record.get("Emission_angle")),
        map_resolution=_to_float(record.get("Map_resolution")),
        footprint_bbox=_footprint_bbox(record),
        file_url=url,
        file_bytes=file_bytes,
    )


def parse_products(payload: dict[str, Any]) -> list[OdeProduct]:
    results = payload.get("ODEResults") or {}
    products_block = results.get("Products") or {}
    if not isinstance(products_block, dict):
        return []
    records = _as_list(products_block.get("Product"))
    return [parse_product(r) for r in records if isinstance(r, dict)]


def group_products_by_year(products: Iterable[OdeProduct]) -> dict[int, list[OdeProduct]]:
    grouped: dict[int, list[OdeProduct]] = {}
    for product in products:
        if product.acquisition_year is None or product.file_url is None:
            continue
        grouped.setdefault(product.acquisition_year, []).append(product)
    return grouped


async def search_products(
    options: OdeSearchOptions,
    *,
    westlon: float,
    eastlon: float,
    minlat: float,
    maxlat: float,
    min_obtime: str | None,
    max_obtime: str | None,
    timeout: float = 60.0,
    retry_policy: RetryPolicy | None = None,
    client: httpx.AsyncClient | None = None,
) -> list[OdeProduct]:
    """Page ODE by offset until a short page or `max_pages`. Returns parsed products."""
    owns_client = client is None
    active = client or httpx.AsyncClient(timeout=timeout, headers={"User-Agent": "satmap_dataset/0.1"})
    policy = retry_policy or RetryPolicy()
    all_products: list[OdeProduct] = []
    try:
        for page in range(options.max_pages):
            url = build_query_url(
                options.url,
                product_type=options.product_type,
                westlon=westlon, eastlon=eastlon, minlat=minlat, maxlat=maxlat,
                loc=options.loc, min_obtime=min_obtime, max_obtime=max_obtime,
                results=options.results, limit=options.limit, offset=page * options.limit,
            )
            logger.info("ODE GET %s", url)
            response = None
            for attempt in range(1, policy.max_attempts + 1):
                try:
                    response = await active.get(url)
                    if response.status_code in policy.retry_for_statuses and attempt < policy.max_attempts:
                        continue
                    response.raise_for_status()
                    break
                except httpx.HTTPError:
                    if attempt >= policy.max_attempts:
                        raise
            assert response is not None
            products = parse_products(response.json())
            all_products.extend(products)
            if len(products) < options.limit:
                break
        return all_products
    finally:
        if owns_client:
            await active.aclose()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_lroc_nac_ode_parse.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add src/satmap_dataset/providers/lroc_nac/__init__.py src/satmap_dataset/providers/lroc_nac/ode.py tests/fixtures/lroc_nac/ode_search_two_years.json tests/test_lroc_nac_ode_parse.py
git commit -m "feat(lroc-nac): ODE REST client — URL builder + product parser"
```

---

### Task 2: Lunar CRS bbox normalization (`crs.py`)

Convert the request bbox into ODE's planetocentric lon/lat order `(westlon, eastlon, minlat, maxlat)`. For `IAU_2015:30100` the bbox is already lon/lat degrees in `xmin,ymin,xmax,ymax` order; a projected lunar CRS is converted corner-wise via pyproj.

**Files:**
- Create: `src/satmap_dataset/providers/lroc_nac/crs.py`
- Test: `tests/test_lroc_nac_crs.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `normalize_bbox_to_ode(bbox: str, srs: str) -> tuple[float, float, float, float]` returning `(westlon, eastlon, minlat, maxlat)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_lroc_nac_crs.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.providers.lroc_nac.crs import normalize_bbox_to_ode


def test_geographic_bbox_passthrough_reorders_to_ode() -> None:
    # bbox is xmin,ymin,xmax,ymax = westlon,minlat,eastlon,maxlat
    west, east, minlat, maxlat = normalize_bbox_to_ode(
        "30.60,20.00,30.90,20.35", "IAU_2015:30100"
    )
    assert (west, east, minlat, maxlat) == (30.60, 20.00, 30.90, 20.35)


def test_rejects_non_lunar_crs() -> None:
    with pytest.raises(ValueError, match="lunar"):
        normalize_bbox_to_ode("30.6,20.0,30.9,20.35", "EPSG:2180")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lroc_nac_crs.py -v`
Expected: FAIL — module/function missing.

- [ ] **Step 3: Write the implementation**

Create `src/satmap_dataset/providers/lroc_nac/crs.py`:

```python
"""CRS helpers for the LROC NAC provider.

ODE expects planetocentric lon/lat degrees. For the geographic lunar CRS
(`IAU_2015:30100`) the request bbox is already lon/lat and only needs
reordering. A projected lunar CRS (equirectangular/sinusoidal/polar) is
converted corner-wise via pyproj's IAU_2015 authority.
"""

from __future__ import annotations

_GEOGRAPHIC_LUNAR = "IAU_2015:30100"


def _is_lunar(srs: str) -> bool:
    return srs.upper().startswith("IAU_2015:301")


def normalize_bbox_to_ode(bbox: str, srs: str) -> tuple[float, float, float, float]:
    """Return (westlon, eastlon, minlat, maxlat) in degrees for ODE."""
    if not _is_lunar(srs):
        raise ValueError(
            f"lroc_nac requires a lunar IAU_2015:301xx CRS; got srs={srs!r}."
        )
    parts = [float(p.strip()) for p in bbox.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must have format xmin,ymin,xmax,ymax")
    xmin, ymin, xmax, ymax = parts
    if srs.upper() == _GEOGRAPHIC_LUNAR:
        return (xmin, xmax, ymin, ymax)
    # Projected lunar CRS: convert the four corners to geographic lon/lat.
    from pyproj import Transformer

    transformer = Transformer.from_crs(srs, _GEOGRAPHIC_LUNAR, always_xy=True)
    lons: list[float] = []
    lats: list[float] = []
    for x, y in ((xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)):
        lon, lat = transformer.transform(x, y)
        lons.append(float(lon))
        lats.append(float(lat))
    return (min(lons), max(lons), min(lats), max(lats))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_lroc_nac_crs.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/providers/lroc_nac/crs.py tests/test_lroc_nac_crs.py
git commit -m "feat(lroc-nac): lunar CRS bbox normalization for ODE"
```

---

### Task 3: Provider registration + config validation + skeleton class

Wires `lroc_nac` into the config allow-list, CRS validation, and the provider registry. `index()`/`download()` are stubs here (filled in Tasks 4–5) so the registry/config tests pass independently.

**Files:**
- Modify: `src/satmap_dataset/config.py:9-18` (provider constants), `:40-46` (`_validate_provider_srs`)
- Create: `src/satmap_dataset/providers/lroc_nac/provider.py`
- Modify: `src/satmap_dataset/providers/lroc_nac/__init__.py`
- Modify: `src/satmap_dataset/providers/__init__.py:8-23` (`get_provider`)
- Test: `tests/test_lroc_nac_config.py`, extend `tests/test_providers_selection.py`

**Interfaces:**
- Consumes: `ode` and `crs` modules from Tasks 1–2.
- Produces:
  - `config.PROVIDER_LROC_NAC = "lroc_nac"` added to `ALLOWED_PROVIDERS`.
  - `LrocNacProvider(Provider)` with `name = "lroc_nac"`, `default_target_srs = "IAU_2015:30100"`, and `index`/`download` methods (stubs this task).
  - `get_provider("lroc_nac") -> LrocNacProvider`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_lroc_nac_config.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.config import IndexConfig


def test_lroc_nac_accepts_lunar_crs() -> None:
    cfg = IndexConfig(
        year_start=2009, year_end=2026,
        bbox="30.6,20.0,30.9,20.35", srs="IAU_2015:30100",
        provider="lroc_nac",
    )
    assert cfg.provider == "lroc_nac"


def test_lroc_nac_rejects_earth_crs() -> None:
    with pytest.raises(ValueError, match="lunar"):
        IndexConfig(
            year_start=2009, year_end=2026,
            bbox="30.6,20.0,30.9,20.35", srs="EPSG:2180",
            provider="lroc_nac",
        )
```

Append to `tests/test_providers_selection.py`:

```python
def test_get_provider_returns_lroc_nac() -> None:
    from satmap_dataset.providers.lroc_nac import LrocNacProvider

    provider = get_provider("lroc_nac")
    assert isinstance(provider, LrocNacProvider)
    assert isinstance(provider, Provider)
    assert provider.name == "lroc_nac"
    assert provider.default_target_srs == "IAU_2015:30100"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_lroc_nac_config.py tests/test_providers_selection.py -v`
Expected: FAIL — `lroc_nac` not in `ALLOWED_PROVIDERS`; `LrocNacProvider` missing.

- [ ] **Step 3: Update config.py**

In `src/satmap_dataset/config.py`, add the constant beside the others (after line 12) and include it in `ALLOWED_PROVIDERS`:

```python
PROVIDER_LROC_NAC = "lroc_nac"
ALLOWED_PROVIDERS = {
    PROVIDER_GEOPORTAL,
    PROVIDER_LANTMATERIET,
    PROVIDER_SENTINEL2,
    PROVIDER_NLS,
    PROVIDER_LROC_NAC,
}
```

Extend `_validate_provider_srs` (after the existing `nls` check):

```python
def _validate_provider_srs(provider: str, srs: str) -> None:
    if provider == "nls" and srs.upper() != _NLS_NATIVE_SRS:
        raise ValueError(
            f"provider='nls' requires srs='{_NLS_NATIVE_SRS}' (NLS WCS/OAPIF "
            f"only accept TM35FIN coordinates); got srs={srs!r}. "
            "Reproject your bbox to EPSG:3067 before configuring an NLS run."
        )
    if provider == "lroc_nac" and not srs.upper().startswith("IAU_2015:301"):
        raise ValueError(
            f"provider='lroc_nac' requires a lunar IAU_2015:301xx CRS "
            f"(e.g. 'IAU_2015:30100'); got srs={srs!r}."
        )
```

- [ ] **Step 4: Create the skeleton provider**

Create `src/satmap_dataset/providers/lroc_nac/provider.py`:

```python
"""LROC NAC (Moon) multi-temporal provider — ODE index + download.

Sources lunar NAC observations from the PDS Orbital Data Explorer REST API.
Index enumerates every overlapping NAC observation across a lat/lon bbox and
date range; download pulls the PDS frames. Map projection (ISIS cam2map) and
render are intentionally out of scope for this provider.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from satmap_dataset.config import DownloadConfig, IndexConfig
from satmap_dataset.providers.base import Provider

logger = logging.getLogger("satmap_dataset.lroc_nac")

DEFAULT_TARGET_SRS = "IAU_2015:30100"


class LrocNacProvider(Provider):
    name = "lroc_nac"
    default_target_srs = DEFAULT_TARGET_SRS

    def index(self, config: IndexConfig) -> tuple[int, Path]:
        return asyncio.run(self._index_async(config))

    async def _index_async(self, config: IndexConfig) -> tuple[int, Path]:
        raise NotImplementedError("Implemented in Task 4")

    def download(self, config: DownloadConfig) -> tuple[int, Path]:
        return asyncio.run(self._download_async(config))

    async def _download_async(self, config: DownloadConfig) -> tuple[int, Path]:
        raise NotImplementedError("Implemented in Task 5")
```

Replace `src/satmap_dataset/providers/lroc_nac/__init__.py` contents:

```python
from __future__ import annotations

from satmap_dataset.providers.lroc_nac.provider import LrocNacProvider

__all__ = ["LrocNacProvider"]
```

- [ ] **Step 5: Register in `get_provider`**

In `src/satmap_dataset/providers/__init__.py`, add before the `raise`:

```python
    if name == "lroc_nac":
        from satmap_dataset.providers.lroc_nac import LrocNacProvider

        return LrocNacProvider()
```

And update the error message to include the new name:

```python
    raise ValueError(
        f"Unknown provider: {name!r}. Expected 'geoportal', 'lantmateriet', "
        "'sentinel2', or 'lroc_nac'."
    )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_lroc_nac_config.py tests/test_providers_selection.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/satmap_dataset/config.py src/satmap_dataset/providers/__init__.py src/satmap_dataset/providers/lroc_nac/ tests/test_lroc_nac_config.py tests/test_providers_selection.py
git commit -m "feat(lroc-nac): register provider + lunar CRS config validation"
```

---

### Task 4: `index()` — ODE search → IndexManifest

Builds the year-keyed index manifest from ODE results. Tests monkeypatch `ode.search_products` so no network is touched.

**Files:**
- Modify: `src/satmap_dataset/providers/lroc_nac/provider.py`
- Test: `tests/test_lroc_nac_index.py`

**Interfaces:**
- Consumes: `ode.OdeProduct`, `ode.OdeSearchOptions`, `ode.group_products_by_year`, `ode.search_products`; `crs.normalize_bbox_to_ode`; `pipeline.validator.evaluate_year_policy`; manifest models `IndexManifest`, `YearAvailabilityReport`, `YearStatus`, `TileAcquisitionMetadata`.
- Produces: `LrocNacProvider._index_async` writing `config.output_json` + `config.year_availability_output_json`, returning `(0|1, output_json)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_lroc_nac_index.py`:

```python
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.config import IndexConfig
from satmap_dataset.providers.lroc_nac import provider as lroc_provider
from satmap_dataset.providers.lroc_nac.ode import OdeProduct


def _products() -> list[OdeProduct]:
    return [
        OdeProduct("M101013931LC", "2009-09-15T12:00:00", 2009, 42.5, 1.2, 0.5,
                   (30.60, 20.05, 30.72, 20.30), "https://pds.example/a.IMG", 51200000),
        OdeProduct("M198273648LC", "2012-04-03T08:15:00", 2012, 44.1, None, 0.52,
                   (30.61, 20.06, 30.73, 20.31), "https://pds.example/b.IMG", 49000000),
    ]


def test_index_builds_multitemporal_manifest(tmp_path, monkeypatch) -> None:
    async def fake_search(options, **kwargs):
        return _products()

    monkeypatch.setattr(lroc_provider.ode, "search_products", fake_search)

    out = tmp_path / "index.json"
    avail = tmp_path / "avail.json"
    cfg = IndexConfig(
        year_start=2009, year_end=2026,
        bbox="30.6,20.0,30.9,20.35", srs="IAU_2015:30100",
        provider="lroc_nac", min_years=2,
        output_json=out, year_availability_output_json=avail,
    )

    code, path = lroc_provider.LrocNacProvider().index(cfg)

    assert code == 0
    assert path == out
    manifest = json.loads(out.read_text())
    assert manifest["provider"] == "lroc_nac"
    assert manifest["years_included"] == [2009, 2012]
    assert manifest["tile_sources_by_year"]["2009"]["M101013931LC"] == "https://pds.example/a.IMG"
    assert avail.exists()


def test_index_fails_when_below_min_years(tmp_path, monkeypatch) -> None:
    async def fake_search(options, **kwargs):
        return _products()[:1]  # only 2009

    monkeypatch.setattr(lroc_provider.ode, "search_products", fake_search)

    cfg = IndexConfig(
        year_start=2009, year_end=2026,
        bbox="30.6,20.0,30.9,20.35", srs="IAU_2015:30100",
        provider="lroc_nac", min_years=2,
        output_json=tmp_path / "i.json",
        year_availability_output_json=tmp_path / "a.json",
    )
    code, _ = lroc_provider.LrocNacProvider().index(cfg)
    assert code == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lroc_nac_index.py -v`
Expected: FAIL — `NotImplementedError`.

- [ ] **Step 3: Implement `_index_async`**

In `src/satmap_dataset/providers/lroc_nac/provider.py`, add imports and replace `_index_async`. Add at the top with the other imports:

```python
from typing import Any

from satmap_dataset.models import (
    IndexManifest,
    TileAcquisitionMetadata,
    YearAvailabilityReport,
    YearStatus,
)
from satmap_dataset.pipeline.validator import evaluate_year_policy
from satmap_dataset.providers.lroc_nac import crs, ode
```

Replace the `_index_async` stub:

```python
    async def _index_async(self, config: IndexConfig) -> tuple[int, Path]:
        options = dict(config.provider_options)
        search_options = ode.OdeSearchOptions(
            url=str(options.get("ode_url", ode.OdeSearchOptions.url)),
            product_type=str(options.get("product_type", "CDRNAC4")),
            loc=str(options.get("loc", "f")),
            limit=int(options.get("page_limit", 100)),
            max_pages=int(options.get("max_pages", 20)),
        )
        westlon, eastlon, minlat, maxlat = crs.normalize_bbox_to_ode(config.bbox, config.srs)
        min_obtime = str(options.get("min_obtime", f"{config.year_start}-01-01"))
        max_obtime = str(options.get("max_obtime", f"{config.year_end}-12-31"))

        warnings: list[str] = []
        errors: list[str] = []
        try:
            products = await ode.search_products(
                search_options,
                westlon=westlon, eastlon=eastlon, minlat=minlat, maxlat=maxlat,
                min_obtime=min_obtime, max_obtime=max_obtime,
                timeout=float(options.get("timeout", 60.0)),
            )
        except Exception as exc:  # noqa: BLE001 — surface transport errors in manifest
            products = []
            errors.append(f"ODE search failed: {exc}")

        max_incidence = options.get("max_incidence_angle")
        if max_incidence is not None:
            limit = float(max_incidence)
            products = [
                p for p in products
                if p.incidence_angle is None or p.incidence_angle <= limit
            ]

        grouped = ode.group_products_by_year(products)
        years_available = sorted(grouped.keys())

        tile_sources_by_year: dict[int, dict[str, str]] = {}
        tile_bboxes_by_year: dict[int, dict[str, list[float]]] = {}
        tile_acquisition_by_year: dict[int, dict[str, TileAcquisitionMetadata]] = {}
        year_statuses: list[YearStatus] = []
        years_excluded: dict[int, str] = {}

        for year in config.requested_years:
            items = grouped.get(year, [])
            if not items:
                year_statuses.append(
                    YearStatus(year=year, typename_exists=year in years_available,
                               feature_count=0, status="zero_features",
                               reason="no_nac_observation")
                )
                years_excluded[year] = "no_nac_observation"
                continue
            sources: dict[str, str] = {}
            bboxes: dict[str, list[float]] = {}
            acquisition: dict[str, TileAcquisitionMetadata] = {}
            for product in items:
                if product.file_url is None:
                    continue
                sources[product.pdsid] = product.file_url
                if product.footprint_bbox is not None:
                    bboxes[product.pdsid] = list(product.footprint_bbox)
                acquisition[product.pdsid] = TileAcquisitionMetadata(
                    acquisition_date=product.observation_time,
                    publication_date=None,
                    acquisition_year=year,
                )
            tile_sources_by_year[year] = sources
            if bboxes:
                tile_bboxes_by_year[year] = bboxes
            tile_acquisition_by_year[year] = acquisition
            year_statuses.append(
                YearStatus(year=year, typename_exists=True,
                           feature_count=len(sources), status="has_features", reason=None)
            )

        years_included = sorted(tile_sources_by_year.keys())
        policy = evaluate_year_policy(
            requested_years=config.requested_years,
            available_years=years_included,
            strict_years=config.strict_years,
            min_years=config.min_years,
        )
        if not years_included and not errors:
            errors.append("ODE returned no NAC observations for the requested bbox/date range.")

        provider_metadata: dict[str, Any] = {
            "ode_url": search_options.url,
            "product_type": search_options.product_type,
            "loc": search_options.loc,
            "min_obtime": min_obtime,
            "max_obtime": max_obtime,
            "available_years": years_available,
            "observations_per_year": {y: len(grouped[y]) for y in years_available},
            "incidence_by_tile": {
                p.pdsid: p.incidence_angle for p in products if p.incidence_angle is not None
            },
            "map_resolution_by_tile": {
                p.pdsid: p.map_resolution for p in products if p.map_resolution is not None
            },
        }

        manifest = IndexManifest(
            provider="lroc_nac",
            year_start=config.year_start, year_end=config.year_end,
            bbox=config.bbox, srs=config.srs,
            strict_years=config.strict_years, min_years=config.min_years,
            wfs_bbox_axes_swapped=False,
            years_requested=config.requested_years,
            year_statuses=year_statuses,
            years_available_wfs=years_available,
            years_included=years_included,
            years_excluded_with_reason=years_excluded,
            common_tile_ids=[],
            tile_sources_by_year=tile_sources_by_year,
            tile_bboxes_by_year=tile_bboxes_by_year,
            tile_acquisition_by_year=tile_acquisition_by_year,
            passed=policy.passed and bool(years_included),
            errors=list(errors) + list(policy.errors),
            warnings=list(warnings) + list(policy.warnings),
            run_parameters=config.model_dump(mode="json"),
            provider_metadata=provider_metadata,
        )
        availability = YearAvailabilityReport(
            year_start=config.year_start, year_end=config.year_end,
            bbox=config.bbox, srs=config.srs, wfs_bbox_axes_swapped=False,
            years_requested=manifest.years_requested,
            year_statuses=manifest.year_statuses,
            years_available_wfs=manifest.years_available_wfs,
            years_included=manifest.years_included,
            years_excluded_with_reason=manifest.years_excluded_with_reason,
            strict_years=manifest.strict_years, min_years=manifest.min_years,
            passed=manifest.passed, errors=manifest.errors, warnings=manifest.warnings,
            run_parameters=manifest.run_parameters,
        )

        config.output_json.parent.mkdir(parents=True, exist_ok=True)
        config.year_availability_output_json.parent.mkdir(parents=True, exist_ok=True)
        config.year_availability_output_json.write_text(
            availability.model_dump_json(indent=2), encoding="utf-8"
        )
        config.output_json.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        logger.info(
            "LROC NAC index: years_included=%s available=%s passed=%s",
            years_included, years_available, manifest.passed,
        )
        return (0 if manifest.passed else 1), config.output_json
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_lroc_nac_index.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/providers/lroc_nac/provider.py tests/test_lroc_nac_index.py
git commit -m "feat(lroc-nac): index() — ODE search to year-keyed manifest"
```

---

### Task 5: `download()` — fetch PDS frames → LayerManifest

Pulls each `(year, pdsid, url)` to `download_root/<year>/<pdsid><ext>` using the async + RetryPolicy + jitter pattern from the lantmateriet downloader. Tests monkeypatch the per-asset fetch so no network is touched.

**Files:**
- Modify: `src/satmap_dataset/providers/lroc_nac/provider.py`
- Test: `tests/test_lroc_nac_download.py`

**Interfaces:**
- Consumes: `IndexManifest`, `LayerManifest` from `satmap_dataset.models`; the index manifest produced by Task 4.
- Produces: `LrocNacProvider._download_async` writing `config.output_json`, returning `(0|1, output_json)`. Internal helper `_download_asset_with_retry(client, url, output_path, *, retries, retry_delay, sleep_min, sleep_max) -> bool`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_lroc_nac_download.py`:

```python
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.config import DownloadConfig
from satmap_dataset.models import IndexManifest
from satmap_dataset.providers.lroc_nac import provider as lroc_provider


def _write_index(path: Path) -> None:
    manifest = IndexManifest(
        provider="lroc_nac", year_start=2009, year_end=2012,
        bbox="30.6,20.0,30.9,20.35", srs="IAU_2015:30100",
        strict_years=False, min_years=1, wfs_bbox_axes_swapped=False,
        years_requested=[2009, 2010, 2011, 2012], year_statuses=[],
        years_available_wfs=[2009, 2012], years_included=[2009, 2012],
        years_excluded_with_reason={}, common_tile_ids=[],
        tile_sources_by_year={
            2009: {"M101013931LC": "https://pds.example/a.IMG"},
            2012: {"M198273648LC": "https://pds.example/b.IMG"},
        },
        tile_bboxes_by_year={}, tile_acquisition_by_year={},
        passed=True, errors=[], warnings=[], run_parameters={}, provider_metadata={},
    )
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def test_download_writes_assets_and_manifest(tmp_path, monkeypatch) -> None:
    index_path = tmp_path / "index.json"
    _write_index(index_path)

    async def fake_fetch(client, url, output_path, **kwargs):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"FAKE")
        return True

    monkeypatch.setattr(lroc_provider, "_download_asset_with_retry", fake_fetch)

    cfg = DownloadConfig(
        index_manifest=index_path, download_root=tmp_path / "dl",
        srs="IAU_2015:30100", provider="lroc_nac", bbox="30.6,20.0,30.9,20.35",
        wms_fallback_missing_years=False, output_json=tmp_path / "out.json",
    )
    code, path = lroc_provider.LrocNacProvider().download(cfg)

    assert code == 0
    assert (tmp_path / "dl" / "2009" / "M101013931LC.IMG").read_bytes() == b"FAKE"
    assert (tmp_path / "dl" / "2012" / "M198273648LC.IMG").exists()
    manifest = json.loads(path.read_text())
    assert manifest["provider"] == "lroc_nac"
    assert sorted(manifest["years_included"]) == [2009, 2012]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lroc_nac_download.py -v`
Expected: FAIL — `NotImplementedError` / `_download_asset_with_retry` missing.

- [ ] **Step 3: Implement download path**

In `src/satmap_dataset/providers/lroc_nac/provider.py`, add imports:

```python
import asyncio
import random
from urllib.parse import urlparse

import aiofiles
import httpx

from satmap_dataset.models import IndexManifest, LayerManifest
```

Add the module-level helper and constant (above the class):

```python
_NON_RETRYABLE_STATUSES = frozenset({400, 401, 403, 404, 410})


async def _download_asset_with_retry(
    client: httpx.AsyncClient,
    url: str,
    output_path: Path,
    *,
    retries: int,
    retry_delay: float,
    sleep_min: float,
    sleep_max: float,
) -> bool:
    attempts = retries + 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        await asyncio.sleep(random.uniform(sleep_min, sleep_max))
        try:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                async with aiofiles.open(output_path, "wb") as handle:
                    async for chunk in response.aiter_bytes():
                        await handle.write(chunk)
            return True
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in _NON_RETRYABLE_STATUSES:
                return False
            if attempt >= attempts:
                return False
            await asyncio.sleep(retry_delay * (2 ** (attempt - 1)))
        except (httpx.HTTPError, OSError):
            if attempt >= attempts:
                return False
            await asyncio.sleep(retry_delay * (2 ** (attempt - 1)))
    return False


def _ext_for_url(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix
    return suffix or ".IMG"
```

Replace the `_download_async` stub:

```python
    async def _download_async(self, config: DownloadConfig) -> tuple[int, Path]:
        index_manifest = IndexManifest.model_validate_json(
            config.index_manifest.read_text(encoding="utf-8")
        )
        jobs: list[tuple[int, str, str, Path]] = []
        for year in index_manifest.years_included:
            for pdsid, url in index_manifest.tile_sources_by_year.get(year, {}).items():
                output_path = config.download_root / str(year) / f"{pdsid}{_ext_for_url(url)}"
                jobs.append((year, pdsid, url, output_path))

        assets: list[str] = []
        failed: list[str] = []
        years_source_map: dict[int, str] = {}

        timeout = httpx.Timeout(timeout=config.timeout, connect=min(config.timeout, 20.0))
        limits = httpx.Limits(
            max_connections=config.concurrency, max_keepalive_connections=config.concurrency
        )
        headers = {"User-Agent": "satmap_dataset/0.1"}

        if jobs:
            queue: asyncio.Queue[tuple[int, str, str, Path] | None] = asyncio.Queue()
            for job in jobs:
                queue.put_nowait(job)
            lock = asyncio.Lock()

            async def worker() -> None:
                async with httpx.AsyncClient(
                    follow_redirects=True, timeout=timeout, limits=limits, headers=headers
                ) as client:
                    while True:
                        item = await queue.get()
                        if item is None:
                            queue.task_done()
                            return
                        year, _pdsid, url, output_path = item
                        ok = (
                            output_path.exists()
                            and output_path.stat().st_size > 0
                            and not config.overwrite
                        )
                        if not ok:
                            ok = await _download_asset_with_retry(
                                client, url, output_path,
                                retries=config.retries, retry_delay=config.retry_delay,
                                sleep_min=config.sleep_min, sleep_max=config.sleep_max,
                            )
                        async with lock:
                            if ok:
                                assets.append(str(output_path))
                                years_source_map[year] = "ode"
                            else:
                                failed.append(url)
                        queue.task_done()

            workers = [asyncio.create_task(worker()) for _ in range(max(1, config.concurrency))]
            await queue.join()
            for _ in workers:
                queue.put_nowait(None)
            await asyncio.gather(*workers)

        years_included_effective = sorted(years_source_map.keys())
        manifest = LayerManifest(
            layer="lroc_nac_mono",
            role="rgb",
            stage="download",
            provider="lroc_nac",
            years_requested=index_manifest.years_requested,
            years_available_wfs=index_manifest.years_available_wfs,
            years_included=years_included_effective,
            years_excluded_with_reason=index_manifest.years_excluded_with_reason,
            common_tile_ids=index_manifest.common_tile_ids,
            tile_sources_by_year=index_manifest.tile_sources_by_year,
            tile_bboxes_by_year=index_manifest.tile_bboxes_by_year,
            tile_acquisition_by_year=index_manifest.tile_acquisition_by_year,
            assets=sorted(set(assets)),
            source_manifest=str(config.index_manifest),
            mode="ode",
            target_bbox=config.bbox,
            target_srs=config.srs,
            profile=config.profile,
            px_per_meter=config.px_per_meter,
            years_source_map=years_source_map,
            forced_wms_years=[],
            passed=bool(assets) and not failed,
            notes=(
                f"provider=lroc_nac downloaded={len(assets)} failed={len(failed)} "
                f"years_included={years_included_effective}"
            ),
            run_parameters=config.model_dump(mode="json"),
            provider_metadata={"failed_urls": failed},
        )
        config.output_json.parent.mkdir(parents=True, exist_ok=True)
        config.output_json.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        logger.info(
            "LROC NAC download: assets=%s failed=%s passed=%s",
            len(assets), len(failed), manifest.passed,
        )
        return (0 if manifest.passed else 1), config.output_json
```

Remove the now-duplicate top-level `import asyncio`/`from pathlib import Path` if your editor flags them — keep a single import of each.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_lroc_nac_download.py -v`
Expected: PASS.

- [ ] **Step 5: Verify `LayerManifest.role="rgb"` is accepted**

Run: `pytest tests/test_lroc_nac_download.py -v` (already covers manifest construction). If `role` is enum-restricted and rejects, check `models.py` `LayerManifest.role` for allowed values and use the permitted one; re-run.

- [ ] **Step 6: Commit**

```bash
git add src/satmap_dataset/providers/lroc_nac/provider.py tests/test_lroc_nac_download.py
git commit -m "feat(lroc-nac): download() — fetch PDS frames to LayerManifest"
```

---

### Task 6: Sample config + docs + live smoke test

A checked-in sample config for Apollo 17 / Taurus-Littrow and a one-line invocation, plus a network-gated smoke test (skipped by default) and CLAUDE.md documentation. This is the "after adding, download samples" deliverable.

**Files:**
- Create: `configs/run/lroc_nac_apollo17.index.json`
- Create: `configs/run/lroc_nac_apollo17.download.json`
- Create: `tests/test_lroc_nac_live_smoke.py`
- Modify: `CLAUDE.md` (add an "LROC NAC provider" subsection under External services / providers)

**Interfaces:**
- Consumes: the full provider from Tasks 1–5 via the existing `index-json` / `download-json` CLI commands.

- [ ] **Step 1: Create the index sample config**

Create `configs/run/lroc_nac_apollo17.index.json`:

```json
{
  "year_start": 2009,
  "year_end": 2026,
  "bbox": "30.60,20.00,30.90,20.35",
  "srs": "IAU_2015:30100",
  "min_years": 2,
  "provider": "lroc_nac",
  "provider_options": {"product_type": "CDRNAC4", "page_limit": 100, "max_pages": 5},
  "output_json": "artifacts_lroc_nac_apollo17/index_manifest.json",
  "year_availability_output_json": "artifacts_lroc_nac_apollo17/year_availability_report.json"
}
```

- [ ] **Step 2: Create the download sample config**

Create `configs/run/lroc_nac_apollo17.download.json`:

```json
{
  "index_manifest": "artifacts_lroc_nac_apollo17/index_manifest.json",
  "download_root": "downloads_lroc_nac_apollo17",
  "srs": "IAU_2015:30100",
  "bbox": "30.60,20.00,30.90,20.35",
  "provider": "lroc_nac",
  "wms_fallback_missing_years": false,
  "concurrency": 4,
  "output_json": "artifacts_lroc_nac_apollo17/dataset_manifest_download.json"
}
```

- [ ] **Step 3: Write the network-gated smoke test**

Create `tests/test_lroc_nac_live_smoke.py`:

```python
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

pytestmark = pytest.mark.skipif(
    os.environ.get("SATMAP_LIVE_TESTS") != "1",
    reason="set SATMAP_LIVE_TESTS=1 to hit the live ODE API",
)


def test_live_ode_returns_multitemporal_nac(tmp_path) -> None:
    from satmap_dataset.config import IndexConfig
    from satmap_dataset.providers.lroc_nac import LrocNacProvider

    cfg = IndexConfig(
        year_start=2009, year_end=2026,
        bbox="30.60,20.00,30.90,20.35", srs="IAU_2015:30100",
        provider="lroc_nac", min_years=2,
        provider_options={"product_type": "CDRNAC4", "max_pages": 3},
        output_json=tmp_path / "index.json",
        year_availability_output_json=tmp_path / "avail.json",
    )
    code, path = LrocNacProvider().index(cfg)
    assert code == 0, "expected ≥2 distinct NAC years over Apollo 17"
```

- [ ] **Step 4: Run the offline suite (smoke test skips)**

Run: `pytest tests/test_lroc_nac_live_smoke.py -v`
Expected: SKIPPED (1 skipped).

- [ ] **Step 5: Run the live smoke test manually**

Run: `SATMAP_LIVE_TESTS=1 pytest tests/test_lroc_nac_live_smoke.py -v`
Expected: PASS — confirms real ODE returns ≥2 distinct NAC years over Apollo 17. (If ODE is unreachable, note it; do not commit a failing gate.)

- [ ] **Step 6: Download real samples end-to-end**

Run:
```bash
python -m satmap_dataset.cli index-json configs/run/lroc_nac_apollo17.index.json
python -m satmap_dataset.cli download-json configs/run/lroc_nac_apollo17.download.json
ls -R downloads_lroc_nac_apollo17 | head
```
Expected: NAC `.IMG` files under `downloads_lroc_nac_apollo17/<year>/`, spanning ≥2 years. (These dirs are runtime artifacts — confirm `downloads_*`/`artifacts_*` are gitignored; do not commit downloaded data.)

- [ ] **Step 7: Document in CLAUDE.md**

Add a subsection under the providers/External services area of `CLAUDE.md`:

```markdown
### LROC NAC provider (Moon, multi-temporal)

`provider="lroc_nac"` sources multi-temporal lunar LROC NAC observations from
the PDS Orbital Data Explorer (ODE) REST API
(`https://oderest.rsl.wustl.edu/live2`). Requires a lunar CRS
(`srs="IAU_2015:30100"`, ocentric lon/lat degrees). `index` enumerates every
overlapping NAC observation across the bbox + year range (each `pdsid` a
distinct tile under its acquisition year — the multi-temporal axis);
`download` pulls the PDS frames. `provider_options`: `product_type`
(default `CDRNAC4`), `page_limit`, `max_pages`, `max_incidence_angle`,
`min_obtime`/`max_obtime`. Downloaded frames are unprojected camera-geometry
rasters — ISIS `cam2map` projection and render are a separate, deferred stage.
Sample configs: `configs/run/lroc_nac_apollo17.{index,download}.json`.
```

- [ ] **Step 8: Commit**

```bash
git add configs/run/lroc_nac_apollo17.index.json configs/run/lroc_nac_apollo17.download.json tests/test_lroc_nac_live_smoke.py CLAUDE.md
git commit -m "feat(lroc-nac): sample Apollo 17 configs + live smoke test + docs"
```

---

### Task 7: Full suite green + CLI provider help

Confirm nothing regressed and the provider is visible in the CLI help/listing path used by `test_cli_provider_help.py`.

**Files:**
- Modify (if needed): `src/satmap_dataset/cli.py` (only if provider names are enumerated in help text)
- Test: existing `tests/test_cli_provider_help.py`

- [ ] **Step 1: Run the full suite**

Run: `pytest`
Expected: all pass (new tests included, live smoke skipped).

- [ ] **Step 2: Check CLI provider help**

Run: `pytest tests/test_cli_provider_help.py -v`
Expected: PASS. If it asserts the set of provider names in help text, add `lroc_nac` wherever that list is rendered in `cli.py`, then re-run.

- [ ] **Step 3: Commit any CLI help fix**

```bash
git add src/satmap_dataset/cli.py
git commit -m "feat(lroc-nac): surface provider in CLI help"
```

(Skip the commit if no `cli.py` change was needed.)

---

## Self-Review

**Spec coverage:**
- ODE client (URL builder, parser, search) → Task 1 ✓
- Lunar IAU CRS / bbox normalization → Task 2 ✓
- Provider registration + `_validate_provider_srs` for `lroc_nac` → Task 3 ✓
- `index()` year-keyed multi-temporal manifest, `evaluate_year_policy`, `min_years` gating → Task 4 ✓
- `download()` with RetryPolicy + jitter, non-retryable 4xx, LayerManifest → Task 5 ✓
- Sample Apollo 17 config + real sample download + live acceptance (≥2 years) → Task 6 ✓
- CLAUDE.md docs → Task 6 ✓
- Empty-result → `passed=false` exit 1 → Task 4 (`test_index_fails_when_below_min_years` + empty-error branch) ✓
- Out-of-scope (ISIS/render/other providers) → not implemented, documented ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code. The one runtime-confirm item (longitude convention, ODE `Product_files` dict-vs-list) is handled defensively in code (`_as_list`) and exercised by the fixture.

**Type consistency:** `OdeProduct` fields used identically across Tasks 1/4/5. `search_products` signature (keyword-only bbox/obtime) matches its monkeypatch in Task 4. `_download_asset_with_retry` signature matches its monkeypatch in Task 5. `tile_sources_by_year` keyed `pdsid → url` consistently in index (Task 4) and download (Task 5). Manifest constructors mirror the verified `lantmateriet` field set.

**Known runtime-confirm points (flagged, not blockers):**
- ODE longitude convention (0–360 vs −180–180) — confirm during Task 6 live run; `crs.normalize_bbox_to_ode` owns the rule if a shift is needed.
- `LayerManifest.role` allowed values — Task 5 Step 5 verifies `"rgb"` is accepted, else switch to the permitted value.
- `cli.py` provider-name help list — Task 7 handles if present.
