# Finland NLS Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a self-contained Maanmittauslaitos (NLS Finland) orthophoto provider that downloads year-aware GeoTIFFs via WCS for AOIs ≤ 2 km × 2 km in EPSG:3067, writing manifests compatible with the existing render and validate stages.

**Architecture:** New `src/satmap_dataset/providers/nls/` sub-package with a single-shot WCS GetCoverage per (AOI, year). Year discovery via `WCS DescribeCoverage`. Minimal additions to `models.py`/`config.py` to carry `provider` and `provider_options` fields without disturbing the existing Polish-Geoportal flow (which keeps `provider="geoportal"` by default). Three new Typer subcommands wire JSON configs to the new provider.

**Tech Stack:** Python 3.10+, `httpx` async client, `pydantic` v2 models, `typer` CLI, `pytest` with monkeypatch + `tmp_path`. No new third-party dependencies.

**Spec:** `docs/superpowers/specs/2026-05-09-finland-nls-provider-design.md`

---

## File Structure

**New files:**
- `src/satmap_dataset/providers/__init__.py` — empty marker, lets sub-package imports work
- `src/satmap_dataset/providers/nls/__init__.py` — re-exports `NlsProvider`
- `src/satmap_dataset/providers/nls/auth.py` — API-key resolution + Basic Auth header
- `src/satmap_dataset/providers/nls/wcs.py` — DescribeCoverage parser + GetCoverage URL builder
- `src/satmap_dataset/providers/nls/provider.py` — `NlsProvider.index()` and `download()` glue
- `tests/fixtures/nls/describe_coverage_ortokuva_vari.xml` — fixture WCS DescribeCoverage payload
- `tests/test_nls_config.py` — bbox-cap and provider field validation
- `tests/test_nls_auth.py` — key resolution precedence + Basic Auth shape
- `tests/test_nls_wcs_urls.py` — URL builders match documented format byte-for-byte
- `tests/test_nls_describe_coverage.py` — XML parser extracts year list correctly
- `tests/test_nls_index.py` — `NlsProvider.index()` happy path + edge cases
- `tests/test_nls_download.py` — `NlsProvider.download()` with mocked httpx
- `tests/test_cli_nls_commands.py` — Typer commands route through the provider

**Modified files:**
- `src/satmap_dataset/models.py` — add `provider` Literal and `provider_metadata` to `IndexManifest` and `DatasetManifest`; add `"wcs"` to `DatasetManifest.mode`
- `src/satmap_dataset/config.py` — add `provider` (default `"geoportal"`) and `provider_options` (default `{}`) to `IndexConfig` and `DownloadConfig`
- `src/satmap_dataset/cli.py` — add three Typer commands: `nls-index-json`, `nls-download-json`, `nls-run-json`

---

## Task 1: Extend models with provider field

**Files:**
- Modify: `src/satmap_dataset/models.py`
- Test: `tests/test_models_schema.py` (extend existing)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_models_schema.py`:

```python
from satmap_dataset.models import DatasetManifest, IndexManifest


def test_index_manifest_default_provider_is_geoportal():
    m = IndexManifest(
        year_start=2020,
        year_end=2020,
        bbox="0,0,1,1",
        srs="EPSG:2180",
        years_requested=[2020],
        year_statuses=[],
        years_available_wfs=[],
        years_included=[],
        passed=True,
    )
    assert m.provider == "geoportal"
    assert m.provider_metadata == {}


def test_dataset_manifest_accepts_nls_provider_and_wcs_mode():
    m = DatasetManifest(provider="nls", mode="wcs")
    assert m.provider == "nls"
    assert m.mode == "wcs"


def test_dataset_manifest_rejects_unknown_provider():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        DatasetManifest(provider="unknown")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models_schema.py -v -k "provider or wcs"`
Expected: FAIL — `IndexManifest` has no attribute `provider`.

- [ ] **Step 3: Implement minimal change in `models.py`**

In `IndexManifest` class, add after `kind`:

```python
provider: Literal["geoportal", "nls"] = "geoportal"
provider_metadata: dict[str, Any] = Field(default_factory=dict)
```

In `DatasetManifest` class, add after `kind`:

```python
provider: Literal["geoportal", "nls"] = "geoportal"
provider_metadata: dict[str, Any] = Field(default_factory=dict)
```

Change the `mode` Literal in `DatasetManifest`:

```python
mode: Literal["wms_tiled", "wfs_render", "hybrid", "wcs"] = "hybrid"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_models_schema.py -v -k "provider or wcs"`
Expected: PASS, 3 tests.

- [ ] **Step 5: Run full test suite to confirm no regression**

Run: `pytest -q`
Expected: All tests pass (the existing Polish flow uses defaults so adds match).

- [ ] **Step 6: Commit**

```bash
git add src/satmap_dataset/models.py tests/test_models_schema.py
git commit -m "feat(models): add provider field to manifests for multi-provider support"
```

---

## Task 2: Extend configs with provider + provider_options

**Files:**
- Modify: `src/satmap_dataset/config.py`
- Test: `tests/test_nls_config.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_nls_config.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest
from pydantic import ValidationError

from satmap_dataset.config import DownloadConfig, IndexConfig


def test_index_config_default_provider_geoportal():
    cfg = IndexConfig(year_start=2020, year_end=2020, bbox="0,0,1,1")
    assert cfg.provider == "geoportal"
    assert cfg.provider_options == {}


def test_index_config_accepts_nls_provider_with_options():
    cfg = IndexConfig(
        year_start=2020,
        year_end=2020,
        bbox="0,0,2000,2000",
        srs="EPSG:3067",
        provider="nls",
        provider_options={"api_key": "abc"},
    )
    assert cfg.provider == "nls"
    assert cfg.provider_options == {"api_key": "abc"}


def test_index_config_rejects_unknown_provider():
    with pytest.raises(ValidationError):
        IndexConfig(year_start=2020, year_end=2020, bbox="0,0,1,1", provider="boom")


def test_download_config_carries_provider_options():
    cfg = DownloadConfig(provider="nls", provider_options={"api_key": "abc"}, bbox="0,0,1,1")
    assert cfg.provider == "nls"
    assert cfg.provider_options["api_key"] == "abc"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_nls_config.py -v`
Expected: FAIL — `provider` field unknown on `IndexConfig`.

- [ ] **Step 3: Implement in `config.py`**

In `IndexConfig` class, after `output_json` and `year_availability_output_json`, add:

```python
provider: Literal["geoportal", "nls"] = "geoportal"
provider_options: dict[str, Any] = Field(default_factory=dict)
```

Add to imports at top of file:

```python
from typing import Any, Literal
```

Apply the same two fields to `DownloadConfig`. Leave `RunConfig` alone for now — the new NLS Typer commands take their own JSON shape.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_nls_config.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 5: Verify no regression**

Run: `pytest -q`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/satmap_dataset/config.py tests/test_nls_config.py
git commit -m "feat(config): add provider and provider_options to IndexConfig/DownloadConfig"
```

---

## Task 3: NLS API-key auth resolution

**Files:**
- Create: `src/satmap_dataset/providers/__init__.py`
- Create: `src/satmap_dataset/providers/nls/__init__.py`
- Create: `src/satmap_dataset/providers/nls/auth.py`
- Test: `tests/test_nls_auth.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_nls_auth.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from satmap_dataset.providers.nls.auth import basic_auth_header, resolve_api_key


def test_resolve_api_key_prefers_provider_options(monkeypatch, tmp_path):
    monkeypatch.setenv("SATMAP_NLS_API_KEY", "from-env")
    secret = tmp_path / ".secret"
    secret.write_text("from-secret\n", encoding="utf-8")
    key = resolve_api_key({"api_key": "from-options"}, secret_path=secret)
    assert key == "from-options"


def test_resolve_api_key_falls_back_to_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SATMAP_NLS_API_KEY", "from-env")
    secret = tmp_path / ".secret"
    secret.write_text("from-secret\n", encoding="utf-8")
    key = resolve_api_key({}, secret_path=secret)
    assert key == "from-env"


def test_resolve_api_key_falls_back_to_secret_file(monkeypatch, tmp_path):
    monkeypatch.delenv("SATMAP_NLS_API_KEY", raising=False)
    secret = tmp_path / ".secret"
    secret.write_text("from-secret\n", encoding="utf-8")
    key = resolve_api_key({}, secret_path=secret)
    assert key == "from-secret"


def test_resolve_api_key_missing_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("SATMAP_NLS_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="No NLS API key"):
        resolve_api_key({}, secret_path=tmp_path / "nope")


def test_basic_auth_header_uses_api_key_username():
    header = basic_auth_header("MYKEY")
    # base64("api-key:MYKEY") == "YXBpLWtleTpNWUtFWQ=="
    assert header == "Basic YXBpLWtleTpNWUtFWQ=="
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_nls_auth.py -v`
Expected: FAIL — module `satmap_dataset.providers.nls.auth` not found.

- [ ] **Step 3: Create package markers and the auth module**

Create `src/satmap_dataset/providers/__init__.py`:

```python
```

(empty file)

Create `src/satmap_dataset/providers/nls/__init__.py`:

```python
from __future__ import annotations

from satmap_dataset.providers.nls.provider import NlsProvider

__all__ = ["NlsProvider"]
```

Create `src/satmap_dataset/providers/nls/auth.py`:

```python
from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any


ENV_VAR = "SATMAP_NLS_API_KEY"


def resolve_api_key(
    provider_options: dict[str, Any],
    *,
    secret_path: Path | None = None,
) -> str:
    """Resolve the NLS open-data API key.

    Order: provider_options["api_key"] -> env var SATMAP_NLS_API_KEY ->
    single-line .secret file at secret_path. Raises RuntimeError if none set.
    """
    candidate = provider_options.get("api_key")
    if candidate:
        return str(candidate).strip()
    env_value = os.environ.get(ENV_VAR)
    if env_value:
        return env_value.strip()
    if secret_path is not None and secret_path.is_file():
        text = secret_path.read_text(encoding="utf-8").strip()
        if text:
            return text
    raise RuntimeError(
        "No NLS API key found. Set provider_options['api_key'], "
        f"env var {ENV_VAR}, or place a single-line .secret file at the project root."
    )


def basic_auth_header(api_key: str) -> str:
    """Return the HTTP Basic Auth header value used by NLS endpoints.

    Per NLS docs, username is the literal string 'api-key' and password is
    the API key itself.
    """
    token = base64.b64encode(f"api-key:{api_key}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"
```

Note: `__init__.py` imports `NlsProvider` which doesn't exist yet. To avoid breaking the test that only imports `auth`, leave the `__init__.py` import deferred. Replace the `__init__.py` contents above with this minimal version for now:

```python
from __future__ import annotations

__all__ = ["NlsProvider"]


def __getattr__(name: str):
    if name == "NlsProvider":
        from satmap_dataset.providers.nls.provider import NlsProvider as _NlsProvider

        return _NlsProvider
    raise AttributeError(name)
```

This is lazy import — `auth.py` is importable on its own.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_nls_auth.py -v`
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/providers/ tests/test_nls_auth.py
git commit -m "feat(nls): API key resolution and Basic Auth header"
```

---

## Task 4: WCS GetCoverage URL builder

**Files:**
- Create: `src/satmap_dataset/providers/nls/wcs.py`
- Test: `tests/test_nls_wcs_urls.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_nls_wcs_urls.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import parse_qsl, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.providers.nls.wcs import (
    build_describe_coverage_url,
    build_get_coverage_url,
)


WCS_BASE = "https://avoin-karttakuva.maanmittauslaitos.fi/ortokuvat-ja-korkeusmallit/wcs/v2"


def _qs(url: str) -> dict[str, str]:
    return dict(parse_qsl(urlparse(url).query, keep_blank_values=True))


def test_describe_coverage_url_shape():
    url = build_describe_coverage_url(WCS_BASE, coverage_id="ortokuva_vari")
    assert urlparse(url).path.endswith("/wcs/v2")
    qs = _qs(url)
    assert qs["service"] == "WCS"
    assert qs["version"] == "2.0.1"
    assert qs["request"] == "DescribeCoverage"
    assert qs["coverageID"] == "ortokuva_vari"


def test_get_coverage_url_includes_subsets_and_geotiff_options():
    url = build_get_coverage_url(
        WCS_BASE,
        coverage_id="ortokuva_vari",
        bbox=(393450, 7495450, 393650, 7495650),
        year=2010,
    )
    qs = _qs(url)
    assert qs["service"] == "WCS"
    assert qs["version"] == "2.0.1"
    assert qs["request"] == "GetCoverage"
    assert qs["CoverageID"] == "ortokuva_vari"
    assert qs["format"] == "image/tiff"
    assert qs["geotiff:compression"] == "LZW"
    assert qs["geotiff:tiling"] == "true"
    assert qs["geotiff:tilewidth"] == "256"
    assert qs["geotiff:tileheight"] == "256"
    assert "EPSG/0/3067" in qs["SubsettingCRS"]
    assert "EPSG/0/3067" in qs["OutputCRS"]
    # urllib parses repeated keys via parse_qsl — re-extract
    pairs = parse_qsl(urlparse(url).query, keep_blank_values=True)
    subsets = [v for k, v in pairs if k == "SUBSET"]
    assert any(s.startswith("E(393450") for s in subsets)
    assert any(s.startswith("N(7495450") for s in subsets)
    assert any('time("2010-12-31' in s for s in subsets)


def test_get_coverage_url_uses_provided_base():
    url = build_get_coverage_url(
        "https://example.test/wcs/v2",
        coverage_id="ortokuva_vari",
        bbox=(0, 0, 1000, 1000),
        year=2020,
    )
    assert url.startswith("https://example.test/wcs/v2?")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_nls_wcs_urls.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `wcs.py`**

Create `src/satmap_dataset/providers/nls/wcs.py`:

```python
from __future__ import annotations

from typing import Iterable
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
    # `urlencode` with `doseq=True` does not preserve repeated SUBSET keys cleanly
    # for our needs — build the list manually.
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
    """Format coords without scientific notation, trimming pointless zeros."""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_nls_wcs_urls.py -v`
Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/providers/nls/wcs.py tests/test_nls_wcs_urls.py
git commit -m "feat(nls): WCS DescribeCoverage and GetCoverage URL builders"
```

---

## Task 5: WCS DescribeCoverage XML parser

**Files:**
- Modify: `src/satmap_dataset/providers/nls/wcs.py`
- Create: `tests/fixtures/nls/describe_coverage_ortokuva_vari.xml`
- Test: `tests/test_nls_describe_coverage.py`

- [ ] **Step 1: Create the fixture XML**

Create `tests/fixtures/nls/describe_coverage_ortokuva_vari.xml`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<wcs:CoverageDescriptions
    xmlns:wcs="http://www.opengis.net/wcs/2.0"
    xmlns:gml="http://www.opengis.net/gml/3.2"
    xmlns:gmlrgrid="http://www.opengis.net/gml/3.3/rgrid">
  <wcs:CoverageDescription gml:id="ortokuva_vari">
    <gml:boundedBy>
      <gml:EnvelopeWithTimePeriod srsName="http://www.opengis.net/def/crs/EPSG/0/3067" axisLabels="E N time" uomLabels="m m d">
        <gml:lowerCorner>50000 6600000 2008-01-01T00:00:00.000Z</gml:lowerCorner>
        <gml:upperCorner>770000 7780000 2024-12-31T00:00:00.000Z</gml:upperCorner>
        <gml:beginPosition>2008-01-01T00:00:00.000Z</gml:beginPosition>
        <gml:endPosition>2024-12-31T00:00:00.000Z</gml:endPosition>
      </gml:EnvelopeWithTimePeriod>
    </gml:boundedBy>
    <wcs:CoverageId>ortokuva_vari</wcs:CoverageId>
    <gml:domainSet>
      <gmlrgrid:ReferenceableGridByVectors gml:id="grid_ortokuva_vari" dimension="3">
        <gmlrgrid:generalGridAxis>
          <gmlrgrid:GeneralGridAxis>
            <gmlrgrid:coefficients>"2008-12-31T00:00:00.000Z" "2010-12-31T00:00:00.000Z" "2012-12-31T00:00:00.000Z" "2014-12-31T00:00:00.000Z" "2016-12-31T00:00:00.000Z" "2018-12-31T00:00:00.000Z" "2020-12-31T00:00:00.000Z" "2022-12-31T00:00:00.000Z" "2024-12-31T00:00:00.000Z"</gmlrgrid:coefficients>
          </gmlrgrid:GeneralGridAxis>
        </gmlrgrid:generalGridAxis>
      </gmlrgrid:ReferenceableGridByVectors>
    </gml:domainSet>
  </wcs:CoverageDescription>
</wcs:CoverageDescriptions>
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_nls_describe_coverage.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.providers.nls.wcs import parse_describe_coverage_years


def test_parse_describe_coverage_extracts_unique_sorted_years():
    fixture = (
        Path(__file__).parent / "fixtures" / "nls" / "describe_coverage_ortokuva_vari.xml"
    )
    xml_bytes = fixture.read_bytes()
    years = parse_describe_coverage_years(xml_bytes)
    assert years == [2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]


def test_parse_describe_coverage_handles_no_time_axis():
    xml_bytes = b'<?xml version="1.0"?><wcs:CoverageDescriptions xmlns:wcs="http://www.opengis.net/wcs/2.0"/>'
    years = parse_describe_coverage_years(xml_bytes)
    assert years == []
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_nls_describe_coverage.py -v`
Expected: FAIL — `parse_describe_coverage_years` not defined.

- [ ] **Step 4: Implement the parser**

Append to `src/satmap_dataset/providers/nls/wcs.py`:

```python
import re
import xml.etree.ElementTree as ET


_TIME_PATTERN = re.compile(r'"(\d{4})-\d{2}-\d{2}T')


def parse_describe_coverage_years(xml_bytes: bytes) -> list[int]:
    """Extract the set of years that the WCS coverage's time axis advertises.

    Looks at any `coefficients` element holding ISO-like timestamps in quoted
    form ("YYYY-MM-DDT..."). The endpoint emits these inside the temporal
    `GeneralGridAxis`; we scan all matching elements to be tolerant of schema
    drift.
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return []
    years: set[int] = set()
    for element in root.iter():
        local = element.tag.split("}", 1)[-1]
        if local != "coefficients":
            continue
        text = element.text or ""
        for match in _TIME_PATTERN.finditer(text):
            years.add(int(match.group(1)))
    return sorted(years)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_nls_describe_coverage.py -v`
Expected: PASS, 2 tests.

- [ ] **Step 6: Commit**

```bash
git add src/satmap_dataset/providers/nls/wcs.py tests/fixtures/nls/ tests/test_nls_describe_coverage.py
git commit -m "feat(nls): parse available years from WCS DescribeCoverage XML"
```

---

## Task 6: NlsProvider.index() — happy path

**Files:**
- Create: `src/satmap_dataset/providers/nls/provider.py`
- Test: `tests/test_nls_index.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_nls_index.py`:

```python
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from satmap_dataset.config import IndexConfig
from satmap_dataset.providers.nls import provider as nls_provider
from satmap_dataset.providers.nls.provider import NlsProvider


def _fixture_xml() -> bytes:
    return (
        Path(__file__).parent / "fixtures" / "nls" / "describe_coverage_ortokuva_vari.xml"
    ).read_bytes()


def _config(tmp_path: Path, **overrides) -> IndexConfig:
    base = dict(
        year_start=2018,
        year_end=2022,
        bbox="385000,6675000,387000,6677000",
        srs="EPSG:3067",
        provider="nls",
        provider_options={"api_key": "test-key"},
        output_json=tmp_path / "index_manifest.json",
        year_availability_output_json=tmp_path / "year_availability_report.json",
    )
    base.update(overrides)
    return IndexConfig(**base)


def test_index_writes_manifest_with_one_url_per_year(monkeypatch, tmp_path):
    monkeypatch.setattr(nls_provider, "_fetch_describe_coverage_xml", lambda **kw: _fixture_xml())
    cfg = _config(tmp_path)
    exit_code, manifest_path = NlsProvider().index(cfg)
    assert exit_code == 0
    assert manifest_path == cfg.output_json
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["provider"] == "nls"
    # available years from fixture intersected with [2018..2022]: 2018, 2020, 2022
    assert data["years_included"] == [2018, 2020, 2022]
    for year in data["years_included"]:
        sources = data["tile_sources_by_year"][str(year)]
        assert list(sources.keys()) == [f"nls_{year}"]
        url = sources[f"nls_{year}"]
        assert "request=GetCoverage" in url
        assert f'time(%22{year}-12-31' in url or f'time("{year}-12-31' in url


def test_index_rejects_bbox_larger_than_2km(tmp_path):
    cfg = _config(tmp_path, bbox="385000,6675000,388000,6678000")  # 3 km square
    exit_code, _ = NlsProvider().index(cfg)
    assert exit_code != 0
    text = (tmp_path / "index_manifest.json").read_text(encoding="utf-8")
    assert "exceeds NLS WCS cap" in text or "bbox" in text.lower()


def test_index_fails_when_no_years_in_range(monkeypatch, tmp_path):
    monkeypatch.setattr(nls_provider, "_fetch_describe_coverage_xml", lambda **kw: _fixture_xml())
    cfg = _config(tmp_path, year_start=2030, year_end=2031)
    exit_code, _ = NlsProvider().index(cfg)
    assert exit_code != 0


def test_index_uses_default_wcs_url_when_not_overridden(monkeypatch, tmp_path):
    seen = {}

    def fake_fetch(**kwargs):
        seen.update(kwargs)
        return _fixture_xml()

    monkeypatch.setattr(nls_provider, "_fetch_describe_coverage_xml", fake_fetch)
    NlsProvider().index(_config(tmp_path))
    assert seen["base_url"].endswith("/wcs/v2")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_nls_index.py -v`
Expected: FAIL — `NlsProvider` not defined.

- [ ] **Step 3: Implement `provider.py` (index half)**

Create `src/satmap_dataset/providers/nls/provider.py`:

```python
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import httpx

from satmap_dataset.config import DownloadConfig, IndexConfig
from satmap_dataset.geoportal.http import RetryPolicy
from satmap_dataset.models import IndexManifest, YearAvailabilityReport, YearStatus
from satmap_dataset.pipeline.validator import evaluate_year_policy
from satmap_dataset.providers.nls.auth import basic_auth_header, resolve_api_key
from satmap_dataset.providers.nls.wcs import (
    DEFAULT_COVERAGE_ID,
    DEFAULT_WCS_URL,
    build_describe_coverage_url,
    build_get_coverage_url,
    parse_describe_coverage_years,
)


logger = logging.getLogger("satmap_dataset.nls")

WCS_AOI_CAP_METERS = 2000.0


def _option(options: dict[str, Any], key: str, default: Any) -> Any:
    if key in options and options[key] not in (None, ""):
        return options[key]
    return default


def _parse_bbox(bbox: str) -> tuple[float, float, float, float]:
    parts = [float(p.strip()) for p in bbox.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must have format xmin,ymin,xmax,ymax")
    xmin, ymin, xmax, ymax = parts
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("bbox must satisfy xmin<xmax and ymin<ymax")
    return (xmin, ymin, xmax, ymax)


def _check_aoi_cap(bbox: tuple[float, float, float, float]) -> str | None:
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin
    if width > WCS_AOI_CAP_METERS or height > WCS_AOI_CAP_METERS:
        return (
            f"bbox {width:.0f}m x {height:.0f}m exceeds NLS WCS cap of "
            f"{WCS_AOI_CAP_METERS:.0f}m on either side"
        )
    return None


def _fetch_describe_coverage_xml(
    *,
    base_url: str,
    coverage_id: str,
    api_key: str,
    timeout: float = 60.0,
) -> bytes:
    """Synchronous DescribeCoverage fetch — kept simple; pulled per index run."""
    url = build_describe_coverage_url(base_url, coverage_id=coverage_id)
    headers = {"Authorization": basic_auth_header(api_key), "User-Agent": "satmap_dataset/0.1"}
    with httpx.Client(timeout=timeout, headers=headers) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.content


def _write_failed_manifest(config: IndexConfig, error: str) -> None:
    manifest = IndexManifest(
        provider="nls",
        year_start=config.year_start,
        year_end=config.year_end,
        bbox=config.bbox,
        srs=config.srs,
        strict_years=config.strict_years,
        min_years=config.min_years,
        years_requested=config.requested_years,
        year_statuses=[],
        years_available_wfs=[],
        years_included=[],
        years_excluded_with_reason={year: error for year in config.requested_years},
        passed=False,
        errors=[error],
        run_parameters=config.model_dump(mode="json"),
    )
    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    config.output_json.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


class NlsProvider:
    name = "nls"
    default_target_srs = "EPSG:3067"

    def index(self, config: IndexConfig) -> tuple[int, Path]:
        bbox = _parse_bbox(config.bbox)
        cap_error = _check_aoi_cap(bbox)
        if cap_error is not None:
            logger.error("NLS index: %s", cap_error)
            _write_failed_manifest(config, cap_error)
            return 2, config.output_json

        options = dict(config.provider_options)
        base_url = str(_option(options, "wcs_url", DEFAULT_WCS_URL))
        coverage_id = str(_option(options, "coverage_id", DEFAULT_COVERAGE_ID))
        api_key = resolve_api_key(options, secret_path=Path(".secret"))

        try:
            xml_bytes = _fetch_describe_coverage_xml(
                base_url=base_url,
                coverage_id=coverage_id,
                api_key=api_key,
            )
        except httpx.HTTPError as exc:
            error = f"DescribeCoverage failed: {exc}"
            _write_failed_manifest(config, error)
            return 1, config.output_json

        available_years = parse_describe_coverage_years(xml_bytes)
        requested_years = config.requested_years
        years_included = [y for y in requested_years if y in set(available_years)]
        years_excluded = {
            y: "year_not_in_wcs_describe_coverage"
            for y in requested_years
            if y not in set(available_years)
        }
        year_statuses = [
            YearStatus(
                year=y,
                typename_exists=(y in set(available_years)),
                feature_count=1 if y in set(available_years) else 0,
                status="has_features" if y in set(available_years) else "no_typename",
                reason=None if y in set(available_years) else "year_not_in_wcs_describe_coverage",
            )
            for y in requested_years
        ]

        tile_sources_by_year: dict[int, dict[str, str]] = {}
        tile_bboxes_by_year: dict[int, dict[str, list[float]]] = {}
        for year in years_included:
            url = build_get_coverage_url(
                base_url,
                coverage_id=coverage_id,
                bbox=bbox,
                year=year,
            )
            tile_sources_by_year[year] = {f"nls_{year}": url}
            tile_bboxes_by_year[year] = {f"nls_{year}": list(bbox)}

        policy = evaluate_year_policy(
            requested_years=requested_years,
            available_years=years_included,
            strict_years=config.strict_years,
            min_years=config.min_years,
        )
        errors = list(policy.errors)
        warnings = list(policy.warnings)
        if not years_included:
            errors.append("WCS DescribeCoverage returned no years intersecting the requested range.")

        provider_metadata: dict[str, Any] = {
            "wcs_url": base_url,
            "coverage_id": coverage_id,
            "available_years_in_coverage": available_years,
            "native_srs": "EPSG:3067",
            "aoi_cap_meters": WCS_AOI_CAP_METERS,
        }

        manifest = IndexManifest(
            provider="nls",
            year_start=config.year_start,
            year_end=config.year_end,
            bbox=config.bbox,
            srs=config.srs,
            strict_years=config.strict_years,
            min_years=config.min_years,
            years_requested=requested_years,
            year_statuses=year_statuses,
            years_available_wfs=available_years,
            years_included=years_included,
            years_excluded_with_reason=years_excluded,
            common_tile_ids=[f"nls_{y}" for y in years_included],
            tile_sources_by_year=tile_sources_by_year,
            tile_bboxes_by_year=tile_bboxes_by_year,
            passed=policy.passed and bool(years_included),
            errors=errors,
            warnings=warnings,
            run_parameters=config.model_dump(mode="json"),
            provider_metadata=provider_metadata,
        )

        availability = YearAvailabilityReport(
            year_start=config.year_start,
            year_end=config.year_end,
            bbox=config.bbox,
            srs=config.srs,
            years_requested=requested_years,
            year_statuses=year_statuses,
            years_available_wfs=available_years,
            years_included=years_included,
            years_excluded_with_reason=years_excluded,
            strict_years=config.strict_years,
            min_years=config.min_years,
            passed=manifest.passed,
            errors=errors,
            warnings=warnings,
            run_parameters=config.model_dump(mode="json"),
        )

        config.output_json.parent.mkdir(parents=True, exist_ok=True)
        config.year_availability_output_json.parent.mkdir(parents=True, exist_ok=True)
        config.output_json.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        config.year_availability_output_json.write_text(
            availability.model_dump_json(indent=2), encoding="utf-8"
        )
        logger.info(
            "NLS index: years_included=%s available=%s passed=%s",
            len(years_included),
            len(available_years),
            manifest.passed,
        )
        return (0 if manifest.passed else 1), config.output_json

    def download(self, config: DownloadConfig) -> tuple[int, Path]:
        raise NotImplementedError("download() is implemented in Task 7")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_nls_index.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/providers/nls/provider.py tests/test_nls_index.py
git commit -m "feat(nls): NlsProvider.index() builds WCS-backed manifests"
```

---

## Task 7: NlsProvider.download()

**Files:**
- Modify: `src/satmap_dataset/providers/nls/provider.py`
- Test: `tests/test_nls_download.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_nls_download.py`:

```python
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import httpx
import pytest

from satmap_dataset.config import DownloadConfig
from satmap_dataset.models import IndexManifest, YearStatus
from satmap_dataset.providers.nls.provider import NlsProvider


def _write_index_manifest(tmp_path: Path, years: list[int]) -> Path:
    manifest = IndexManifest(
        provider="nls",
        year_start=min(years),
        year_end=max(years),
        bbox="385000,6675000,387000,6677000",
        srs="EPSG:3067",
        years_requested=years,
        year_statuses=[
            YearStatus(year=y, typename_exists=True, feature_count=1, status="has_features")
            for y in years
        ],
        years_available_wfs=years,
        years_included=years,
        common_tile_ids=[f"nls_{y}" for y in years],
        tile_sources_by_year={y: {f"nls_{y}": f"https://example.test/wcs?year={y}"} for y in years},
        tile_bboxes_by_year={y: {f"nls_{y}": [0, 0, 1, 1]} for y in years},
        passed=True,
    )
    path = tmp_path / "index_manifest.json"
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return path


def test_download_writes_one_geotiff_per_year(monkeypatch, tmp_path):
    index_path = _write_index_manifest(tmp_path, [2018, 2020])
    cfg = DownloadConfig(
        index_manifest=index_path,
        download_root=tmp_path / "downloads",
        provider="nls",
        provider_options={"api_key": "test-key"},
        bbox="385000,6675000,387000,6677000",
        srs="EPSG:3067",
        output_json=tmp_path / "dataset_manifest_download.json",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"FAKE_TIFF_BYTES")

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "satmap_dataset.providers.nls.provider._make_async_client",
        lambda **kw: httpx.AsyncClient(transport=transport, **kw),
    )

    exit_code, manifest_path = NlsProvider().download(cfg)
    assert exit_code == 0
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["provider"] == "nls"
    assert data["mode"] == "wcs"
    assert sorted(data["years_included"]) == [2018, 2020]
    for year in [2018, 2020]:
        out = cfg.download_root / str(year) / f"nls_{year}.tif"
        assert out.is_file()
        assert out.read_bytes() == b"FAKE_TIFF_BYTES"


def test_download_marks_failed_on_http_error(monkeypatch, tmp_path):
    index_path = _write_index_manifest(tmp_path, [2018])
    cfg = DownloadConfig(
        index_manifest=index_path,
        download_root=tmp_path / "downloads",
        provider="nls",
        provider_options={"api_key": "test-key"},
        bbox="385000,6675000,387000,6677000",
        srs="EPSG:3067",
        retries=0,
        output_json=tmp_path / "dataset_manifest_download.json",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, content=b"unauthorized")

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "satmap_dataset.providers.nls.provider._make_async_client",
        lambda **kw: httpx.AsyncClient(transport=transport, **kw),
    )

    exit_code, manifest_path = NlsProvider().download(cfg)
    assert exit_code != 0
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["passed"] is False
    assert data["years_included"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_nls_download.py -v`
Expected: FAIL — `NotImplementedError` from the placeholder.

- [ ] **Step 3: Implement download() in `provider.py`**

Replace the `download()` method body in `provider.py` with:

```python
    def download(self, config: DownloadConfig) -> tuple[int, Path]:
        return asyncio.run(self._download_async(config))

    async def _download_async(self, config: DownloadConfig) -> tuple[int, Path]:
        from satmap_dataset.models import DatasetManifest

        index_manifest = IndexManifest.model_validate_json(
            config.index_manifest.read_text(encoding="utf-8")
        )
        options = dict(config.provider_options)
        api_key = resolve_api_key(options, secret_path=Path(".secret"))
        headers = {
            "Authorization": basic_auth_header(api_key),
            "User-Agent": "satmap_dataset/0.1",
        }

        timeout = httpx.Timeout(timeout=config.timeout, connect=min(config.timeout, 20.0))
        limits = httpx.Limits(
            max_connections=max(1, config.concurrency),
            max_keepalive_connections=max(1, config.concurrency),
        )

        assets: list[str] = []
        failed: list[str] = []
        years_source_map: dict[int, str] = {}
        years_included_effective: list[int] = []

        async with _make_async_client(
            timeout=timeout, limits=limits, headers=headers, follow_redirects=True
        ) as client:
            for year in index_manifest.years_included:
                sources = index_manifest.tile_sources_by_year.get(year, {})
                if not sources:
                    failed.append(f"year_{year}_no_source")
                    continue
                tile_id, url = next(iter(sources.items()))
                output_path = config.download_root / str(year) / f"{tile_id}.tif"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                ok = output_path.exists() and output_path.stat().st_size > 0 and not config.overwrite
                if not ok:
                    ok = await _download_one(client, url, output_path, retries=config.retries)
                if ok:
                    assets.append(str(output_path))
                    years_source_map[year] = "wcs"
                    years_included_effective.append(year)
                else:
                    failed.append(url)

        manifest = DatasetManifest(
            provider="nls",
            stage="download",
            mode="wcs",
            years_requested=index_manifest.years_requested,
            years_available_wfs=index_manifest.years_available_wfs,
            years_included=sorted(years_included_effective),
            years_excluded_with_reason=index_manifest.years_excluded_with_reason,
            common_tile_ids=index_manifest.common_tile_ids,
            tile_sources_by_year=index_manifest.tile_sources_by_year,
            tile_bboxes_by_year=index_manifest.tile_bboxes_by_year,
            assets=sorted(set(assets)),
            source_manifest=str(config.index_manifest),
            target_bbox=config.bbox,
            target_srs=config.srs,
            profile=config.profile,
            px_per_meter=config.px_per_meter,
            years_source_map=years_source_map,
            forced_wms_years=[],
            passed=bool(assets) and not failed,
            notes=f"provider=nls downloaded={len(assets)} failed={len(failed)}",
            run_parameters=config.model_dump(mode="json"),
            provider_metadata={"failed_urls": failed},
        )
        config.output_json.parent.mkdir(parents=True, exist_ok=True)
        config.output_json.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return (0 if manifest.passed else 1), config.output_json
```

Add at module level (above the class):

```python
import aiofiles


def _make_async_client(**kwargs: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(**kwargs)


async def _download_one(
    client: httpx.AsyncClient,
    url: str,
    output_path: Path,
    *,
    retries: int,
) -> bool:
    attempts = max(1, retries + 1)
    for attempt in range(1, attempts + 1):
        try:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                async with aiofiles.open(output_path, "wb") as fp:
                    async for chunk in response.aiter_bytes():
                        await fp.write(chunk)
            return True
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (400, 401, 403, 404):
                logger.error("NLS download terminal status=%s url=%s", exc.response.status_code, url)
                return False
            if attempt >= attempts:
                return False
            await asyncio.sleep(0.5 * attempt)
        except httpx.HTTPError as exc:
            if attempt >= attempts:
                logger.error("NLS download exhausted retries url=%s err=%s", url, exc)
                return False
            await asyncio.sleep(0.5 * attempt)
    return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_nls_download.py -v`
Expected: PASS, 2 tests.

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/providers/nls/provider.py tests/test_nls_download.py
git commit -m "feat(nls): NlsProvider.download() streams WCS GeoTIFFs"
```

---

## Task 8: CLI commands `nls-index-json`, `nls-download-json`, `nls-run-json`

**Files:**
- Modify: `src/satmap_dataset/cli.py`
- Test: `tests/test_cli_nls_commands.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli_nls_commands.py`:

```python
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from typer.testing import CliRunner

from satmap_dataset.cli import app
from satmap_dataset.providers.nls import provider as nls_provider

runner = CliRunner()


def _fixture_xml() -> bytes:
    return (
        Path(__file__).parent / "fixtures" / "nls" / "describe_coverage_ortokuva_vari.xml"
    ).read_bytes()


def test_nls_index_json_invokes_provider(monkeypatch, tmp_path):
    monkeypatch.setattr(nls_provider, "_fetch_describe_coverage_xml", lambda **kw: _fixture_xml())
    cfg = {
        "year_start": 2018,
        "year_end": 2022,
        "bbox": "385000,6675000,387000,6677000",
        "srs": "EPSG:3067",
        "provider": "nls",
        "provider_options": {"api_key": "test-key"},
        "output_json": str(tmp_path / "index_manifest.json"),
        "year_availability_output_json": str(tmp_path / "year_availability_report.json"),
    }
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")
    result = runner.invoke(app, ["nls-index-json", str(config_path)])
    assert result.exit_code == 0, result.output
    data = json.loads((tmp_path / "index_manifest.json").read_text(encoding="utf-8"))
    assert data["provider"] == "nls"
    assert data["years_included"]


def test_nls_index_json_validation_error_exits_2(tmp_path):
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps({"year_start": 2030, "year_end": 2020, "bbox": "0,0,1,1"}))
    result = runner.invoke(app, ["nls-index-json", str(config_path)])
    assert result.exit_code == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_nls_commands.py -v`
Expected: FAIL — `nls-index-json` command does not exist.

- [ ] **Step 3: Add CLI commands in `cli.py`**

Append at the bottom of `src/satmap_dataset/cli.py`, just before any final wiring (search for `if __name__ == "__main__":` or the end of file; place before that):

```python
@app.command("nls-index-json")
def nls_index_json(config_json: Path = typer.Argument(..., exists=True)) -> None:
    from satmap_dataset.providers.nls import NlsProvider

    payload = json.loads(config_json.read_text(encoding="utf-8"))
    try:
        cfg = IndexConfig(**payload)
    except ValidationError as error:
        _print_validation_error(error)
        raise typer.Exit(code=2) from error
    exit_code, artifact = NlsProvider().index(cfg)
    _finish(exit_code, artifact)


@app.command("nls-download-json")
def nls_download_json(config_json: Path = typer.Argument(..., exists=True)) -> None:
    from satmap_dataset.providers.nls import NlsProvider

    payload = json.loads(config_json.read_text(encoding="utf-8"))
    try:
        cfg = DownloadConfig(**payload)
    except ValidationError as error:
        _print_validation_error(error)
        raise typer.Exit(code=2) from error
    exit_code, artifact = NlsProvider().download(cfg)
    _finish(exit_code, artifact)


@app.command("nls-run-json")
def nls_run_json(config_json: Path = typer.Argument(..., exists=True)) -> None:
    """Single-shot NLS index + download from one JSON config.

    The same JSON is used for both stages; index_manifest is taken from
    output_json after the index step.
    """
    from satmap_dataset.providers.nls import NlsProvider

    payload = json.loads(config_json.read_text(encoding="utf-8"))
    try:
        index_cfg = IndexConfig(**{k: v for k, v in payload.items() if k in IndexConfig.model_fields})
    except ValidationError as error:
        _print_validation_error(error)
        raise typer.Exit(code=2) from error
    provider = NlsProvider()
    exit_code, index_artifact = provider.index(index_cfg)
    if exit_code != 0:
        _finish(exit_code, index_artifact)
    download_payload = {k: v for k, v in payload.items() if k in DownloadConfig.model_fields}
    download_payload.setdefault("index_manifest", str(index_artifact))
    download_payload.setdefault("provider", "nls")
    download_payload.setdefault("provider_options", payload.get("provider_options", {}))
    download_payload.setdefault("bbox", payload.get("bbox"))
    download_payload.setdefault("srs", payload.get("srs", "EPSG:3067"))
    try:
        download_cfg = DownloadConfig(**download_payload)
    except ValidationError as error:
        _print_validation_error(error)
        raise typer.Exit(code=2) from error
    exit_code, artifact = provider.download(download_cfg)
    _finish(exit_code, artifact)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_nls_commands.py -v`
Expected: PASS, 2 tests.

- [ ] **Step 5: Run the full test suite**

Run: `pytest -q`
Expected: All tests pass — Polish flow unaffected because `provider` defaults to `"geoportal"` everywhere.

- [ ] **Step 6: Commit**

```bash
git add src/satmap_dataset/cli.py tests/test_cli_nls_commands.py
git commit -m "feat(cli): nls-index-json / nls-download-json / nls-run-json commands"
```

---

## Task 9: README + example config snippet

**Files:**
- Modify: `README.md`
- Create: `configs/run/base_nls.json`

- [ ] **Step 1: Add an example NLS base config**

Create `configs/run/base_nls.json`:

```json
{
  "year_start": 2018,
  "year_end": 2024,
  "bbox": "385000,6675000,387000,6677000",
  "srs": "EPSG:3067",
  "strict_years": false,
  "min_years": 1,
  "provider": "nls",
  "provider_options": {
    "wcs_url": "https://avoin-karttakuva.maanmittauslaitos.fi/ortokuvat-ja-korkeusmallit/wcs/v2",
    "coverage_id": "ortokuva_vari"
  },
  "output_json": "artifacts/index_manifest.json",
  "year_availability_output_json": "artifacts/year_availability_report.json",
  "download_root": "downloads_nls"
}
```

- [ ] **Step 2: Append a README section**

Add to `README.md` (immediately after the existing usage examples, or under a new "## Providers" heading if none exists):

```markdown
## Finland (Maanmittauslaitos / NLS) provider

The `nls` provider downloads year-aware orthophotos via Maanmittauslaitos's open
WCS endpoint. License: CC BY 4.0. Requires a free API key from
https://omatili.maanmittauslaitos.fi/ — paste it into a `.secret` file at the
repo root, set `SATMAP_NLS_API_KEY`, or pass it via `provider_options.api_key`.

Hard limit: AOIs must be ≤ 2000 m × 2000 m in EPSG:3067 (matches the project's
default `square_km=4.0`). Larger AOIs are rejected at config time.

```bash
# Index + download in one shot
python -m satmap_dataset.cli nls-run-json configs/run/base_nls.json

# Or run stages separately
python -m satmap_dataset.cli nls-index-json configs/run/base_nls.json
python -m satmap_dataset.cli nls-download-json configs/run/base_nls.json
```

The downloaded GeoTIFFs land under `<download_root>/<year>/nls_<year>.tif` and
are consumed by the existing `render` and `validate` stages unchanged.
```

- [ ] **Step 3: Commit**

```bash
git add README.md configs/run/base_nls.json
git commit -m "docs(nls): example config and README usage section"
```

---

## Task 10: Final verification

- [ ] **Step 1: Run the full test suite one last time**

Run: `pytest -q`
Expected: All tests pass with no regressions on the Polish flow.

- [ ] **Step 2: Smoke-test the CLI shape**

Run: `python -m satmap_dataset.cli --help | grep nls`
Expected: Three new commands listed: `nls-index-json`, `nls-download-json`, `nls-run-json`.

- [ ] **Step 3: Confirm no live network calls**

Run: `pytest -q tests/test_nls_*.py tests/test_cli_nls_commands.py`
Expected: All pass under offline conditions (CI-friendly).

- [ ] **Step 4: Commit any remaining cleanup if needed**

If lint/format hooks make changes:

```bash
git add -A && git commit -m "chore(nls): formatting cleanup after implementation"
```

---

## Implementation order summary

1. Task 1 — manifest provider field (foundation)
2. Task 2 — config provider + provider_options (foundation)
3. Task 3 — auth (independent)
4. Task 4 — WCS URL builders (depends on nothing else NLS-side)
5. Task 5 — DescribeCoverage parser (extends Task 4 file)
6. Task 6 — `index()` (uses Tasks 1, 2, 3, 4, 5)
7. Task 7 — `download()` (uses Tasks 1, 2, 3, 6 file)
8. Task 8 — CLI commands (uses Tasks 6, 7)
9. Task 9 — docs/config example (uses everything)
10. Task 10 — final verification

Each task ends with green tests and a commit, so the worktree is always shippable.
