# GSD in Index Manifests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record per-tile and per-year ground sample distance (GSD) from the WFS `piksel` attribute into the index manifest and year availability report.

**Architecture:** Extract `gsd` per tile in `wfs_client._extract_tile_acquisition_metadata`, carry it on `TileAcquisitionMetadata`, then aggregate a per-year `YearGsdSummary` (histogram + finest/coarsest) in `index_builder` and attach `gsd_by_year` to both `IndexManifest` and `YearAvailabilityReport`.

**Tech Stack:** Python 3.10+, Pydantic v2, pytest, `xml.etree.ElementTree`.

## Global Constraints

- All new model fields are optional with defaults (`Field(default_factory=...)` / `= None`) so existing checked-in manifest fixtures keep validating.
- GSD is in meters, parsed as `float`. Missing/blank `piksel` → `None`.
- Histogram keys are strings (JSON object keys); canonicalize via `_gsd_key`.
- Follow existing patterns: mirror `_parse_int_or_none` for float parsing; mirror `test_wfs_client_grid_filter.py` fixture style.

---

### Task 1: Per-tile GSD extraction in WFS client

**Files:**
- Modify: `src/satmap_dataset/models.py` (add `gsd` to `TileAcquisitionMetadata`, ~line 23-26)
- Modify: `src/satmap_dataset/geoportal/wfs_client.py` (add `_parse_float_or_none` near `_parse_int_or_none` line 231; extend `_extract_tile_acquisition_metadata` line 243; widen dict type hints at lines 263, 289)
- Test: `tests/test_wfs_gsd_extraction.py` (create)

**Interfaces:**
- Produces: `TileAcquisitionMetadata.gsd: float | None`; `wfs_client._parse_float_or_none(value: str | None) -> float | None`; `_extract_tile_acquisition_metadata` returns dict now including key `"gsd"`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_wfs_gsd_extraction.py`:

```python
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.geoportal import wfs_client


def _feature(piksel_xml: str) -> ET.Element:
    xml = f"""<wfs:member xmlns:gugik="http://www.gugik.gov.pl" xmlns:wfs="http://www.opengis.net/wfs/2.0">
      <gugik:SkorowidzOrtofomapy2024>
        <gugik:godlo>N-33-130-D-d-1-2</gugik:godlo>
        <gugik:akt_rok>2024</gugik:akt_rok>
        {piksel_xml}
        <gugik:url_do_pobrania>https://x/y_N-33-130-D-d-1-2.tif</gugik:url_do_pobrania>
      </gugik:SkorowidzOrtofomapy2024>
    </wfs:member>"""
    return ET.fromstring(xml)


def test_parse_float_or_none():
    assert wfs_client._parse_float_or_none("0.05") == 0.05
    assert wfs_client._parse_float_or_none(" 0.25 ") == 0.25
    assert wfs_client._parse_float_or_none("") is None
    assert wfs_client._parse_float_or_none(None) is None
    assert wfs_client._parse_float_or_none("abc") is None


def test_extract_metadata_includes_gsd():
    meta = wfs_client._extract_tile_acquisition_metadata(
        _feature("<gugik:piksel>0.05</gugik:piksel>"), 2024
    )
    assert meta["gsd"] == 0.05


def test_extract_metadata_missing_piksel():
    meta = wfs_client._extract_tile_acquisition_metadata(_feature(""), 2024)
    assert meta["gsd"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_wfs_gsd_extraction.py -v`
Expected: FAIL with `AttributeError: module ... has no attribute '_parse_float_or_none'`

- [ ] **Step 3: Implement**

In `src/satmap_dataset/models.py`, extend `TileAcquisitionMetadata`:

```python
class TileAcquisitionMetadata(BaseModel):
    acquisition_date: str | None = None
    publication_date: str | None = None
    acquisition_year: int | None = None
    gsd: float | None = None
```

In `src/satmap_dataset/geoportal/wfs_client.py`, add after `_parse_int_or_none` (line ~240):

```python
def _parse_float_or_none(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None
```

Extend `_extract_tile_acquisition_metadata` (and widen its return annotation to include `float`):

```python
def _extract_tile_acquisition_metadata(feature: ET.Element, year: int) -> dict[str, int | str | float | None]:
    acquisition_date = _find_timeinstant_value(feature, "akt_data")
    publication_date = _find_timeinstant_value(feature, "dt_pzgik")
    acquisition_year = _parse_int_or_none(_find_attr_value(feature, "akt_rok")) or year
    gsd = _parse_float_or_none(_find_attr_value(feature, "piksel"))
    return {
        "acquisition_date": acquisition_date,
        "publication_date": publication_date,
        "acquisition_year": acquisition_year,
        "gsd": gsd,
    }
```

Widen the two affected type hints so mypy/readers stay consistent:
- line ~263 in `get_year_tiles` signature: `dict[str, dict[str, int | str | float | None]]`
- line ~289: `tile_acquisition: dict[str, dict[str, int | str | float | None]] = {}`

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_wfs_gsd_extraction.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/models.py src/satmap_dataset/geoportal/wfs_client.py tests/test_wfs_gsd_extraction.py
git commit -m "feat(index): extract per-tile GSD from WFS piksel attribute"
```

---

### Task 2: Per-year GSD summary model + aggregation

**Files:**
- Modify: `src/satmap_dataset/models.py` (add `YearGsdSummary` after `TileAcquisitionMetadata`)
- Modify: `src/satmap_dataset/pipeline/index_builder.py` (add `_gsd_key` and `_summarize_gsd_by_year` helpers)
- Test: `tests/test_gsd_summary.py` (create)

**Interfaces:**
- Consumes: `TileAcquisitionMetadata.gsd` (Task 1).
- Produces: `models.YearGsdSummary(histogram: dict[str, int], finest: float | None, coarsest: float | None)`; `index_builder._summarize_gsd_by_year(tile_acquisition_by_year: dict[int, dict[str, TileAcquisitionMetadata | dict]]) -> dict[int, YearGsdSummary]`; `index_builder._gsd_key(value: float) -> str`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_gsd_summary.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.models import TileAcquisitionMetadata, YearGsdSummary
from satmap_dataset.pipeline import index_builder


def test_gsd_key_canonicalizes():
    assert index_builder._gsd_key(0.05) == "0.05"
    assert index_builder._gsd_key(0.050) == "0.05"
    assert index_builder._gsd_key(0.25) == "0.25"


def test_summarize_mixed_year():
    tiles = {
        2024: {
            "a": TileAcquisitionMetadata(gsd=0.05),
            "b": TileAcquisitionMetadata(gsd=0.05),
            "c": TileAcquisitionMetadata(gsd=0.25),
        }
    }
    summary = index_builder._summarize_gsd_by_year(tiles)
    assert summary[2024].histogram == {"0.05": 2, "0.25": 1}
    assert summary[2024].finest == 0.05
    assert summary[2024].coarsest == 0.25


def test_summarize_all_none():
    tiles = {2014: {"a": TileAcquisitionMetadata(gsd=None)}}
    summary = index_builder._summarize_gsd_by_year(tiles)
    assert summary[2014].histogram == {}
    assert summary[2014].finest is None
    assert summary[2014].coarsest is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gsd_summary.py -v`
Expected: FAIL with `ImportError: cannot import name 'YearGsdSummary'`

- [ ] **Step 3: Implement**

In `src/satmap_dataset/models.py`, add after `TileAcquisitionMetadata`:

```python
class YearGsdSummary(BaseModel):
    histogram: dict[str, int] = Field(default_factory=dict)
    finest: float | None = None
    coarsest: float | None = None
```

In `src/satmap_dataset/pipeline/index_builder.py`, add the import for the new model (alongside the existing `IndexManifest`/`YearAvailabilityReport` imports) and these helpers near the top-level helper functions:

```python
def _gsd_key(value: float) -> str:
    # Canonical string key: trim trailing zeros but keep at least one decimal.
    return f"{value:g}"


def _summarize_gsd_by_year(
    tile_acquisition_by_year: dict[int, dict[str, Any]],
) -> dict[int, YearGsdSummary]:
    summaries: dict[int, YearGsdSummary] = {}
    for year, tiles in tile_acquisition_by_year.items():
        histogram: dict[str, int] = {}
        known: list[float] = []
        for meta in tiles.values():
            gsd = meta.gsd if hasattr(meta, "gsd") else meta.get("gsd")
            if gsd is None:
                continue
            known.append(gsd)
            histogram[_gsd_key(gsd)] = histogram.get(_gsd_key(gsd), 0) + 1
        summaries[year] = YearGsdSummary(
            histogram=histogram,
            finest=min(known) if known else None,
            coarsest=max(known) if known else None,
        )
    return summaries
```

Ensure `Any` is imported (`from typing import Any`) and `YearGsdSummary` is added to the models import line.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_gsd_summary.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/models.py src/satmap_dataset/pipeline/index_builder.py tests/test_gsd_summary.py
git commit -m "feat(index): add YearGsdSummary model and per-year aggregation"
```

---

### Task 3: Wire `gsd_by_year` into both manifests

**Files:**
- Modify: `src/satmap_dataset/models.py` (add `gsd_by_year` to `IndexManifest` ~line 101 and `YearAvailabilityReport`)
- Modify: `src/satmap_dataset/pipeline/index_builder.py` (compute summary line ~164; pass to both constructors lines ~216 and ~240)
- Test: `tests/test_index_gsd_by_year.py` (create)

**Interfaces:**
- Consumes: `_summarize_gsd_by_year` (Task 2), `tile_acquisition_by_year` (existing local in `index_builder.run`).
- Produces: `IndexManifest.gsd_by_year: dict[int, YearGsdSummary]`, `YearAvailabilityReport.gsd_by_year: dict[int, YearGsdSummary]`, both populated by `run()`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_index_gsd_by_year.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.models import (
    IndexManifest,
    TileAcquisitionMetadata,
    YearAvailabilityReport,
    YearGsdSummary,
)


def test_index_manifest_accepts_gsd_by_year():
    m = IndexManifest(
        year_start=2024,
        year_end=2024,
        bbox="0,0,1,1",
        srs="EPSG:2180",
        years_requested=[2024],
        year_statuses=[],
        years_available_wfs=[2024],
        years_included=[2024],
        passed=True,
        gsd_by_year={2024: YearGsdSummary(histogram={"0.05": 3}, finest=0.05, coarsest=0.05)},
    )
    assert m.gsd_by_year[2024].finest == 0.05


def test_year_report_defaults_empty_gsd_by_year():
    r = YearAvailabilityReport(
        year_start=2024,
        year_end=2024,
        bbox="0,0,1,1",
        srs="EPSG:2180",
        years_requested=[2024],
        year_statuses=[],
        years_available_wfs=[2024],
        years_included=[2024],
        passed=True,
    )
    assert r.gsd_by_year == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_index_gsd_by_year.py -v`
Expected: FAIL — `IndexManifest` has no field `gsd_by_year` (pydantic rejects the kwarg).

- [ ] **Step 3: Implement**

In `src/satmap_dataset/models.py`, add to `IndexManifest` (after `tile_acquisition_by_year`, line ~101):

```python
    gsd_by_year: dict[int, YearGsdSummary] = Field(default_factory=dict)
```

Add to `YearAvailabilityReport` (after `years_excluded_with_reason` / alongside its other fields):

```python
    gsd_by_year: dict[int, YearGsdSummary] = Field(default_factory=dict)
```

In `src/satmap_dataset/pipeline/index_builder.py`, after `tile_acquisition_by_year` is built (line ~164) add:

```python
    gsd_by_year = _summarize_gsd_by_year(tile_acquisition_by_year)
```

Pass `gsd_by_year=gsd_by_year` into the `IndexManifest(...)` constructor (after `tile_acquisition_by_year=...`, line ~232) and into the `YearAvailabilityReport(...)` constructor (line ~240 block).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_index_gsd_by_year.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full suite (regression / fixture check)**

Run: `pytest -q`
Expected: PASS — existing index/manifest fixtures still validate (new fields are optional).

- [ ] **Step 6: Commit**

```bash
git add src/satmap_dataset/models.py src/satmap_dataset/pipeline/index_builder.py tests/test_index_gsd_by_year.py
git commit -m "feat(index): write gsd_by_year into index manifest and availability report"
```

---

### Task 4: End-to-end verification on real data (no new download)

**Files:**
- None (verification only).

**Interfaces:**
- Consumes: the full feature from Tasks 1-3.

- [ ] **Step 1: Re-run the index stage for an existing location**

The index stage is cheap (WFS only) and idempotent. Run it against the already-configured Poznań 15 km² location:

Run: `just index-location-json location_json=configs/run/locations/poznan_15km2.json`
Expected: exit 0; prints the path to `index_manifest.json`.

- [ ] **Step 2: Confirm GSD appears per available year**

```bash
python3 -c "
import json
m = json.load(open('artifacts_poznan_15km2/index_manifest.json'))
for y, s in sorted(m['gsd_by_year'].items()):
    print(y, s['finest'], s['histogram'])
"
```

Expected: 2021/2022/2024 finest `0.05`; 2014/2017/2019/2020/2023/2025 finest `0.25`; histograms non-empty for available years.

- [ ] **Step 3: Confirm the availability report also carries it**

```bash
python3 -c "
import json
r = json.load(open('artifacts_poznan_15km2/year_availability_report.json'))
print('gsd_by_year keys:', sorted(r['gsd_by_year']))
"
```

Expected: keys match the included years.

---

## Self-Review

**Spec coverage:**
- Per-tile `gsd` on `TileAcquisitionMetadata` → Task 1. ✓
- `piksel` extraction in `wfs_client` → Task 1. ✓
- `YearGsdSummary` (histogram + finest/coarsest) → Task 2. ✓
- `_summarize_gsd_by_year` aggregation in `index_builder` → Task 2. ✓
- `gsd_by_year` on `IndexManifest` + `YearAvailabilityReport`, wired in `run()` → Task 3. ✓
- Tests: parsing, missing-tag, aggregation mixed/all-none, manifest acceptance, backward-compat full suite → Tasks 1-3. ✓
- Out of scope (download/render manifests, backfill, CLI) — not present in any task. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code. ✓

**Type consistency:** `gsd: float | None`, `_parse_float_or_none`, `YearGsdSummary(histogram/finest/coarsest)`, `_summarize_gsd_by_year`, `_gsd_key` used identically across tasks. ✓
