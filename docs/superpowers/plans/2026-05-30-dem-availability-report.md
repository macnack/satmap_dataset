# DEM Availability Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A read-only, location-based command that queries all NMT/NMPT skorowidz endpoints across all advertised years and writes `artifacts_<slug>/dem_availability.json` (+ a console table) reporting, per (product, datum, year), which tiles cover the AOI and the coverage % — telling the user what data is and is not available, without downloading rasters.

**Architecture:** A new `pipeline/dem_availability.py` reuses `geoportal/dem_skorowidz_client.py` (`year_typenames`/`tiles_for_year`, with the verified EPSG:2180 axis-swap) to probe coverage. Coverage % is computed by sampling a grid over the AOI against the returned tile bboxes (in the same swapped coordinate space, so the area ratio is orientation-invariant). New `DemAvailabilityConfig`, `DemAvailabilityReport`/`DemAvailabilityEntry`, three CLI commands mirroring the existing `*-location-json` flavors, and a `just dem-availability` task.

**Tech Stack:** Python ≥3.10, Pydantic v2, httpx (async, via the existing WFS client), numpy (grid coverage), Typer, pytest. No GDAL, no downloads.

**Reference spec:** `docs/superpowers/specs/2026-05-30-dem-availability-report-design.md`

---

## File Structure

- Modify `src/satmap_dataset/models.py` — `DemAvailabilityEntry`, `DemAvailabilityReport`.
- Modify `src/satmap_dataset/config.py` — `DemAvailabilityConfig`.
- Create `src/satmap_dataset/pipeline/dem_availability.py` — coverage helper + `run`.
- Modify `src/satmap_dataset/cli.py` — 3 commands + builder + console table helper.
- Modify `Justfile` — `dem-availability` / `dem-availability-all` tasks.
- Tests: `tests/test_dem_availability.py` (models + config + coverage + run), `tests/test_dem_availability_cli.py`.

---

## Task 1: Availability models

**Files:**
- Modify: `src/satmap_dataset/models.py`
- Test: `tests/test_dem_availability.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_dem_availability.py`:

```python
from satmap_dataset.models import DemAvailabilityEntry, DemAvailabilityReport


def test_availability_report_round_trip():
    report = DemAvailabilityReport(
        aoi_bbox="0,0,10,10", srs="EPSG:2180",
        entries=[
            DemAvailabilityEntry(
                product="nmpt", datum="evrf2007", year=2024,
                godla=["N-33-130-D-a-3-3", "N-33-130-D-a-3-4"], tile_count=2,
                formats=["asc", "xyz.zip"], coverage="full", coverage_pct=100.0,
                acquisition_dates=["2024-03-01"],
            ),
            DemAvailabilityEntry(
                product="nmpt", datum="evrf2007", year=2020,
                godla=[], tile_count=0, formats=[], coverage="none", coverage_pct=0.0,
                acquisition_dates=[],
            ),
        ],
        errors={"nmt|kron86": "capabilities timeout"},
        full_coverage_options=[{"product": "nmpt", "datum": "evrf2007", "year": 2024}],
    )
    restored = DemAvailabilityReport.model_validate_json(report.model_dump_json())
    assert restored.kind == "dem_availability"
    assert restored.entries[0].coverage == "full"
    assert restored.entries[0].coverage_pct == 100.0
    assert restored.entries[1].coverage == "none"
    assert restored.errors == {"nmt|kron86": "capabilities timeout"}
    assert restored.full_coverage_options[0]["year"] == 2024
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_dem_availability.py -q`
Expected: FAIL — `ImportError: cannot import name 'DemAvailabilityEntry'`.

- [ ] **Step 3: Implement the models**

Append to `src/satmap_dataset/models.py` (end of file; `datetime`, `Any`, `Literal`, `BaseModel`, `Field`, `_utc_now` are already imported):

```python
class DemAvailabilityEntry(BaseModel):
    product: Literal["nmt", "nmpt"]
    datum: Literal["evrf2007", "kron86"]
    year: int = Field(..., ge=1900)
    godla: list[str] = Field(default_factory=list)
    tile_count: int = Field(default=0, ge=0)
    formats: list[str] = Field(default_factory=list)
    coverage: Literal["full", "partial", "none"] = "none"
    coverage_pct: float = 0.0
    acquisition_dates: list[str] = Field(default_factory=list)


class DemAvailabilityReport(BaseModel):
    kind: Literal["dem_availability"] = "dem_availability"
    generated_at: datetime = Field(default_factory=_utc_now)
    provider: str = "geoportal"
    aoi_bbox: str
    srs: str
    entries: list[DemAvailabilityEntry] = Field(default_factory=list)
    errors: dict[str, str] = Field(default_factory=dict)
    full_coverage_options: list[dict[str, Any]] = Field(default_factory=list)
    run_parameters: dict[str, Any] = Field(default_factory=dict)
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_dem_availability.py -q` (1 pass). Then `pytest -q` (full suite stays green; an unrelated `test_osm_*` may exist from concurrent work — ignore OSM-only issues).

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/models.py tests/test_dem_availability.py
git commit -m "feat(dem): availability report models"
```

---

## Task 2: `DemAvailabilityConfig`

**Files:**
- Modify: `src/satmap_dataset/config.py`
- Test: `tests/test_dem_availability.py`

- [ ] **Step 1: Add failing tests**

APPEND to `tests/test_dem_availability.py`:

```python
import pytest
from pydantic import ValidationError
from satmap_dataset.config import DemAvailabilityConfig


def test_availability_config_defaults_and_validation():
    cfg = DemAvailabilityConfig(bbox="0,0,10,10")
    assert cfg.products == ["nmt", "nmpt"]
    assert cfg.datums == ["evrf2007", "kron86"]
    assert cfg.year_start is None and cfg.year_end is None
    with pytest.raises(ValidationError):
        DemAvailabilityConfig(bbox="10,10,0,0")  # bad bbox order
    with pytest.raises(ValidationError):
        DemAvailabilityConfig(bbox="0,0,10,10", products=["lidar"])
    with pytest.raises(ValidationError):
        DemAvailabilityConfig(bbox="0,0,10,10", datums=["pl1965"])
    assert DemAvailabilityConfig(bbox="0,0,10,10", products=["NMT"]).products == ["nmt"]
    assert DemAvailabilityConfig(bbox="0,0,10,10", datums=["KRON86"]).datums == ["kron86"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_dem_availability.py -k availability_config -q`
Expected: FAIL — `ImportError: cannot import name 'DemAvailabilityConfig'`.

- [ ] **Step 3: Implement the config**

Append to `src/satmap_dataset/config.py` (after `DemConfig`; reuses `_validate_bbox`, `PROVIDER_GEOPORTAL`, `ALLOWED_PROVIDERS`, `BaseModel`, `Field`, `field_validator`, `model_validator`, `Path`, `Any`):

```python
class DemAvailabilityConfig(BaseModel):
    bbox: str
    srs: str = "EPSG:2180"
    products: list[str] = Field(default_factory=lambda: ["nmt", "nmpt"])
    datums: list[str] = Field(default_factory=lambda: ["evrf2007", "kron86"])
    year_start: int | None = Field(default=None, ge=1900)
    year_end: int | None = Field(default=None, ge=1900)
    location_name: str | None = None
    timeout: float = Field(default=45.0, gt=0.0)
    retries: int = Field(default=6, ge=1, le=20)
    retry_delay: float = Field(default=1.0, gt=0.0)
    sleep_min: float = Field(default=0.6, ge=0.0)
    sleep_max: float = Field(default=2.2, ge=0.0)
    output_json: Path = Path("artifacts/dem_availability.json")
    provider: str = PROVIDER_GEOPORTAL
    provider_options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value: str) -> str:
        return _validate_bbox(value)

    @field_validator("products")
    @classmethod
    def validate_products(cls, value: list[str]) -> list[str]:
        allowed = {"nmt", "nmpt"}
        normalized = [str(v).strip().lower() for v in value]
        if not normalized:
            raise ValueError("products must not be empty")
        bad = [v for v in normalized if v not in allowed]
        if bad:
            raise ValueError(f"products must be a subset of {sorted(allowed)}; got {bad}")
        seen: list[str] = []
        for v in normalized:
            if v not in seen:
                seen.append(v)
        return seen

    @field_validator("datums")
    @classmethod
    def validate_datums(cls, value: list[str]) -> list[str]:
        allowed = {"evrf2007", "kron86"}
        normalized = [str(v).strip().lower() for v in value]
        if not normalized:
            raise ValueError("datums must not be empty")
        bad = [v for v in normalized if v not in allowed]
        if bad:
            raise ValueError(f"datums must be a subset of {sorted(allowed)}; got {bad}")
        seen: list[str] = []
        for v in normalized:
            if v not in seen:
                seen.append(v)
        return seen

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        if value not in ALLOWED_PROVIDERS:
            raise ValueError(f"provider must be one of {sorted(ALLOWED_PROVIDERS)}")
        return value

    @model_validator(mode="after")
    def validate_invariants(self) -> "DemAvailabilityConfig":
        if self.sleep_max < self.sleep_min:
            raise ValueError("sleep_max must be >= sleep_min")
        if self.year_start is not None and self.year_end is not None and self.year_end < self.year_start:
            raise ValueError("year_end must be >= year_start")
        return self

    @property
    def requested_years(self) -> list[int] | None:
        if self.year_start is None or self.year_end is None:
            return None
        return list(range(self.year_start, self.year_end + 1))
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_dem_availability.py -q` (all pass). Then `pytest -q` (green).

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/config.py tests/test_dem_availability.py
git commit -m "feat(dem): DemAvailabilityConfig"
```

---

## Task 3: Coverage helper

**Files:**
- Create: `src/satmap_dataset/pipeline/dem_availability.py`
- Test: `tests/test_dem_availability.py`

- [ ] **Step 1: Add failing tests**

APPEND to `tests/test_dem_availability.py`:

```python
from satmap_dataset.pipeline import dem_availability as dav


def test_coverage_full_partial_none():
    aoi = (0.0, 0.0, 100.0, 100.0)
    assert dav._coverage_pct(aoi, [(0.0, 0.0, 100.0, 100.0)]) == 100.0
    half = dav._coverage_pct(aoi, [(0.0, 0.0, 50.0, 100.0)])
    assert 45.0 <= half <= 55.0
    assert dav._coverage_pct(aoi, []) == 0.0
    # two tiles tiling the AOI -> full
    two = dav._coverage_pct(aoi, [(0.0, 0.0, 50.0, 100.0), (50.0, 0.0, 100.0, 100.0)])
    assert two == 100.0


def test_classify_coverage():
    assert dav._classify(100.0) == "full"
    assert dav._classify(99.95) == "full"
    assert dav._classify(60.0) == "partial"
    assert dav._classify(0.0) == "none"


def test_formats_from_urls():
    urls = [
        "https://x/a_M-1-1.asc",
        "https://x/b_M-1-2.xyz.zip",
        "https://x/c_M-1-3.zip",
        "https://x/d_M-1-4.xyz",
        "https://x/e_M-1-5.tif",
    ]
    assert dav._formats_from_urls(urls) == ["asc", "tif", "xyz", "xyz.zip", "zip"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_dem_availability.py -k "coverage or classify or formats" -q`
Expected: FAIL — `ModuleNotFoundError: ...pipeline.dem_availability`.

- [ ] **Step 3: Implement the helpers**

Create `src/satmap_dataset/pipeline/dem_availability.py`:

```python
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from satmap_dataset.config import DemAvailabilityConfig
from satmap_dataset.geoportal import dem_skorowidz_client
from satmap_dataset.geoportal.http import RetryPolicy
from satmap_dataset.models import DemAvailabilityEntry, DemAvailabilityReport

logger = logging.getLogger("satmap_dataset.dem_availability")

_FULL_THRESHOLD = 99.9


def _parse_bbox(value: str) -> tuple[float, float, float, float]:
    parts = [float(p.strip()) for p in value.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must have format xmin,ymin,xmax,ymax")
    return (parts[0], parts[1], parts[2], parts[3])


def _coverage_pct(
    aoi: tuple[float, float, float, float],
    tile_bboxes: list[tuple[float, float, float, float]],
    *,
    grid: int = 200,
) -> float:
    """Percent of the AOI rectangle covered by the union of tile rectangles.

    Both the AOI and the tile bboxes must be in the SAME coordinate convention
    (the report uses the swapped WFS query space for both, so the ratio is
    orientation-invariant). Computed by sampling a ``grid`` x ``grid`` lattice of
    cell centres over the AOI — no geometry dependency.
    """
    import numpy as np

    a0, b0, a1, b1 = aoi
    if a1 <= a0 or b1 <= b0:
        return 0.0
    if not tile_bboxes:
        return 0.0
    ax = a0 + (np.arange(grid) + 0.5) * (a1 - a0) / grid
    by = b0 + (np.arange(grid) + 0.5) * (b1 - b0) / grid
    gx, gy = np.meshgrid(ax, by)
    covered = np.zeros((grid, grid), dtype=bool)
    for t0, u0, t1, u1 in tile_bboxes:
        lo_a, hi_a = (t0, t1) if t0 <= t1 else (t1, t0)
        lo_b, hi_b = (u0, u1) if u0 <= u1 else (u1, u0)
        covered |= (gx >= lo_a) & (gx <= hi_a) & (gy >= lo_b) & (gy <= hi_b)
    return float(round(covered.mean() * 100.0, 1))


def _classify(pct: float) -> str:
    if pct >= _FULL_THRESHOLD:
        return "full"
    if pct > 0.0:
        return "partial"
    return "none"


def _formats_from_urls(urls: list[str]) -> list[str]:
    found: set[str] = set()
    for url in urls:
        name = Path(url).name.lower()
        if name.endswith(".xyz.zip"):
            found.add("xyz.zip")
        elif name.endswith(".zip"):
            found.add("zip")
        elif name.endswith(".xyz"):
            found.add("xyz")
        elif name.endswith(".asc"):
            found.add("asc")
        elif name.endswith((".tif", ".tiff")):
            found.add("tif")
    return sorted(found)
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_dem_availability.py -q` (all pass). Then `pytest -q` (green).

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/pipeline/dem_availability.py tests/test_dem_availability.py
git commit -m "feat(dem): availability coverage/format helpers"
```

---

## Task 4: `dem_availability.run` orchestration

**Files:**
- Modify: `src/satmap_dataset/pipeline/dem_availability.py`
- Test: `tests/test_dem_availability.py`

- [ ] **Step 1: Add failing tests**

APPEND to `tests/test_dem_availability.py`:

```python
import json
from satmap_dataset.models import YearStatus


def _patch_av(monkeypatch, *, years_by_combo, tiles_for):
    async def _yt(product, datum, options=None, *, timeout=45.0, retry_policy=None):
        key = f"{product}|{datum}"
        if isinstance(years_by_combo.get(key), Exception):
            raise years_by_combo[key]
        return {y: f"gugik:Skorowidz{product.upper()}{y}" for y in years_by_combo.get(key, [])}

    async def _tf(product, datum, year, bbox, srs, *, year_to_typename, options=None, timeout=45.0, retry_policy=None):
        tiles, bboxes = tiles_for(product, datum, year)
        status = YearStatus(year=year, typename_exists=True, feature_count=len(tiles),
                            status="has_features" if tiles else "zero_features")
        acq = {tid: {"acquisition_date": f"{year}-03-01"} for tid in tiles}
        return status, dict(tiles), dict(bboxes), acq

    monkeypatch.setattr(dem_availability.dem_skorowidz_client, "year_typenames", _yt)
    monkeypatch.setattr(dem_availability.dem_skorowidz_client, "tiles_for_year", _tf)


def test_run_builds_report(tmp_path, monkeypatch):
    years = {
        "nmt|evrf2007": [2019],
        "nmpt|evrf2007": [2019, 2024],
        "nmt|kron86": RuntimeError("caps down"),
        "nmpt|kron86": [],
    }

    def tiles_for(product, datum, year):
        # nmpt/evrf2007 2024 -> full (one tile spanning AOI); 2019 -> half
        if product == "nmpt" and datum == "evrf2007" and year == 2024:
            return {"g1": "https://x/a_g1.asc"}, {"g1": [0.0, 0.0, 100.0, 100.0]}
        if product == "nmpt" and datum == "evrf2007" and year == 2019:
            return {"g2": "https://x/a_g2.asc"}, {"g2": [0.0, 0.0, 50.0, 100.0]}
        if product == "nmt" and datum == "evrf2007" and year == 2019:
            return {"g3": "https://x/a_g3.xyz.zip"}, {"g3": [0.0, 0.0, 100.0, 100.0]}
        return {}, {}

    _patch_av(monkeypatch, years_by_combo=years, tiles_for=tiles_for)
    from satmap_dataset.config import DemAvailabilityConfig
    cfg = DemAvailabilityConfig(bbox="0,0,100,100", output_json=tmp_path / "av.json")
    code, path = dem_availability.run(cfg)
    assert code == 0
    report = DemAvailabilityReport.model_validate_json(Path(path).read_text())
    by = {(e.product, e.datum, e.year): e for e in report.entries}
    assert by[("nmpt", "evrf2007", 2024)].coverage == "full"
    assert by[("nmpt", "evrf2007", 2024)].formats == ["asc"]
    assert by[("nmpt", "evrf2007", 2019)].coverage == "partial"
    assert by[("nmt", "evrf2007", 2019)].formats == ["xyz.zip"]
    assert report.errors["nmt|kron86"].startswith("caps")
    assert {"product": "nmpt", "datum": "evrf2007", "year": 2024} in report.full_coverage_options
    # acquisition dates surfaced
    assert by[("nmpt", "evrf2007", 2024)].acquisition_dates == ["2024-03-01"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_dem_availability.py -k run_builds_report -q`
Expected: FAIL — `AttributeError: module ... has no attribute 'run'`.

- [ ] **Step 3: Implement `run`**

Append to `src/satmap_dataset/pipeline/dem_availability.py`:

```python
def _swap_bbox(bbox: tuple[float, float, float, float]) -> str:
    xmin, ymin, xmax, ymax = bbox
    return f"{ymin},{xmin},{ymax},{xmax}"


async def _run_async(config: DemAvailabilityConfig) -> tuple[int, Path]:
    retry_policy = RetryPolicy(max_attempts=config.retries, backoff_seconds=config.retry_delay)
    options = dict(config.provider_options)
    bbox = _parse_bbox(config.bbox)
    swap = bool(options.get("wfs_swap_bbox_axes", config.srs.strip().upper() == "EPSG:2180"))
    query_bbox = _swap_bbox(bbox) if swap else config.bbox
    cov_aoi = _parse_bbox(query_bbox)  # coverage computed in the same (query) space
    year_filter = config.requested_years  # None = all advertised

    entries: list[DemAvailabilityEntry] = []
    errors: dict[str, str] = {}

    for product in config.products:
        for datum in config.datums:
            combo = f"{product}|{datum}"
            try:
                year_to_typename = await dem_skorowidz_client.year_typenames(
                    product, datum, options, timeout=config.timeout, retry_policy=retry_policy
                )
            except Exception as exc:  # noqa: BLE001 - record and continue
                errors[combo] = str(exc)
                continue
            years = sorted(year_to_typename)
            if year_filter is not None:
                years = [y for y in years if y in set(year_filter)]
            for year in years:
                try:
                    _status, tiles, tile_bboxes, tile_acq = await dem_skorowidz_client.tiles_for_year(
                        product, datum, year, query_bbox, config.srs,
                        year_to_typename=year_to_typename, options=options,
                        timeout=config.timeout, retry_policy=retry_policy,
                    )
                except Exception as exc:  # noqa: BLE001
                    errors[f"{combo}|{year}"] = str(exc)
                    continue
                pct = _coverage_pct(cov_aoi, [tuple(v) for v in tile_bboxes.values()])
                dates = sorted({
                    str(meta.get("acquisition_date"))
                    for meta in tile_acq.values()
                    if meta.get("acquisition_date")
                })
                entries.append(
                    DemAvailabilityEntry(
                        product=product, datum=datum, year=year,
                        godla=sorted(tiles.keys()), tile_count=len(tiles),
                        formats=_formats_from_urls(list(tiles.values())),
                        coverage=_classify(pct), coverage_pct=pct,
                        acquisition_dates=dates,
                    )
                )

    full_options = [
        {"product": e.product, "datum": e.datum, "year": e.year}
        for e in entries if e.coverage == "full"
    ]
    report = DemAvailabilityReport(
        provider="geoportal", aoi_bbox=config.bbox, srs=config.srs,
        entries=entries, errors=errors, full_coverage_options=full_options,
        run_parameters=config.model_dump(mode="json"),
    )
    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    config.output_json.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    logger.info(
        "DEM availability: entries=%s full=%s errors=%s",
        len(entries), len(full_options), len(errors),
    )
    return 0, config.output_json


def run(config: DemAvailabilityConfig) -> tuple[int, Path]:
    return asyncio.run(_run_async(config))
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_dem_availability.py -q` (all pass). Then `pytest -q` (green).

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/pipeline/dem_availability.py tests/test_dem_availability.py
git commit -m "feat(dem): availability report orchestration (run)"
```

---

## Task 5: CLI commands + console table + justfile

**Files:**
- Modify: `src/satmap_dataset/cli.py`
- Modify: `Justfile`
- Test: `tests/test_dem_availability_cli.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dem_availability_cli.py`:

```python
import json
from pathlib import Path

from typer.testing import CliRunner

from satmap_dataset.cli import app, _build_dem_availability_config_from_base_and_location

runner = CliRunner()


def test_dem_availability_json_command(tmp_path, monkeypatch):
    captured = {}

    def _fake_run(config):
        captured["config"] = config
        out = tmp_path / "av.json"
        out.write_text(json.dumps({
            "kind": "dem_availability", "aoi_bbox": config.bbox, "srs": "EPSG:2180",
            "entries": [], "errors": {}, "full_coverage_options": [],
        }))
        return (0, out)

    monkeypatch.setattr("satmap_dataset.pipeline.dem_availability.run", _fake_run)
    params = tmp_path / "p.json"
    params.write_text(json.dumps({
        "bbox": "210300,521900,210500,522100",
        "output_json": str(tmp_path / "av.json"),
    }))
    result = runner.invoke(app, ["dem-availability-json", str(params)])
    assert result.exit_code == 0
    assert captured["config"].products == ["nmt", "nmpt"]
    assert result.stdout.strip().splitlines()[-1].endswith("av.json")


def test_dem_availability_location_builder(tmp_path):
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"products": ["nmpt"]}))
    loc = tmp_path / "location_x.json"
    loc.write_text(json.dumps({"location_name": "Test", "center_lat": 52.4, "center_lon": 16.9, "square_km": 1.0}))
    cfg = _build_dem_availability_config_from_base_and_location(base_json=base, location_json=loc)
    assert cfg.products == ["nmpt"]
    assert cfg.bbox  # center resolved
    assert str(cfg.output_json).endswith("dem_availability.json")
    assert "artifacts_test" in str(cfg.output_json)


def test_dem_availability_json_bad_config_exit_2(tmp_path):
    params = tmp_path / "p.json"
    params.write_text(json.dumps({"bbox": "10,10,0,0"}))
    result = runner.invoke(app, ["dem-availability-json", str(params)])
    assert result.exit_code == 2
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_dem_availability_cli.py -q`
Expected: FAIL — `ImportError: cannot import name '_build_dem_availability_config_from_base_and_location'`.

- [ ] **Step 3a: Imports**

In `src/satmap_dataset/cli.py`, add `DemAvailabilityConfig` to the `from satmap_dataset.config import (...)` block (alphabetical, next to `DemConfig`), and add `dem_availability` to the `from satmap_dataset.pipeline import ...` imports (next to `dem`).

- [ ] **Step 3b: Console table helper**

Add this helper near the other private helpers in `cli.py` (e.g. after `_finish`):

```python
def _print_availability_table(report) -> None:
    rows = sorted(
        [e for e in report.entries if e.tile_count > 0],
        key=lambda e: (e.product, e.datum, e.year),
    )
    console.print(f"[cyan]DEM availability[/cyan] AOI={report.aoi_bbox} ({report.srs})")
    console.print("  product datum     year  tiles  coverage      formats")
    for e in rows:
        cov = e.coverage if e.coverage != "partial" else f"partial({e.coverage_pct:g}%)"
        console.print(
            f"  {e.product:<7} {e.datum:<9} {e.year}   {e.tile_count:<5} {cov:<13} {','.join(e.formats)}"
        )
    empty = sorted(
        {(e.product, e.datum) for e in report.entries}
        - {(e.product, e.datum) for e in rows}
    )
    for product, datum in empty:
        missing = sorted(e.year for e in report.entries if e.product == product and e.datum == datum and e.tile_count == 0)
        if missing:
            console.print(f"  [yellow]no data:[/yellow] {product}/{datum} {missing}")
    for combo, msg in report.errors.items():
        console.print(f"  [red]error:[/red] {combo}: {msg}")
```

- [ ] **Step 3c: Builder + commands**

After `_build_dem_config_from_base_and_location` in `cli.py`, add:

```python
def _build_dem_availability_config_from_base_and_location(*, base_json: Path, location_json: Path) -> DemAvailabilityConfig:
    base_payload = _load_params_json_dict(base_json)
    location_payload = _load_params_json_dict(location_json)
    merged: dict[str, object] = dict(base_payload)
    merged.update(location_payload)
    repo_root = base_json.resolve().parents[2] if len(base_json.resolve().parents) >= 3 else Path.cwd().resolve()
    merged = _apply_location_paths_policy(merged, repo_root)
    merged = _resolve_json_center_bbox(merged, required=True)
    artifacts_dir = merged.get("artifacts_dir")
    if artifacts_dir is not None:
        merged.setdefault("output_json", str(Path(str(artifacts_dir)) / "dem_availability.json"))
    return DemAvailabilityConfig.model_validate(merged)
```

At the end of `cli.py`, add the three commands:

```python
@app.command("dem-availability-json")
def dem_availability_json_command(
    params_json: Path = typer.Argument(..., help="JSON with DemAvailabilityConfig fields (center_lat/lon + square_km|area_km2 supported)."),
) -> None:
    try:
        payload = _load_params_json_dict(params_json)
        payload = _resolve_json_center_bbox(payload, required=True)
        config = DemAvailabilityConfig.model_validate(payload)
    except typer.BadParameter as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=2) from error
    except ValidationError as error:
        _print_validation_error(error)
        raise typer.Exit(code=2) from error

    exit_code, artifact_path = dem_availability.run(config)
    from satmap_dataset.models import DemAvailabilityReport
    _print_availability_table(DemAvailabilityReport.model_validate_json(artifact_path.read_text(encoding="utf-8")))
    _finish(exit_code, artifact_path)


@app.command("dem-availability-location-json")
def dem_availability_location_json_command(
    location_json: Path = typer.Argument(..., help="Location JSON (location_name, center_lat, center_lon)."),
    base_json: Path = typer.Option(Path("configs/run/base.json"), "--base-json", help="Base JSON with shared parameters."),
) -> None:
    try:
        config = _build_dem_availability_config_from_base_and_location(base_json=base_json, location_json=location_json)
    except typer.BadParameter as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=2) from error
    except ValidationError as error:
        _print_validation_error(error)
        raise typer.Exit(code=2) from error

    exit_code, artifact_path = dem_availability.run(config)
    from satmap_dataset.models import DemAvailabilityReport
    _print_availability_table(DemAvailabilityReport.model_validate_json(artifact_path.read_text(encoding="utf-8")))
    _finish(exit_code, artifact_path)


@app.command("dem-availability-all-location-json")
def dem_availability_all_location_json_command(
    locations_dir: Path = typer.Option(Path("configs/run/locations"), "--locations-dir", help="Directory with location JSON files."),
    base_json: Path = typer.Option(Path("configs/run/base.json"), "--base-json", help="Base JSON with shared parameters."),
    continue_on_error: bool = typer.Option(False, "--continue-on-error/--no-continue-on-error", help="Continue when one location fails."),
) -> None:
    location_files = _location_files_or_exit(locations_dir)
    failures: list[str] = []
    for location_json in location_files:
        console.print(f"[cyan]dem-availability:[/cyan] {location_json}")
        try:
            config = _build_dem_availability_config_from_base_and_location(base_json=base_json, location_json=location_json)
        except (typer.BadParameter, ValidationError) as error:
            if isinstance(error, ValidationError):
                _print_validation_error(error)
            else:
                console.print(f"[red]{error}[/red]")
            failures.append(f"{location_json}: invalid")
            if not continue_on_error:
                raise typer.Exit(code=2) from error
            continue
        exit_code, artifact_path = dem_availability.run(config)
        console.print(str(artifact_path))
        if exit_code != 0:
            failures.append(f"{location_json}: exit={exit_code}")
            if not continue_on_error:
                raise typer.Exit(code=exit_code)
    if failures:
        console.print("[yellow]dem-availability-all finished with failures:[/yellow]")
        for entry in failures:
            console.print(f"- {entry}")
        raise typer.Exit(code=1)
    raise typer.Exit(code=0)
```

Verify the referenced helpers exist with these names: `_load_params_json_dict`, `_resolve_json_center_bbox`, `_apply_location_paths_policy`, `_finish`, `_print_validation_error`, `_location_files_or_exit`, `console`, `ValidationError`, `typer`. If any differ, adapt; if a referenced helper is genuinely absent, STOP and report NEEDS_CONTEXT.

- [ ] **Step 3d: Justfile**

Append to `Justfile` (match the existing 2-space recipe-body indentation):

```just
# Report available NMT/NMPT skorowidz data for a single location (no download)
dem-availability location_json:
  python -m satmap_dataset.cli dem-availability-location-json {{location_json}}

# Report availability for all locations in the default dir
dem-availability-all:
  python -m satmap_dataset.cli dem-availability-all-location-json
```

- [ ] **Step 4: Run tests + full suite + help**

Run: `pytest tests/test_dem_availability_cli.py -q` (3 pass). Then `pytest -q` (green). Then `python -m satmap_dataset.cli --help` and confirm `dem-availability-json`, `dem-availability-location-json`, `dem-availability-all-location-json` appear.

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/cli.py Justfile tests/test_dem_availability_cli.py
git commit -m "feat(dem): availability CLI commands + console table + just task"
```

---

## Task 6: Live smoke (manual, outside sandbox)

**Files:** none (manual — the dev sandbox blocks geoportal).

- [ ] **Step 1: Run availability for Przeźmierowo**

```bash
python -m satmap_dataset.cli dem-availability-json /dev/stdin <<'JSON'
{"location_name":"Przezmierowo","center_lat":52.426178,"center_lon":16.785372,"square_km":1.0,
 "output_json":"./dem_av_przezmierowo/dem_availability.json"}
JSON
```

Expected: exit `0`; a console table listing rows per `(product, datum, year)` with `full`/`partial(%)`/coverage and formats; `full_coverage_options` includes `nmpt/evrf2007/2024`; the last stdout line is the JSON path. Cross-check against the known facts: NMPT EVRF2007 has 2019–2025, NMT KRON86 has 2000–2019, and for this AOI NMPT 2024 is full while NMPT 2019 is ~78% partial.

- [ ] **Step 2: Confirm the JSON**

Open `dem_av_przezmierowo/dem_availability.json` and verify `entries` cover all four `(product,datum)` combos across their advertised years, with `coverage`/`coverage_pct`/`formats`/`acquisition_dates`/`godla` populated, and `full_coverage_options` listing the fully-covered triples.

---

## Self-Review

**Spec coverage:**
- Read-only skorowidz query, all 4 combos, all advertised years (optional year filter) → Task 4 (`_run_async` loops products×datums×years; `requested_years` filter). ✓
- Coverage full/partial/none + % from tile-bbox union → Task 3 (`_coverage_pct`/`_classify`), computed in the consistent swapped space (Task 4 `cov_aoi`). ✓
- Per-entry godła/tile_count/formats/acquisition_dates → Task 4. ✓
- `errors` for capability/feature failures, never fatal, exit 0 → Task 4. ✓
- `full_coverage_options` → Task 4. ✓
- Models + config + validators → Tasks 1–2. ✓
- Location-based CLI (3 flavors) + builder writing to `artifacts_<slug>/dem_availability.json` + console table + last-line artifact path → Task 5. ✓
- `just dem-availability` / `dem-availability-all` → Task 5 Step 3d. ✓
- Axis-swap reused for the query; coverage orientation-invariant → Task 4. ✓
- Live smoke validating against known Przeźmierowo facts → Task 6. ✓

**Placeholder scan:** No TBD/TODO; every code step is complete. ✓

**Type consistency:** `_coverage_pct(aoi, tile_bboxes, *, grid)`, `_classify(pct)`, `_formats_from_urls(urls)`, `_swap_bbox(bbox)`, `dem_availability.run(config)`; `DemAvailabilityConfig` (products/datums/year_start/year_end/output_json/requested_years); `DemAvailabilityEntry`/`DemAvailabilityReport` fields; `dem_skorowidz_client.year_typenames`/`tiles_for_year` signatures — all consistent across Tasks 1–6. The `tiles_for_year` 4-tuple `(YearStatus, tiles, tile_bboxes, tile_acq)` matches `wfs_client.get_year_tiles`. ✓

**Reuse safety:** No changes to `wfs_client`, `dem_skorowidz`, or `dem` — purely additive. Reuses the verified axis-swap convention. ✓
