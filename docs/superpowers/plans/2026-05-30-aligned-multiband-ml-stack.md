# Aligned Multi-band ML Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `dem-stack` feature that assembles a single float32 multi-band GeoTIFF (bands `[R,G,B,NMT,NMPT,nDSM,valid_mask]`) per location on the orthophoto render grid, plus a JSON manifest describing band order + provenance, for one-file ML dataloading.

**Architecture:** A new `pipeline/dem_stack.py` auto-discovers the rendered ortho + skorowidz DEM files for a location, takes the RGB raster as the reference grid, resamples NMT/NMPT to it with `gdalwarp`, computes nDSM + a shared `valid_mask` in numpy, writes a plain multi-band TIFF with `tifffile`, then georeferences it with `gdal_translate -a_srs -a_ullr`. New `StackConfig`, `StackManifest`/`StackBandDescriptor`, CLI commands, and a `just dem-stack` task.

**Tech Stack:** Python ≥3.10, Pydantic v2, numpy + tifffile (assembly), GDAL CLI (`gdalinfo`/`gdalwarp`/`gdal_translate`), Typer, pytest. No new dependencies.

**Reference spec:** `docs/superpowers/specs/2026-05-30-aligned-multiband-ml-stack-design.md`

---

## File Structure

- Modify `src/satmap_dataset/models.py` — `StackBandDescriptor`, `StackManifest`.
- Modify `src/satmap_dataset/config.py` — `StackConfig`.
- Create `src/satmap_dataset/pipeline/dem_stack.py` — grid/assemble/write helpers + `run`.
- Modify `src/satmap_dataset/cli.py` — `dem-stack-json` / `dem-stack-location-json` + builder.
- Modify `Justfile`, `.gitignore`, `scripts/manage_location_roots.py` — `stack_<slug>` root.
- Tests: `tests/test_stack_models.py`, `tests/test_stack_config.py`, `tests/test_dem_stack.py`, `tests/test_dem_stack_cli.py`.

Reference facts: render output is `rendered_<slug>/year_<year>.tiff`; skorowidz DEM native is `dem_<slug>/skorowidz/<product>_<datum>/native/year_<year>.tif`; `gdalinfo -json` exposes `size:[W,H]` and `cornerCoordinates.upperLeft/lowerRight`.

---

## Task 1: Stack manifest models

**Files:**
- Modify: `src/satmap_dataset/models.py`
- Test: `tests/test_stack_models.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_stack_models.py`:

```python
from satmap_dataset.models import StackBandDescriptor, StackManifest


def test_stack_manifest_round_trip():
    manifest = StackManifest(
        location_name="Poznan", stack_path="stack_poznan/stack_2024.tif",
        crs="EPSG:2180", width=4000, height=4000, fill_value=0.0,
        bands=[
            StackBandDescriptor(index=1, name="red", role="rgb", unit="DN_0_255",
                                source="rendered_poznan/year_2024.tiff", year=2024),
            StackBandDescriptor(index=4, name="nmt", role="dtm", unit="m",
                                source="dem_poznan/.../year_2024.tif", year=2024, datum="evrf2007"),
            StackBandDescriptor(index=6, name="ndsm", role="object_height", unit="m", derived="nmpt-nmt"),
            StackBandDescriptor(index=7, name="valid_mask", role="mask", unit="bool"),
        ],
        normalization_hint={"rgb": "/255", "elevation": "z-score per dataset"},
        passed=True,
    )
    restored = StackManifest.model_validate_json(manifest.model_dump_json())
    assert restored.kind == "ml_stack"
    assert restored.dtype == "float32"
    assert restored.width == 4000
    assert restored.bands[0].role == "rgb"
    assert restored.bands[2].derived == "nmpt-nmt"
    assert restored.bands[3].name == "valid_mask"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_stack_models.py -q`
Expected: FAIL — `ImportError: cannot import name 'StackBandDescriptor'`.

- [ ] **Step 3: Implement the models**

Append to `src/satmap_dataset/models.py` (end of file; `datetime`, `Any`, `Literal`, `BaseModel`, `Field`, `_utc_now` already imported):

```python
class StackBandDescriptor(BaseModel):
    index: int = Field(..., ge=1)
    name: str
    role: str
    unit: str
    source: str | None = None
    year: int | None = None
    datum: str | None = None
    derived: str | None = None


class StackManifest(BaseModel):
    kind: Literal["ml_stack"] = "ml_stack"
    generated_at: datetime = Field(default_factory=_utc_now)
    provider: str = "geoportal"
    location_name: str | None = None
    stack_path: str
    crs: str
    width: int = Field(..., ge=1)
    height: int = Field(..., ge=1)
    dtype: str = "float32"
    fill_value: float = 0.0
    bands: list[StackBandDescriptor] = Field(default_factory=list)
    normalization_hint: dict[str, str] = Field(default_factory=dict)
    passed: bool = False
    run_parameters: dict[str, Any] = Field(default_factory=dict)
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_stack_models.py -q` (1 pass). Then `pytest -q` (full suite green; ignore unrelated OSM-only issues if any).

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/models.py tests/test_stack_models.py
git commit -m "feat(stack): ML stack manifest models"
```

---

## Task 2: `StackConfig`

**Files:**
- Modify: `src/satmap_dataset/config.py`
- Test: `tests/test_stack_config.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_stack_config.py`:

```python
import pytest
from pydantic import ValidationError

from satmap_dataset.config import StackConfig


def test_defaults():
    cfg = StackConfig(location_name="Poznan", rgb_year=2024, nmt_year=2024, nmpt_year=2024)
    assert cfg.bands == ["rgb", "nmt", "nmpt", "ndsm"]
    assert cfg.vertical_datum == "evrf2007"
    assert cfg.resample == "bilinear"
    assert cfg.fill_value == 0.0


def test_bands_validation():
    with pytest.raises(ValidationError):
        StackConfig(location_name="x", rgb_year=2024, bands=["lidar"])
    assert StackConfig(location_name="x", rgb_year=2024, bands=["RGB"]).bands == ["rgb"]


def test_ndsm_requires_both_years():
    with pytest.raises(ValidationError):
        StackConfig(location_name="x", rgb_year=2024, bands=["rgb", "ndsm"], nmt_year=2024)  # nmpt missing
    ok = StackConfig(location_name="x", rgb_year=2024, bands=["rgb", "ndsm"], nmt_year=2024, nmpt_year=2024)
    assert "ndsm" in ok.bands


def test_value_band_requires_its_year():
    with pytest.raises(ValidationError):
        StackConfig(location_name="x", rgb_year=2024, bands=["rgb", "nmt"])  # nmt_year missing
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_stack_config.py -q`
Expected: FAIL — `ImportError: cannot import name 'StackConfig'`.

- [ ] **Step 3: Implement the config**

Append to `src/satmap_dataset/config.py` (after `DemAvailabilityConfig`; reuse `PROVIDER_GEOPORTAL`, `ALLOWED_PROVIDERS`, `BaseModel`, `Field`, `field_validator`, `model_validator`, `Path`, `Any`):

```python
class StackConfig(BaseModel):
    location_name: str
    rgb_year: int = Field(..., ge=1900)
    nmt_year: int | None = Field(default=None, ge=1900)
    nmpt_year: int | None = Field(default=None, ge=1900)
    vertical_datum: str = "evrf2007"
    bands: list[str] = Field(default_factory=lambda: ["rgb", "nmt", "nmpt", "ndsm"])
    resample: str = "bilinear"
    render_root: Path | None = None
    dem_root: Path | None = None
    output_json: Path | None = None
    fill_value: float = 0.0
    nodata_in: float = -9999.0
    provider: str = PROVIDER_GEOPORTAL
    provider_options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("bands")
    @classmethod
    def validate_bands(cls, value: list[str]) -> list[str]:
        allowed = {"rgb", "nmt", "nmpt", "ndsm"}
        normalized = [str(v).strip().lower() for v in value]
        if not normalized:
            raise ValueError("bands must not be empty")
        bad = [v for v in normalized if v not in allowed]
        if bad:
            raise ValueError(f"bands must be a subset of {sorted(allowed)}; got {bad}")
        seen: list[str] = []
        for v in normalized:
            if v not in seen:
                seen.append(v)
        return seen

    @field_validator("vertical_datum")
    @classmethod
    def validate_vertical_datum(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized not in {"evrf2007", "kron86"}:
            raise ValueError("vertical_datum must be 'evrf2007' or 'kron86'")
        return normalized

    @field_validator("resample")
    @classmethod
    def validate_resample(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized not in {"bilinear", "nearest"}:
            raise ValueError("resample must be 'bilinear' or 'nearest'")
        return normalized

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        if value not in ALLOWED_PROVIDERS:
            raise ValueError(f"provider must be one of {sorted(ALLOWED_PROVIDERS)}")
        return value

    @model_validator(mode="after")
    def validate_invariants(self) -> "StackConfig":
        if "nmt" in self.bands and self.nmt_year is None:
            raise ValueError("nmt_year is required when 'nmt' is in bands")
        if "nmpt" in self.bands and self.nmpt_year is None:
            raise ValueError("nmpt_year is required when 'nmpt' is in bands")
        if "ndsm" in self.bands and (self.nmt_year is None or self.nmpt_year is None):
            raise ValueError("ndsm requires both nmt_year and nmpt_year")
        return self
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_stack_config.py -q` (all pass). Then `pytest -q` (green).

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/config.py tests/test_stack_config.py
git commit -m "feat(stack): StackConfig"
```

---

## Task 3: Assembly + grid helpers

**Files:**
- Create: `src/satmap_dataset/pipeline/dem_stack.py`
- Test: `tests/test_dem_stack.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dem_stack.py`:

```python
import json

import numpy as np

from satmap_dataset.pipeline import dem_stack


def test_assemble_stack_bands_order_ndsm_and_mask():
    rgb = np.array([[[10, 20]], [[30, 40]], [[50, 60]]], dtype="float32")  # (3,1,2)
    nmt = np.array([[100.0, -9999.0]], dtype="float32")   # second pixel invalid
    nmpt = np.array([[118.0, 90.0]], dtype="float32")
    stack, names = dem_stack._assemble_stack(
        rgb, nmt, nmpt, bands=["rgb", "nmt", "nmpt", "ndsm"], nodata=-9999.0, fill=0.0
    )
    # 3 rgb + nmt + nmpt + ndsm + mask = 7 bands
    assert stack.shape == (7, 1, 2)
    assert [n[0] for n in names] == ["red", "green", "blue", "nmt", "nmpt", "ndsm", "valid_mask"]
    # valid mask: pixel0 valid (both elev present), pixel1 invalid (nmt nodata)
    assert stack[6].tolist() == [[1.0, 0.0]]
    # ndsm on valid pixel0 = 118-100 = 18; invalid pixel1 filled 0
    assert stack[5].tolist() == [[18.0, 0.0]]
    # value bands filled 0 at invalid pixel1 (incl rgb)
    assert stack[0].tolist() == [[10.0, 0.0]]
    assert stack[3].tolist() == [[100.0, 0.0]]


def test_assemble_stack_band_subset():
    rgb = np.zeros((3, 1, 1), dtype="float32")
    nmt = np.array([[5.0]], dtype="float32")
    stack, names = dem_stack._assemble_stack(rgb, nmt, None, bands=["rgb", "nmt"], nodata=-9999.0, fill=0.0)
    assert [n[0] for n in names] == ["red", "green", "blue", "nmt", "valid_mask"]
    assert stack.shape == (5, 1, 1)


def test_raster_grid_parses_gdalinfo_json(tmp_path, monkeypatch):
    fake = {
        "size": [4000, 4000],
        "cornerCoordinates": {"upperLeft": [359699.75, 506900.25], "lowerRight": [361699.75, 504900.25]},
        "coordinateSystem": {"wkt": 'PROJCS[...,ID["EPSG",2180]]'},
    }

    def _fake_gdalinfo(path):
        return fake

    monkeypatch.setattr(dem_stack, "_gdalinfo_json", _fake_gdalinfo)
    grid = dem_stack._raster_grid("whatever.tif")
    assert grid == (359699.75, 504900.25, 361699.75, 506900.25, 4000, 4000, 2180)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_dem_stack.py -q`
Expected: FAIL — `ModuleNotFoundError: ...pipeline.dem_stack`.

- [ ] **Step 3: Implement the helpers**

Create `src/satmap_dataset/pipeline/dem_stack.py`:

```python
from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path

logger = logging.getLogger("satmap_dataset.dem_stack")

_RGB_BANDS = (("red", "rgb", "DN_0_255"), ("green", "rgb", "DN_0_255"), ("blue", "rgb", "DN_0_255"))


def _tool_path(name: str) -> str | None:
    import shutil

    return shutil.which(name)


def _gdalinfo_json(path: str) -> dict:
    gdalinfo = _tool_path("gdalinfo")
    if not gdalinfo:
        raise RuntimeError("Reading the reference grid requires the GDAL CLI (gdalinfo). Install GDAL.")
    result = subprocess.run([gdalinfo, "-json", str(path)], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def _raster_grid(path: str) -> tuple[float, float, float, float, int, int, int]:
    """Return (xmin, ymin, xmax, ymax, width, height, epsg) for a raster."""
    info = _gdalinfo_json(path)
    width, height = int(info["size"][0]), int(info["size"][1])
    ulx, uly = info["cornerCoordinates"]["upperLeft"]
    lrx, lry = info["cornerCoordinates"]["lowerRight"]
    wkt = info.get("coordinateSystem", {}).get("wkt", "")
    matches = re.findall(r'(?:ID|AUTHORITY)\[\s*"EPSG"\s*,\s*"?(\d+)"?\s*\]', wkt)
    epsg = int(matches[-1]) if matches else 2180
    return (float(ulx), float(lry), float(lrx), float(uly), width, height, epsg)


def _assemble_stack(
    rgb,
    nmt,
    nmpt,
    *,
    bands: list[str],
    nodata: float = -9999.0,
    fill: float = 0.0,
):
    """Build a (C,H,W) float32 stack in band order, plus a list of (name, role, unit).

    valid_mask = pixels where every present elevation layer is valid; value bands are
    filled with ``fill`` at invalid pixels; valid_mask is always the final band.
    """
    import numpy as np

    rgb = np.asarray(rgb, dtype="float32")
    _, height, width = rgb.shape

    def _valid(arr):
        a = np.asarray(arr, dtype="float64")
        return np.isfinite(a) & (a != nodata)

    valid = np.ones((height, width), dtype=bool)
    nmt_a = np.asarray(nmt, dtype="float32") if nmt is not None else None
    nmpt_a = np.asarray(nmpt, dtype="float32") if nmpt is not None else None
    if nmt_a is not None:
        valid &= _valid(nmt_a)
    if nmpt_a is not None:
        valid &= _valid(nmpt_a)
    ndsm_a = (nmpt_a - nmt_a).astype("float32") if (nmt_a is not None and nmpt_a is not None) else None

    layers: list[tuple[str, str, str, object]] = []
    for band in bands:
        if band == "rgb":
            for i, (name, role, unit) in enumerate(_RGB_BANDS):
                layers.append((name, role, unit, rgb[i]))
        elif band == "nmt":
            layers.append(("nmt", "dtm", "m", nmt_a))
        elif band == "nmpt":
            layers.append(("nmpt", "dsm", "m", nmpt_a))
        elif band == "ndsm":
            layers.append(("ndsm", "object_height", "m", ndsm_a))

    invalid = ~valid
    out_bands = []
    names: list[tuple[str, str, str]] = []
    for name, role, unit, arr in layers:
        a = np.zeros((height, width), dtype="float32") if arr is None else np.array(arr, dtype="float32")
        a[invalid] = fill
        out_bands.append(a)
        names.append((name, role, unit))
    out_bands.append(valid.astype("float32"))
    names.append(("valid_mask", "mask", "bool"))
    stack = np.stack(out_bands, axis=0).astype("float32")
    return stack, names
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_dem_stack.py -q` (3 pass). Then `pytest -q` (green).

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/pipeline/dem_stack.py tests/test_dem_stack.py
git commit -m "feat(stack): grid + assemble helpers"
```

---

## Task 4: `dem_stack.run` orchestration

**Files:**
- Modify: `src/satmap_dataset/pipeline/dem_stack.py`
- Test: `tests/test_dem_stack.py`

- [ ] **Step 1: Add failing tests**

APPEND to `tests/test_dem_stack.py`:

```python
from pathlib import Path

from satmap_dataset.config import StackConfig
from satmap_dataset.models import StackManifest


def _seed_inputs(tmp_path):
    render_root = tmp_path / "rendered_poznan"
    dem_root = tmp_path / "dem_poznan"
    (render_root).mkdir(parents=True)
    (render_root / "year_2024.tiff").write_bytes(b"RGB")
    nmt_dir = dem_root / "skorowidz" / "nmt_evrf2007" / "native"
    nmpt_dir = dem_root / "skorowidz" / "nmpt_evrf2007" / "native"
    nmt_dir.mkdir(parents=True)
    nmpt_dir.mkdir(parents=True)
    (nmt_dir / "year_2024.tif").write_bytes(b"NMT")
    (nmpt_dir / "year_2024.tif").write_bytes(b"NMPT")
    return render_root, dem_root


def _patch_io(monkeypatch):
    import numpy as np

    monkeypatch.setattr(dem_stack, "_raster_grid", lambda p: (0.0, 0.0, 2.0, 1.0, 2, 1, 2180))
    monkeypatch.setattr(dem_stack, "_align_to_grid", lambda src, out, grid, resample, nodata: Path(out).write_bytes(b"X"))

    def _read(path):
        name = Path(path).name.lower()
        if "rgb" in name or name.startswith("year_2024.tiff") or name.endswith(".tiff"):
            return np.array([[[10, 20]], [[30, 40]], [[50, 60]]], dtype="float32")
        if "nmt" in str(path):
            return np.array([[100.0, 100.0]], dtype="float32")
        return np.array([[118.0, 90.0]], dtype="float32")  # nmpt

    monkeypatch.setattr(dem_stack, "_read_raster", _read)
    captured = {}

    def _write(stack, out_path, grid, *, epsg):
        captured["shape"] = stack.shape
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"STACK")

    monkeypatch.setattr(dem_stack, "_write_geotiff", _write)
    return captured


def test_run_builds_stack_manifest(tmp_path, monkeypatch):
    render_root, dem_root = _seed_inputs(tmp_path)
    captured = _patch_io(monkeypatch)
    cfg = StackConfig(
        location_name="Poznan", rgb_year=2024, nmt_year=2024, nmpt_year=2024,
        render_root=render_root, dem_root=dem_root,
        output_json=tmp_path / "stack_poznan" / "stack_2024.json",
    )
    code, path = dem_stack.run(cfg)
    assert code == 0
    assert captured["shape"] == (7, 1, 2)
    m = StackManifest.model_validate_json(Path(path).read_text())
    assert m.kind == "ml_stack"
    assert [b.name for b in m.bands] == ["red", "green", "blue", "nmt", "nmpt", "ndsm", "valid_mask"]
    assert m.bands[3].source.endswith("nmt_evrf2007/native/year_2024.tif")
    assert m.bands[3].year == 2024 and m.bands[3].datum == "evrf2007"
    assert m.width == 2 and m.height == 1 and m.crs == "EPSG:2180"
    assert m.stack_path.endswith("stack_2024.tif")


def test_run_missing_input_exits_1(tmp_path, monkeypatch):
    render_root, dem_root = _seed_inputs(tmp_path)
    (render_root / "year_2024.tiff").unlink()  # remove RGB
    _patch_io(monkeypatch)
    cfg = StackConfig(
        location_name="Poznan", rgb_year=2024, nmt_year=2024, nmpt_year=2024,
        render_root=render_root, dem_root=dem_root,
        output_json=tmp_path / "stack_poznan" / "stack_2024.json",
    )
    code, _ = dem_stack.run(cfg)
    assert code == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_dem_stack.py -k "run_builds or missing_input" -q`
Expected: FAIL — `AttributeError: module ... has no attribute 'run'`.

- [ ] **Step 3: Implement `run` + IO seams**

APPEND to `src/satmap_dataset/pipeline/dem_stack.py`:

```python
from satmap_dataset.config import StackConfig
from satmap_dataset.models import StackBandDescriptor, StackManifest

_NORMALIZATION_HINT = {"rgb": "/255", "elevation": "z-score or min-max per dataset"}


def _align_to_grid(src: str, out: str, grid, resample: str, nodata: float) -> None:
    gdalwarp = _tool_path("gdalwarp")
    if not gdalwarp:
        raise RuntimeError("Aligning layers requires the GDAL CLI (gdalwarp). Install GDAL.")
    xmin, ymin, xmax, ymax, width, height, epsg = grid
    try:
        subprocess.run(
            [
                gdalwarp, "-t_srs", f"EPSG:{epsg}",
                "-te", str(xmin), str(ymin), str(xmax), str(ymax),
                "-ts", str(width), str(height), "-r", resample,
                "-dstnodata", str(nodata), "-overwrite", str(src), str(out),
            ],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"gdalwarp align failed: {(exc.stderr or '')[-500:]}") from exc


def _read_raster(path: str):
    import numpy as np
    import tifffile

    arr = np.asarray(tifffile.imread(str(path)))
    if arr.ndim == 3 and arr.shape[-1] in (3, 4) and arr.shape[0] not in (3, 4):
        arr = np.moveaxis(arr, -1, 0)  # (H,W,C) -> (C,H,W)
    return arr.astype("float32")


def _write_geotiff(stack, out_path: str, grid, *, epsg: int) -> None:
    import numpy as np
    import tifffile

    gdal_translate = _tool_path("gdal_translate")
    if not gdal_translate:
        raise RuntimeError("Writing the stack requires the GDAL CLI (gdal_translate). Install GDAL.")
    xmin, ymin, xmax, ymax, _w, _h, _e = grid
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plain = out_path.with_suffix(".plain.tif")
    tifffile.imwrite(str(plain), np.moveaxis(stack, 0, -1), photometric="minisblack", planarconfig="contig")
    try:
        subprocess.run(
            [
                gdal_translate, "-a_srs", f"EPSG:{epsg}",
                "-a_ullr", str(xmin), str(ymax), str(xmax), str(ymin),
                "-co", "COMPRESS=DEFLATE", str(plain), str(out_path),
            ],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"gdal_translate georeference failed: {(exc.stderr or '')[-500:]}") from exc
    finally:
        plain.unlink(missing_ok=True)


def _rgb_path(config: StackConfig) -> Path:
    return Path(config.render_root) / f"year_{config.rgb_year}.tiff"


def _dem_native(config: StackConfig, product: str, year: int) -> Path:
    return Path(config.dem_root) / "skorowidz" / f"{product}_{config.vertical_datum}" / "native" / f"year_{year}.tif"


def run(config: StackConfig) -> tuple[int, Path]:
    import numpy as np  # noqa: F401 (used by helpers)

    output_json = Path(config.output_json)
    rgb_path = _rgb_path(config)
    needed = [("rgb", rgb_path)]
    if config.nmt_year is not None and ("nmt" in config.bands or "ndsm" in config.bands):
        needed.append(("nmt", _dem_native(config, "nmt", config.nmt_year)))
    if config.nmpt_year is not None and ("nmpt" in config.bands or "ndsm" in config.bands):
        needed.append(("nmpt", _dem_native(config, "nmpt", config.nmpt_year)))
    missing = [str(p) for _role, p in needed if not p.exists()]
    if missing:
        logger.error("dem-stack: missing inputs: %s", missing)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            StackManifest(
                location_name=config.location_name, stack_path="", crs="", width=1, height=1,
                passed=False, run_parameters={"missing_inputs": missing},
            ).model_dump_json(indent=2),
            encoding="utf-8",
        )
        return 1, output_json

    grid = _raster_grid(str(rgb_path))
    xmin, ymin, xmax, ymax, width, height, epsg = grid
    stack_tif = output_json.with_suffix(".tif")

    tmp_aligned: dict[str, Path] = {}
    nmt_arr = nmpt_arr = None
    needs_nmt = config.nmt_year is not None and ("nmt" in config.bands or "ndsm" in config.bands)
    needs_nmpt = config.nmpt_year is not None and ("nmpt" in config.bands or "ndsm" in config.bands)
    if needs_nmt:
        out = stack_tif.with_name("_aligned_nmt.tif")
        _align_to_grid(str(_dem_native(config, "nmt", config.nmt_year)), str(out), grid, config.resample, config.nodata_in)
        tmp_aligned["nmt"] = out
        nmt_arr = _read_raster(str(out))
    if needs_nmpt:
        out = stack_tif.with_name("_aligned_nmpt.tif")
        _align_to_grid(str(_dem_native(config, "nmpt", config.nmpt_year)), str(out), grid, config.resample, config.nodata_in)
        tmp_aligned["nmpt"] = out
        nmpt_arr = _read_raster(str(out))

    rgb_arr = _read_raster(str(rgb_path))
    stack, names = _assemble_stack(
        rgb_arr, nmt_arr, nmpt_arr, bands=config.bands, nodata=config.nodata_in, fill=config.fill_value
    )
    _write_geotiff(stack, str(stack_tif), grid, epsg=epsg)
    for path in tmp_aligned.values():
        path.unlink(missing_ok=True)

    descriptors: list[StackBandDescriptor] = []
    for idx, (name, role, unit) in enumerate(names, start=1):
        source = year = datum = derived = None
        if role == "rgb":
            source, year = str(rgb_path), config.rgb_year
        elif name == "nmt":
            source, year, datum = str(_dem_native(config, "nmt", config.nmt_year)), config.nmt_year, config.vertical_datum
        elif name == "nmpt":
            source, year, datum = str(_dem_native(config, "nmpt", config.nmpt_year)), config.nmpt_year, config.vertical_datum
        elif name == "ndsm":
            derived, datum = "nmpt-nmt", config.vertical_datum
        descriptors.append(StackBandDescriptor(index=idx, name=name, role=role, unit=unit, source=source, year=year, datum=datum, derived=derived))

    manifest = StackManifest(
        provider="geoportal", location_name=config.location_name, stack_path=str(stack_tif),
        crs=f"EPSG:{epsg}", width=width, height=height, fill_value=config.fill_value,
        bands=descriptors, normalization_hint=dict(_NORMALIZATION_HINT), passed=True,
        run_parameters=config.model_dump(mode="json"),
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    logger.info("dem-stack: wrote %s bands=%s", stack_tif, [d.name for d in descriptors])
    return 0, output_json
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_dem_stack.py -q` (5 pass). Then `pytest -q` (green).

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/pipeline/dem_stack.py tests/test_dem_stack.py
git commit -m "feat(stack): dem_stack.run orchestration (align, assemble, georeference, manifest)"
```

---

## Task 5: CLI + roots + gitignore + just

**Files:**
- Modify: `src/satmap_dataset/cli.py`
- Modify: `Justfile`, `.gitignore`, `scripts/manage_location_roots.py`
- Test: `tests/test_dem_stack_cli.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dem_stack_cli.py`:

```python
import json
from pathlib import Path

from typer.testing import CliRunner

from satmap_dataset.cli import app, _build_stack_config_from_base_and_location

runner = CliRunner()


def test_dem_stack_json_command(tmp_path, monkeypatch):
    captured = {}

    def _fake_run(config):
        captured["config"] = config
        out = tmp_path / "stack.json"
        out.write_text("{}")
        return (0, out)

    monkeypatch.setattr("satmap_dataset.pipeline.dem_stack.run", _fake_run)
    params = tmp_path / "p.json"
    params.write_text(json.dumps({
        "location_name": "Poznan", "rgb_year": 2024, "nmt_year": 2024, "nmpt_year": 2024,
        "render_root": str(tmp_path / "r"), "dem_root": str(tmp_path / "d"),
        "output_json": str(tmp_path / "stack.json"),
    }))
    result = runner.invoke(app, ["dem-stack-json", str(params)])
    assert result.exit_code == 0
    assert captured["config"].rgb_year == 2024
    assert result.stdout.strip().splitlines()[-1].endswith("stack.json")


def test_dem_stack_location_builder(tmp_path):
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"vertical_datum": "evrf2007"}))
    loc = tmp_path / "location_poznan.json"
    loc.write_text(json.dumps({"location_name": "Poznan", "center_lat": 52.4, "center_lon": 16.95}))
    cfg = _build_stack_config_from_base_and_location(
        base_json=base, location_json=loc, rgb_year=2024, nmt_year=2024, nmpt_year=2024,
    )
    assert cfg.location_name == "Poznan"
    assert cfg.rgb_year == 2024 and cfg.nmt_year == 2024 and cfg.nmpt_year == 2024
    assert str(cfg.render_root).endswith("rendered_poznan")
    assert str(cfg.dem_root).endswith("dem_poznan")
    assert str(cfg.output_json).endswith("stack_2024.json")
    assert "stack_poznan" in str(cfg.output_json)


def test_dem_stack_json_bad_config_exit_2(tmp_path):
    params = tmp_path / "p.json"
    params.write_text(json.dumps({"location_name": "x", "rgb_year": 2024, "bands": ["rgb", "ndsm"], "nmt_year": 2024}))
    result = runner.invoke(app, ["dem-stack-json", str(params)])
    assert result.exit_code == 2
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_dem_stack_cli.py -q`
Expected: FAIL — `ImportError: cannot import name '_build_stack_config_from_base_and_location'`.

- [ ] **Step 3a: Imports**

In `src/satmap_dataset/cli.py`: add `StackConfig` to the `from satmap_dataset.config import (...)` block; add `dem_stack` to the `from satmap_dataset.pipeline import ...` line.

- [ ] **Step 3b: Builder**

After `_build_dem_availability_config_from_base_and_location` in `cli.py`, add:

```python
def _build_stack_config_from_base_and_location(
    *, base_json: Path, location_json: Path,
    rgb_year: int, nmt_year: int | None = None, nmpt_year: int | None = None,
    vertical_datum: str | None = None,
) -> StackConfig:
    base_payload = _load_params_json_dict(base_json)
    location_payload = _load_params_json_dict(location_json)
    merged: dict[str, object] = dict(base_payload)
    merged.update(location_payload)
    repo_root = base_json.resolve().parents[2] if len(base_json.resolve().parents) >= 3 else Path.cwd().resolve()
    merged = _apply_location_paths_policy(merged, repo_root)
    location_name = merged.get("location_name")
    if location_name is None:
        raise typer.BadParameter("location JSON must set location_name for dem-stack.")
    slug = _slugify_location_name(str(location_name))
    payload: dict[str, object] = {
        "location_name": location_name,
        "rgb_year": rgb_year,
        "nmt_year": nmt_year,
        "nmpt_year": nmpt_year,
        "render_root": merged.get("render_root", str(repo_root / f"rendered_{slug}")),
        "dem_root": merged.get("dem_root", str(repo_root / f"dem_{slug}")),
        "output_json": str(repo_root / f"stack_{slug}" / f"stack_{rgb_year}.json"),
    }
    if vertical_datum is not None:
        payload["vertical_datum"] = vertical_datum
    elif "vertical_datum" in merged:
        payload["vertical_datum"] = merged["vertical_datum"]
    if "bands" in merged:
        payload["bands"] = merged["bands"]
    return StackConfig.model_validate(payload)
```

- [ ] **Step 3c: Commands (end of cli.py)**

```python
@app.command("dem-stack-json")
def dem_stack_json_command(
    params_json: Path = typer.Argument(..., help="JSON with StackConfig fields."),
) -> None:
    try:
        payload = _load_params_json_dict(params_json)
        config = StackConfig.model_validate(payload)
    except typer.BadParameter as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=2) from error
    except ValidationError as error:
        _print_validation_error(error)
        raise typer.Exit(code=2) from error

    exit_code, artifact_path = dem_stack.run(config)
    _finish(exit_code, artifact_path)


@app.command("dem-stack-location-json")
def dem_stack_location_json_command(
    location_json: Path = typer.Argument(..., help="Location JSON (location_name)."),
    rgb_year: int = typer.Option(..., "--rgb-year", help="Orthophoto year (reference grid)."),
    nmt_year: int = typer.Option(None, "--nmt-year", help="NMT (DTM) acquisition year."),
    nmpt_year: int = typer.Option(None, "--nmpt-year", help="NMPT (DSM) acquisition year."),
    vertical_datum: str = typer.Option(None, "--vertical-datum", help="evrf2007 or kron86."),
    base_json: Path = typer.Option(Path("configs/run/base.json"), "--base-json"),
) -> None:
    try:
        config = _build_stack_config_from_base_and_location(
            base_json=base_json, location_json=location_json,
            rgb_year=rgb_year, nmt_year=nmt_year, nmpt_year=nmpt_year, vertical_datum=vertical_datum,
        )
    except typer.BadParameter as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=2) from error
    except ValidationError as error:
        _print_validation_error(error)
        raise typer.Exit(code=2) from error

    exit_code, artifact_path = dem_stack.run(config)
    _finish(exit_code, artifact_path)
```

Verify helper names (`_load_params_json_dict`, `_apply_location_paths_policy`, `_slugify_location_name`, `_finish`, `_print_validation_error`, `console`, `ValidationError`, `typer`) exist; adapt if different, STOP+report if absent.

- [ ] **Step 3d: gitignore + roots + just**

In `.gitignore`, after the `dem_*/` line, add:
```
stack_*/
```

In `scripts/manage_location_roots.py`: add `"stack"` to the `KINDS` tuple and `"stack": "stack_root"` to the `_path_for_kind` mapping dict.

Append to `Justfile` (2-space recipe body):
```just
# Build aligned multi-band ML stack (RGB+NMT+NMPT+nDSM+mask) for a location
dem-stack location_json rgb_year nmt_year nmpt_year:
  python -m satmap_dataset.cli dem-stack-location-json {{location_json}} --rgb-year {{rgb_year}} --nmt-year {{nmt_year}} --nmpt-year {{nmpt_year}}
```

- [ ] **Step 4: Run tests + full suite + help**

Run: `pytest tests/test_dem_stack_cli.py -q` (3 pass). Then `pytest -q` (green). Then `python -m satmap_dataset.cli --help` and confirm `dem-stack-json` and `dem-stack-location-json` appear. Confirm `python -c "import scripts.manage_location_roots as m; assert 'stack' in m.KINDS"` (the test file already shims sys.path for `scripts`; if running ad-hoc, run from repo root).

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/cli.py Justfile .gitignore scripts/manage_location_roots.py tests/test_dem_stack_cli.py
git commit -m "feat(stack): CLI dem-stack commands + stack root wiring + just task"
```

---

## Task 6: Live smoke (manual, outside sandbox)

**Files:** none (manual — needs the rendered ortho + downloaded DEM present locally).

- [ ] **Step 1: Ensure inputs exist for Poznań**

`rendered_poznan/year_2024.tiff` (run `render-location-json` for Poznań, year 2024) and
`dem_poznan/skorowidz/{nmt,nmpt}_evrf2007/native/year_2024.tif` (run the skorowidz `dem`
download for NMT and NMPT, EVRF2007, 2024).

- [ ] **Step 2: Build the stack**

```bash
python -m satmap_dataset.cli dem-stack-location-json configs/run/locations/poznan.json \
  --rgb-year 2024 --nmt-year 2024 --nmpt-year 2024 --vertical-datum evrf2007
```

Expected: exit `0`; last stdout line is `stack_poznan/stack_2024.json`;
`stack_poznan/stack_2024.tif` opens as a **7-band float32** EPSG:2180 raster on the ortho
grid. Verify with `gdalinfo stack_poznan/stack_2024.tif` (Band count 7, Type=Float32) and
in Python that band 7 (`valid_mask`) is 1 where elevation exists and the band order matches
the JSON manifest.

- [ ] **Step 3: Record results**

Confirm bands align pixel-for-pixel (RGB and elevation overlay), and note any RGB-nodata
handling needed if the render leaves black borders.

---

## Self-Review

**Spec coverage:**
- Single float32 multi-band GeoTIFF on the RGB grid → Task 4 (`_write_geotiff`, `_raster_grid`). ✓
- Bands `[R,G,B,NMT,NMPT,nDSM,valid_mask]`, configurable value subset + mask always last → Task 3 (`_assemble_stack`) + Task 2 (`bands` validator). ✓
- Raw values, invalid filled 0.0, shared valid_mask (intersection) → Task 3. ✓
- Auto-discovery by location (rendered_<slug>, dem_<slug>) + missing-input error → Task 4 (`_rgb_path`/`_dem_native`/`needed` check) + Task 5 builder. ✓
- nDSM = NMPT−NMT after alignment → Task 3 (`ndsm_a`) + Task 4 (align then assemble). ✓
- StackManifest provenance (per-band source/year/datum/derived) → Task 4 descriptor loop + Task 1 model. ✓
- CLI `dem-stack-json`/`dem-stack-location-json` + builder; last-line artifact path; exit 0/1/2 → Task 5. ✓
- `stack_*/` gitignore + `stack` root kind + `just dem-stack` → Task 5 Step 3d. ✓
- GDAL toolchain (gdalinfo/gdalwarp/gdal_translate) with clear errors → Task 4 (`_gdalinfo_json`/`_align_to_grid`/`_write_geotiff`). ✓
- Sidecar = the manifest JSON → Task 4. ✓
- Live smoke → Task 6. ✓

**Placeholder scan:** No TBD/TODO; every code step complete. ✓

**Type consistency:** `_raster_grid(path) -> (xmin,ymin,xmax,ymax,W,H,epsg)` used identically in `_align_to_grid`/`_write_geotiff`/`run`; `_assemble_stack(rgb,nmt,nmpt,*,bands,nodata,fill) -> (ndarray,names)`; `_read_raster`/`_write_geotiff`/`_align_to_grid` seams match their test monkeypatches; `StackConfig` fields (`rgb_year`/`nmt_year`/`nmpt_year`/`bands`/`render_root`/`dem_root`/`output_json`/`vertical_datum`/`nodata_in`/`fill_value`) and `StackBandDescriptor`/`StackManifest` fields consistent across Tasks 1–5. ✓

**Reuse safety:** Purely additive; no changes to render/dem/availability code. Reuses the established GDAL-CLI posture (capture stderr, clear error if missing). ✓
