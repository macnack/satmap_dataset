# Raw-tile Export + Ingest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Permanent location:** On execution, copy this plan to `docs/superpowers/plans/2026-06-22-raw-tile-export-ingest.md` (the harness plan file is temporary).

## Context

Today, feeding satmap_dataset downloads into sat_roma's `raw_tile_pipeline` is a manual step: reorganise `downloads_<slug>/<year>/*.tif` into `sat_data_raw/<provider>/<area>/<year>/`, then run sat_roma's scripts. This feature adds an **opt-in `raw-export` stage/command** that does that bridging automatically: it lays native download tiles into the canonical `sat_data_raw` layout, ingests them into co-located season-cell stacks (with provider-aware coverage gating), writes the per-area `manifest.yaml`, and can build the cross-location split `test_manifest.yaml` that sat_roma consumes. Pair generation, viz, and training stay in sat_roma — the handoff contract is the manifest.

Design spec: `docs/superpowers/specs/2026-06-22-raw-tile-export-ingest-design.md`.

**Goal:** Add a `raw-export` pipeline stage + CLI command family that turns native download tiles into the `sat_data_raw/<provider>/<area>/…` layout (raw tiles + ingested cells + `manifest.yaml`) plus a `raw-test-manifest` command for the split manifest.

**Architecture:** Port sat_roma's 100%-portable `romatch/datasets/raw_tiles.py` ingestion core verbatim into a new `src/satmap_dataset/raw_tiles/` package (drift-guarded by a shared test vector). Add a `pipeline/raw_export.py` stage that follows the existing `run(config) -> (exit_code, Path)` contract (export native tiles → `ingest_area` → write `raw_export_manifest.json`), plus `RawExportConfig`/`RawExportManifest` models and the standard CLI 3-flavour surface. Not wired into `run-all`.

**Tech Stack:** Python ≥3.10, Pydantic v2, Typer, pyvips, `gdalinfo` CLI (GDAL, already installed), **PyYAML (new dependency)**, pytest.

## Global Constraints

- **Provider domain:** `{geoportal, lantmateriet, nls}`. `sentinel2` is rejected by `RawExportConfig` (not a raw-orthophoto-tile provider). Validation reuses `config.ALLOWED_PROVIDERS`.
- **No resampling.** Export symlinks (or copies) native tiles 1:1; ingest symlinks native cells or losslessly pixel-window crops. Never reproject/resample.
- **Stage contract:** `run(config) -> tuple[int, Path]`, write exactly one JSON manifest, return `(0, path)` on pass / `(1, path)` on fail. CLI prints the absolute artifact path as the last stdout line via `_finish()`.
- **Provider-aware coverage:** geoportal `0.5` (godło nodata borders), others `0.95` default — sourced from the ported registry, overridable by `--min-coverage`.
- **EPSG is a cross-check only:** satmap passes the configured provider; a mismatch between configured provider and the tile's detected-EPSG provider is a **warning**, never a routing decision.
- **cell_key:** `e<round(ulx)>_n<round(uly - h_m)>` (SW corner, provider-CRS metres). Cells are **rectangular** (`Cell(ulx, uly, w_m, h_m)`).
- **Port fidelity:** `raw_tiles/core.py` is a verbatim copy of sat_roma `romatch/datasets/raw_tiles.py`; the module header must point at that file as the source of truth. Drift is caught by `tests/test_raw_tiles_core.py` (shared vectors).
- **Source of truth for the port:** `~/Github/sat_roma` branch `worktree-spec-kiruna-raw-tiles` (checked out as the linked worktree at `/home/maciej/Github/sat_roma/.claude/worktrees/spec-kiruna-raw-tiles/`) — files `romatch/datasets/raw_tiles.py`, `romatch/datasets/raw_tile_providers.yaml`, `raw_tile_pipeline/build_test_manifest.py`, `tests/test_raw_tiles.py`. Read via the worktree path or `git -C ~/Github/sat_roma show worktree-spec-kiruna-raw-tiles:<file>`. Logical source-of-truth name to cite in headers: `sat_roma romatch/datasets/raw_tiles.py`.

---

## File Structure

**New package — `src/satmap_dataset/raw_tiles/`** (ported ingestion core, self-contained):
- `__init__.py` — re-export the public surface used by the stage.
- `core.py` — verbatim port of `raw_tiles.py` (GeoTransform, gdalinfo helpers, tfw/prj writers, TileInfo, Cell, registry loader, `ingest_area`, etc.).
- `raw_tile_providers.yaml` — verbatim port of the EPSG→provider registry (loaded via `Path(__file__).with_name("raw_tile_providers.yaml")`).
- `split_manifest.py` — ported `build_test_manifest(root, out, min_years=2) -> dict` (refactored from sat_roma's `build_test_manifest.py` script `main()`).

**New stage:**
- `src/satmap_dataset/pipeline/raw_export.py` — `run(config: RawExportConfig) -> tuple[int, Path]` + private `_can_reuse_raw_export`, `_export_native_tiles`.

**Modified:**
- `src/satmap_dataset/config.py` — add `RawExportConfig`.
- `src/satmap_dataset/models.py` — add `RawExportManifest`.
- `src/satmap_dataset/cli.py` — add `raw-export`, `raw-export-json`, `raw-export-location-json`, `raw-export-all-location-json`, `raw-test-manifest` commands + `_build_raw_export_config_from_base_and_location`.
- `pyproject.toml` — add `"PyYAML"` to `dependencies`.
- `Justfile` — add `raw-export-location-json`, `raw-export-all-location-json`, `raw-test-manifest` recipes.

**New tests:**
- `tests/test_raw_tiles_core.py` — ported pure unit vectors (drift guard).
- `tests/test_raw_tiles_split_manifest.py` — split-manifest builder on synthetic cells.
- `tests/test_raw_export_config.py` — `RawExportConfig` validation.
- `tests/test_raw_export_stage.py` — end-to-end on a tiny 2-year/2-tile fixture.
- `tests/test_raw_export_cli.py` — CLI smoke + base+location builder.

---

## Task 1: Port the ingestion core (`raw_tiles/` package) + add PyYAML

**Files:**
- Create: `src/satmap_dataset/raw_tiles/__init__.py`
- Create: `src/satmap_dataset/raw_tiles/core.py`
- Create: `src/satmap_dataset/raw_tiles/raw_tile_providers.yaml`
- Modify: `pyproject.toml` (add `"PyYAML"` to `dependencies`)
- Test: `tests/test_raw_tiles_core.py`

**Interfaces:**
- Produces (consumed by later tasks):
  - `core.ingest_area(src_area: Path, out_root: Path, registry: dict, *, cell_size_m: float | None = None, min_coverage: float | None = None) -> dict` — returns the per-area manifest dict (`{provider, area, epsg, cell_size_m, locations}`).
  - `core.load_provider_registry(path: Path | None = None) -> dict`
  - `core.provider_for_epsg(epsg: int, registry: dict) -> str`
  - `core.min_coverage_for_epsg(epsg: int, registry: dict, default: float = 0.95) -> float`
  - `core.GeoTransform`, `core.Cell`, `core.TileInfo`, `core.read_tile_info`, `core.detect_year`, `core.cell_key`, `core.derive_cell_grid`, `core.geotransform_to_tfw_lines`, `core.write_tfw`, `core.write_prj_wkt`, `core._epsg_from_wkt`, `core.valid_pixel_fraction`, `core.resolve_season_tile`.

- [ ] **Step 1: Add PyYAML dependency**

In `pyproject.toml`, add `"PyYAML"` to the `dependencies` list (after `"imagecodecs"`):

```toml
dependencies = [
  "httpx",
  "aiofiles",
  "numpy",
  "pydantic>=2",
  "pyproj",
  "typer",
  "rich",
  "Pillow",
  "pyvips",
  "tifffile",
  "imagecodecs",
  "PyYAML",
]
```

Then run `python -m pip install -e ".[dev]"` to install it.

- [ ] **Step 2: Port the registry YAML verbatim**

Copy `raw_tile_providers.yaml` from the source-of-truth path into `src/satmap_dataset/raw_tiles/raw_tile_providers.yaml`. Exact content:

```yaml
# EPSG code -> provider label (output top-level namespace). Extend for new areas:
# a genuinely new national CRS just needs one line here.
#
# A value may be a plain provider name, or a mapping
#   <epsg>: {provider: <name>, min_coverage: <0..1>}
# to override the default coverage gate for that provider (e.g. Geoportal godlo
# sheets carry nodata borders, so a 0.95 gate would drop most of them).
2180: {provider: geoportal, min_coverage: 0.5}   # Poland  (ETRS89 / Poland CS92)
3006: lantmateriet                                # Sweden  (SWEREF99 TM)
3067: nls                                         # Finland (ETRS89 / TM35FIN)
```

- [ ] **Step 3: Port `core.py` verbatim**

Copy the full body of `romatch/datasets/raw_tiles.py` (source-of-truth path) into `src/satmap_dataset/raw_tiles/core.py` **unchanged**, except replace the module docstring's first line to point at the source of truth:

```python
"""Ported, self-contained copy of sat_roma `romatch/datasets/raw_tiles.py`.

SOURCE OF TRUTH: sat_roma romatch/datasets/raw_tiles.py — keep this file in sync;
drift is caught by tests/test_raw_tiles_core.py (shared vectors).

Pure georeferencing + indexing logic with no training dependencies. Reads each
tile's geotransform and CRS via the `gdalinfo` CLI and writes the `.tfw`/`.prj`
sidecars the raw-tile datasets expect.
"""
```

Keep every function/class/constant identical: `_EPSG_RE`, `GeoTransform`, `gdalinfo_json`, `read_geotransform`, `read_crs_wkt`, `_epsg_from_wkt`, `geotransform_to_tfw_lines`, `write_tfw`, `write_prj_wkt`, `_REGISTRY_PATH`, `_YEAR_DIR_RE`, `_YEAR_TOKEN_RE`, `load_provider_registry`, `_registry_entry`, `provider_for_epsg`, `min_coverage_for_epsg`, `detect_year`, `TileInfo`, `read_tile_info`, `Cell`, `cell_key`, `derive_cell_grid`, `tile_covers_cell`, `world_window_to_pixel`, `valid_pixel_fraction`, `resolve_season_tile`, `_COVERAGE_THUMB_PX`, `ingest_area`. Do **not** rename or refactor — fidelity is the point.

- [ ] **Step 4: Write `__init__.py`**

```python
"""Self-contained raw-tile ingestion core (ported from sat_roma)."""
from satmap_dataset.raw_tiles.core import (
    Cell,
    GeoTransform,
    TileInfo,
    cell_key,
    derive_cell_grid,
    detect_year,
    geotransform_to_tfw_lines,
    ingest_area,
    load_provider_registry,
    min_coverage_for_epsg,
    provider_for_epsg,
    read_tile_info,
    resolve_season_tile,
    valid_pixel_fraction,
    write_prj_wkt,
    write_tfw,
)

__all__ = [
    "Cell",
    "GeoTransform",
    "TileInfo",
    "cell_key",
    "derive_cell_grid",
    "detect_year",
    "geotransform_to_tfw_lines",
    "ingest_area",
    "load_provider_registry",
    "min_coverage_for_epsg",
    "provider_for_epsg",
    "read_tile_info",
    "resolve_season_tile",
    "valid_pixel_fraction",
    "write_prj_wkt",
    "write_tfw",
]
```

- [ ] **Step 5: Write the shared unit-vector test (drift guard)**

Create `tests/test_raw_tiles_core.py`. These mirror sat_roma's `tests/test_raw_tiles.py` **pure** vectors. The one sat_roma test that imports `romatch.datasets.tfw._read_tfw` is replaced with a self-contained read-back of the 6 `.tfw` lines (no romatch dependency):

```python
import math

import numpy as np
import pyvips
import pytest

from satmap_dataset.raw_tiles.core import (
    Cell,
    GeoTransform,
    TileInfo,
    _epsg_from_wkt,
    cell_key,
    derive_cell_grid,
    detect_year,
    geotransform_to_tfw_lines,
    load_provider_registry,
    min_coverage_for_epsg,
    provider_for_epsg,
    resolve_season_tile,
    tile_covers_cell,
    valid_pixel_fraction,
    world_window_to_pixel,
    write_prj_wkt,
    write_tfw,
    _YEAR_TOKEN_RE,
)


def test_geotransform_to_tfw_lines_corner_to_center():
    gt = GeoTransform(ulx=383394.0, xres=0.5, xskew=0.0, uly=6674897.5, yskew=0.0, yres=-0.5)
    a, d, b, e, c, f = geotransform_to_tfw_lines(gt)
    assert (a, d, b, e) == (0.5, 0.0, 0.0, -0.5)
    assert math.isclose(c, 383394.25, abs_tol=1e-9)
    assert math.isclose(f, 6674897.25, abs_tol=1e-9)


def test_write_tfw_reads_back_six_lines(tmp_path):
    gt = GeoTransform(717500.0, 0.16, 0.0, 7537500.0, 0.0, -0.16)
    p = tmp_path / "year_2025.tfw"
    write_tfw(gt, p)
    vals = [float(x) for x in p.read_text().splitlines()]
    assert len(vals) == 6
    assert vals[0] == pytest.approx(0.16) and vals[3] == pytest.approx(-0.16)
    assert vals[4] == pytest.approx(717500.08) and vals[5] == pytest.approx(7537499.92)


def test_epsg_from_wkt_top_level_authority_only():
    wkt = ('PROJCRS["x", BASEGEOGCRS["b", DATUM["d", ID["EPSG",6326]]], '
           'CONVERSION["c", METHOD["m", ID["EPSG",9807]]], CS["xy", ID["EPSG",9001]], '
           'ID["EPSG",2180]]')
    assert _epsg_from_wkt(wkt) == 2180
    generic = 'PROJCRS["Transverse Mercator; WGS84", CS["xy", AXIS["e", ID["EPSG",9001]]]]'
    assert _epsg_from_wkt(generic) is None


def test_write_prj_wkt_verbatim(tmp_path):
    p = tmp_path / "year_2017.prj"
    write_prj_wkt('PROJCS["x"]', p)
    assert p.read_text() == 'PROJCS["x"]'


def test_registry_maps_three_providers():
    reg = load_provider_registry()
    assert provider_for_epsg(2180, reg) == "geoportal"
    assert provider_for_epsg(3006, reg) == "lantmateriet"
    assert provider_for_epsg(3067, reg) == "nls"


def test_provider_for_unknown_epsg_raises_helpful():
    with pytest.raises(KeyError, match="9999"):
        provider_for_epsg(9999, {2180: {"provider": "geoportal", "min_coverage": None}})


def test_min_coverage_per_provider_override():
    reg = load_provider_registry()
    assert min_coverage_for_epsg(2180, reg) == 0.5
    assert min_coverage_for_epsg(3006, reg) == 0.95
    assert min_coverage_for_epsg(3067, reg, default=0.9) == 0.9


def test_detect_year_prefers_parent_dir(tmp_path):
    d = tmp_path / "2014"
    d.mkdir()
    f = d / "nls_2020_0_0.tif"
    f.write_bytes(b"")
    assert detect_year(f) == 2014


def test_detect_year_filename_token_and_coordinate_guard(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    f = d / "nls_2014_0_0.tif"
    f.write_bytes(b"")
    assert detect_year(f) == 2014
    assert _YEAR_TOKEN_RE.search("e717500_n7535000") is None
    assert _YEAR_TOKEN_RE.search("o75350_7175_25_fj08") is None


def _tile(ulx, uly, gsd, size_m, year=2008, ny=None):
    nx = round(size_m / gsd)
    return TileInfo(None, nx, ny if ny is not None else nx,
                    GeoTransform(ulx, gsd, 0.0, uly, 0.0, -gsd), 3006, "wkt", year)


def test_cell_key_sw_corner():
    assert cell_key(Cell(717500, 7537500, 2500, 2500)) == "e717500_n7535000"


def test_derive_cell_grid_uses_smallest_footprint():
    fine = _tile(717500, 7537500, 0.25, 2500)
    coarse = _tile(715000, 7540000, 0.50, 5000)
    cells = derive_cell_grid([fine, coarse])
    assert all(c.w_m == 2500 and c.h_m == 2500 for c in cells)
    assert Cell(717500, 7537500, 2500, 2500) in cells


def test_tile_covers_and_window():
    coarse = _tile(715000, 7540000, 0.50, 5000)
    cell = Cell(717500, 7537500, 2500, 2500)
    assert tile_covers_cell(coarse, cell)
    assert world_window_to_pixel(coarse.gt, 717500, 7537500, 2500, 2500) == (5000, 5000, 5000, 5000)


def test_derive_cell_grid_nonsquare_tile_covers_its_own_cell():
    gt = GeoTransform(383394.0, 0.5, 0.0, 6674897.5, 0.0, -0.5)
    t = TileInfo(None, 3200, 2987, gt, 3067, "wkt", 2014)
    cells = derive_cell_grid([t])
    assert len(cells) == 1
    c = cells[0]
    assert c.w_m == pytest.approx(1600.0) and c.h_m == pytest.approx(1493.5)
    assert tile_covers_cell(t, c)
    assert cell_key(c) == "e383394_n6673404"


def _img(arr):
    h, w, c = arr.shape
    return pyvips.Image.new_from_memory(arr.tobytes(), w, h, c, "uchar")


def test_valid_pixel_fraction_half_nodata():
    arr = np.zeros((10, 20, 3), np.uint8)
    arr[:, :10, :] = 200
    assert valid_pixel_fraction(_img(arr)) == pytest.approx(0.5, abs=1e-6)


def test_resolve_prefers_finest_gsd():
    cell = Cell(717500, 7537500, 2500, 2500)
    fine = _tile(717500, 7537500, 0.16, 2500, year=2019)
    coarse = _tile(715000, 7540000, 0.50, 5000, year=2019)
    assert resolve_season_tile(cell, [coarse, fine]) is fine
    assert resolve_season_tile(cell, [coarse]) is coarse
    assert resolve_season_tile(Cell(720000, 7532500, 2500, 2500), [fine]) is None
```

- [ ] **Step 6: Run the test suite, verify pass**

Run: `pytest tests/test_raw_tiles_core.py -v`
Expected: PASS (all vectors). If `valid_pixel_fraction`/registry tests need libvips/pyyaml, confirm both are installed.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/satmap_dataset/raw_tiles/ tests/test_raw_tiles_core.py
git commit -m "feat(raw-tiles): port self-contained ingestion core + PyYAML dep"
```

---

## Task 2: Port the split-manifest builder (`raw_tiles/split_manifest.py`)

**Files:**
- Create: `src/satmap_dataset/raw_tiles/split_manifest.py`
- Test: `tests/test_raw_tiles_split_manifest.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (only pyvips + yaml + stdlib).
- Produces: `build_test_manifest(root: Path, out: Path, min_years: int = 2) -> dict` — scans `<root>/<provider>/<area>/<cellkey>/year_YYYY.tif`, materialises homogeneous symlink dirs under `<root>/_manifest_locs/`, writes the split manifest YAML to `out`, and returns the manifest dict (`{"roots": {"locs": ...}, "<loc_name>": {"root": "locs", "test": {"query": int, "ref": [int, ...]}}}`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_raw_tiles_split_manifest.py`. It builds two synthetic cells with equal-dimension `year_*.tif` files via pyvips, then asserts the split manifest picks the richest cell and emits `query`/`ref`:

```python
import numpy as np
import pyvips
import yaml

from satmap_dataset.raw_tiles.split_manifest import build_test_manifest


def _write_tif(path, w=8, h=8):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((h, w, 3), 200, np.uint8)
    img = pyvips.Image.new_from_memory(arr.tobytes(), w, h, 3, "uchar")
    img.tiffsave(str(path))


def test_build_test_manifest_picks_richest_cell(tmp_path):
    root = tmp_path / "sat_data_raw"
    area = root / "geoportal" / "poznan"
    # rich cell: 3 equal-dim years; poor cell: 1 year
    for y in (2015, 2018, 2021):
        _write_tif(area / "e500_n600" / f"year_{y}.tif")
    _write_tif(area / "e700_n800" / "year_2019.tif")

    out = root / "test_manifest.yaml"
    manifest = build_test_manifest(root, out, min_years=2)

    loc_name = "geoportal_poznan_e500_n600"
    assert loc_name in manifest
    assert manifest[loc_name]["root"] == "locs"
    assert manifest[loc_name]["test"]["query"] == 2021
    assert manifest[loc_name]["test"]["ref"] == [2018, 2015]
    # poor cell excluded (below min_years)
    assert "geoportal_poznan_e700_n800" not in manifest
    # YAML written and round-trips
    on_disk = yaml.safe_load(out.read_text())
    assert on_disk[loc_name]["test"]["query"] == 2021
    # symlink location dir materialised
    link = root / "_manifest_locs" / loc_name / "year_2021.tif"
    assert link.is_symlink()
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_raw_tiles_split_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError: satmap_dataset.raw_tiles.split_manifest`.

- [ ] **Step 3: Implement `split_manifest.py`**

Port sat_roma's `raw_tile_pipeline/build_test_manifest.py`, refactoring `main()` into a reusable `build_test_manifest()` function (keep `_dims`/`_best_cell` identical):

```python
"""Build a held-out TEST split manifest from ingested raw-tile cells.

Ported from sat_roma `raw_tile_pipeline/build_test_manifest.py`. Scans
`<root>/<provider>/<area>/<cellkey>/year_YYYY.tif`, picks each area's richest
equal-dimension (single-GSD) season group, materialises a homogeneous symlink
location dir, and emits a per-location `roots:` split manifest (query = newest
year, ref = older years).
"""
from __future__ import annotations

import collections
import re
from pathlib import Path

import pyvips
import yaml

_YEAR = re.compile(r"year_(\d{4})")


def _dims(tif: Path) -> tuple:
    im = pyvips.Image.new_from_file(str(tif))
    return (im.width, im.height)


def _best_cell(area: Path):
    """(loc_dir_name, cellkey, sorted_years, dims) for the area's richest
    equal-dimension season group, or None."""
    best = None
    for cell in sorted(area.glob("e*_n*")):
        if not cell.is_dir():
            continue
        by_dims = collections.defaultdict(list)
        for f in sorted(cell.glob("year_*.tif")):
            by_dims[_dims(f)].append(int(_YEAR.search(f.name).group(1)))
        for dims, years in by_dims.items():
            if best is None or len(years) > len(best[2]):
                best = (cell.name, cell.name, sorted(years), dims)
    return best


def build_test_manifest(root: Path, out: Path, min_years: int = 2) -> dict:
    root = Path(root)
    out = Path(out)
    loc_root = (root / "_manifest_locs").resolve()
    loc_root.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"roots": {"locs": str(loc_root)}}

    for area in sorted(root.glob("*/*")):
        if not area.is_dir() or area.name == "_manifest_locs":
            continue
        pick = _best_cell(area)
        if pick is None or len(pick[2]) < min_years:
            continue
        _, cellkey, years, _dims_ = pick
        provider, area_name = area.parent.name, area.name
        loc_name = f"{provider}_{area_name}_{cellkey}"
        d = loc_root / loc_name
        d.mkdir(exist_ok=True)
        for y in years:
            link = d / f"year_{y}.tif"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to((area / cellkey / f"year_{y}.tif").resolve())
        manifest[loc_name] = {
            "root": "locs",
            "test": {"query": years[-1], "ref": years[:-1][::-1]},
        }

    out.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return manifest
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_raw_tiles_split_manifest.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/raw_tiles/split_manifest.py tests/test_raw_tiles_split_manifest.py
git commit -m "feat(raw-tiles): port split-manifest builder"
```

---

## Task 3: `RawExportConfig` (config.py)

**Files:**
- Modify: `src/satmap_dataset/config.py` (append a new config class; reuse existing `ALLOWED_PROVIDERS`, `PROVIDER_SENTINEL2`)
- Test: `tests/test_raw_export_config.py`

**Interfaces:**
- Consumes: `config.ALLOWED_PROVIDERS`, `config.PROVIDER_SENTINEL2` (already defined at `config.py:9-18`).
- Produces:
  ```python
  class RawExportConfig(BaseModel):
      provider: str
      area: str
      download_root: Path
      download_manifest: Path | None = None
      raw_root: Path = <env SATMAP_RAW_ROOT or ~/Github/sat_data_raw>
      min_coverage: float | None = None       # 0 < mc <= 1 when set
      link_mode: str = "symlink"              # "symlink" | "copy"
      cell_size_m: float | None = None        # > 0 when set
      artifacts_dir: Path = Path("artifacts")
      output_json: Path = Path("artifacts/raw_export_manifest.json")
  ```

- [ ] **Step 1: Write the failing test**

Create `tests/test_raw_export_config.py`:

```python
from pathlib import Path

import pytest
from pydantic import ValidationError

from satmap_dataset.config import RawExportConfig


def _base(**over):
    payload = {"provider": "geoportal", "area": "poznan", "download_root": "downloads_poznan"}
    payload.update(over)
    return payload


def test_defaults_resolve():
    cfg = RawExportConfig(**_base())
    assert cfg.provider == "geoportal"
    assert cfg.area == "poznan"
    assert cfg.link_mode == "symlink"
    assert cfg.min_coverage is None
    assert cfg.cell_size_m is None
    assert str(cfg.raw_root)  # non-empty default


def test_sentinel2_rejected():
    with pytest.raises(ValidationError):
        RawExportConfig(**_base(provider="sentinel2"))


def test_unknown_provider_rejected():
    with pytest.raises(ValidationError):
        RawExportConfig(**_base(provider="bogus"))


def test_link_mode_validated():
    assert RawExportConfig(**_base(link_mode="copy")).link_mode == "copy"
    with pytest.raises(ValidationError):
        RawExportConfig(**_base(link_mode="hardlink"))


def test_min_coverage_bounds():
    assert RawExportConfig(**_base(min_coverage=0.5)).min_coverage == 0.5
    with pytest.raises(ValidationError):
        RawExportConfig(**_base(min_coverage=0.0))
    with pytest.raises(ValidationError):
        RawExportConfig(**_base(min_coverage=1.5))


def test_cell_size_positive():
    assert RawExportConfig(**_base(cell_size_m=2500.0)).cell_size_m == 2500.0
    with pytest.raises(ValidationError):
        RawExportConfig(**_base(cell_size_m=0.0))


def test_raw_root_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SATMAP_RAW_ROOT", str(tmp_path / "custom_raw"))
    cfg = RawExportConfig(**_base())
    assert Path(cfg.raw_root) == tmp_path / "custom_raw"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_raw_export_config.py -v`
Expected: FAIL with `ImportError: cannot import name 'RawExportConfig'`.

- [ ] **Step 3: Implement `RawExportConfig`**

Append to `config.py` (after the existing config classes). Note the imports `os`, `Path`, `Field`, `field_validator`, `BaseModel` already exist in the module:

```python
def _default_raw_root() -> Path:
    env = os.environ.get("SATMAP_RAW_ROOT")
    if env:
        return Path(env).expanduser()
    return Path("~/Github/sat_data_raw").expanduser()


class RawExportConfig(BaseModel):
    provider: str
    area: str
    download_root: Path
    download_manifest: Path | None = None
    raw_root: Path = Field(default_factory=_default_raw_root)
    min_coverage: float | None = None
    link_mode: str = "symlink"
    cell_size_m: float | None = None
    artifacts_dir: Path = Path("artifacts")
    output_json: Path = Path("artifacts/raw_export_manifest.json")

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        if value == PROVIDER_SENTINEL2:
            raise ValueError("provider 'sentinel2' is not a raw-orthophoto-tile provider")
        if value not in ALLOWED_PROVIDERS:
            raise ValueError(f"provider must be one of {sorted(ALLOWED_PROVIDERS)}")
        return value

    @field_validator("link_mode")
    @classmethod
    def validate_link_mode(cls, value: str) -> str:
        if value not in {"symlink", "copy"}:
            raise ValueError("link_mode must be 'symlink' or 'copy'")
        return value

    @field_validator("min_coverage")
    @classmethod
    def validate_min_coverage(cls, value: float | None) -> float | None:
        if value is not None and not (0.0 < value <= 1.0):
            raise ValueError("min_coverage must be in (0, 1]")
        return value

    @field_validator("cell_size_m")
    @classmethod
    def validate_cell_size(cls, value: float | None) -> float | None:
        if value is not None and value <= 0.0:
            raise ValueError("cell_size_m must be > 0")
        return value

    @field_validator("area")
    @classmethod
    def validate_area(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("area must be non-empty")
        return value
```

If `import os` is not already present at the top of `config.py`, add it.

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_raw_export_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/config.py tests/test_raw_export_config.py
git commit -m "feat(config): add RawExportConfig"
```

---

## Task 4: `RawExportManifest` (models.py)

**Files:**
- Modify: `src/satmap_dataset/models.py` (append; reuse the module's existing `_utc_now`, `BaseModel`, `Field`, `Literal`, `datetime`, `Any`)
- Test: fold a minimal model test into `tests/test_raw_export_config.py` (or a 1-test `tests/test_raw_export_manifest.py`)

**Interfaces:**
- Produces:
  ```python
  class RawExportManifest(BaseModel):
      kind: Literal["raw_export_manifest"]
      stage: Literal["raw_export"]
      generated_at: datetime
      provider: str
      area: str
      raw_root: str
      epsg: int | None
      epsg_provider_mismatch: bool
      link_mode: str
      min_coverage: float | None
      cell_size_m: list[float] | None
      exported_tile_counts_by_year: dict[int, int]
      cells_produced: int
      seasons_kept: int
      seasons_dropped: int
      per_area_manifest_path: str | None
      cell_dirs: list[str]
      source_download_manifest: str | None
      passed: bool
      warnings: list[str]
      errors: list[str]
      run_parameters: dict[str, Any]
  ```

- [ ] **Step 1: Write the failing test**

Create `tests/test_raw_export_manifest.py`:

```python
from satmap_dataset.models import RawExportManifest


def test_raw_export_manifest_defaults_and_roundtrip():
    m = RawExportManifest(
        provider="geoportal",
        area="poznan",
        raw_root="/data/sat_data_raw",
        epsg=2180,
        cell_size_m=[2500.0, 2500.0],
        exported_tile_counts_by_year={2015: 4, 2018: 4},
        cells_produced=2,
        seasons_kept=3,
        seasons_dropped=1,
    )
    assert m.kind == "raw_export_manifest"
    assert m.stage == "raw_export"
    assert m.epsg_provider_mismatch is False
    assert m.link_mode == "symlink"
    assert m.passed is False  # explicit pass set by the stage
    dumped = m.model_dump(mode="json")
    assert dumped["exported_tile_counts_by_year"]["2015"] == 4
    assert RawExportManifest.model_validate(dumped).cells_produced == 2
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_raw_export_manifest.py -v`
Expected: FAIL with `ImportError: cannot import name 'RawExportManifest'`.

- [ ] **Step 3: Implement `RawExportManifest`**

Append to `models.py`:

```python
class RawExportManifest(BaseModel):
    """On-disk JSON contract for `raw_export_manifest.json` (the raw-export stage artifact)."""

    kind: Literal["raw_export_manifest"] = "raw_export_manifest"
    stage: Literal["raw_export"] = "raw_export"
    generated_at: datetime = Field(default_factory=_utc_now)
    provider: str
    area: str
    raw_root: str
    epsg: int | None = None
    epsg_provider_mismatch: bool = False
    link_mode: str = "symlink"
    min_coverage: float | None = None
    cell_size_m: list[float] | None = None
    exported_tile_counts_by_year: dict[int, int] = Field(default_factory=dict)
    cells_produced: int = 0
    seasons_kept: int = 0
    seasons_dropped: int = 0
    per_area_manifest_path: str | None = None
    cell_dirs: list[str] = Field(default_factory=list)
    source_download_manifest: str | None = None
    passed: bool = False
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    run_parameters: dict[str, Any] = Field(default_factory=dict)
```

Confirm `Literal`, `Any`, `datetime`, `_utc_now`, `Field`, `BaseModel` are imported at the top of `models.py` (they are — `LayerManifest`/`ValidationReport` use them).

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_raw_export_manifest.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/models.py tests/test_raw_export_manifest.py
git commit -m "feat(models): add RawExportManifest"
```

---

## Task 5: `pipeline/raw_export.py` stage

**Files:**
- Create: `src/satmap_dataset/pipeline/raw_export.py`
- Test: `tests/test_raw_export_stage.py`

**Interfaces:**
- Consumes: `RawExportConfig` (Task 3), `RawExportManifest` (Task 4), `raw_tiles.core.ingest_area`/`load_provider_registry`/`provider_for_epsg` (Task 1).
- Produces: `run(config: RawExportConfig) -> tuple[int, Path]` — exports native tiles, ingests cells, writes `config.output_json`, returns `(0|1, config.output_json)`. Also `_export_native_tiles(config) -> dict[int, int]` and `_can_reuse_raw_export(config, prior: RawExportManifest) -> bool`.

**Design notes (no resampling; provider authority):**
- The export step lays tiles into `<raw_root>/<provider>/<area>/<year>/` from the **configured** provider — discover per-year tiles by globbing `download_root/<year>/*.tif` (robust; matches `ingest_area`'s own `*/*.tif` glob). The download manifest path is recorded for provenance/reuse but glob is authoritative for the file list.
- `ingest_area(src_area=<raw_root>/<provider>/<area>, out_root=<raw_root>, registry, …)` derives the output provider dir from the tile EPSG; in the normal case it equals the configured provider, so cells co-locate with the export. The **EPSG cross-check** compares `provider_for_epsg(manifest["epsg"], registry)` to `config.provider` and sets `epsg_provider_mismatch` + a warning on disagreement (never reroutes).
- `link_mode` governs the **export** sidecar tiles only (symlink vs copy). `ingest_area` always symlinks native cells / crops sub-windows (its own contract).

- [ ] **Step 1: Write the failing stage test**

Create `tests/test_raw_export_stage.py`. It builds a tiny 2-year / 2-tile fixture with real georeferenced TIFFs (so `gdalinfo` reads a geotransform + EPSG), runs the stage, and asserts the layout, sidecars, `manifest.yaml`, and the JSON artifact. Uses a helper that writes a GeoTIFF via pyvips + a `.tfw`/`.prj` so `gdalinfo` reports geo (pyvips alone does not embed a CRS):

```python
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyvips
import pytest

from satmap_dataset.config import RawExportConfig
from satmap_dataset.pipeline import raw_export

# Geoportal EPSG:2180 WKT with a top-level authority so _epsg_from_wkt -> 2180.
_WKT_2180 = (
    'PROJCS["ETRS89 / Poland CS92",'
    'GEOGCS["ETRS89",DATUM["European_Terrestrial_Reference_System_1989",'
    'SPHEROID["GRS 1980",6378137,298.257222101]],PRIMEM["Greenwich",0],'
    'UNIT["degree",0.0174532925199433]],'
    'PROJECTION["Transverse_Mercator"],'
    'PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",19],'
    'PARAMETER["scale_factor",0.9993],PARAMETER["false_easting",500000],'
    'PARAMETER["false_northing",-5300000],UNIT["metre",1],'
    'AUTHORITY["EPSG","2180"]]'
)


def _has_gdal():
    return shutil.which("gdalinfo") is not None and shutil.which("gdal_translate") is not None


def _write_geotiff(path: Path, ulx: float, uly: float, gsd: float, n: int = 64):
    """Write an n×n RGB GeoTIFF at (ulx, uly) with the given GSD in EPSG:2180."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((n, n, 3), 200, np.uint8)
    img = pyvips.Image.new_from_memory(arr.tobytes(), n, n, 3, "uchar")
    tmp = path.with_suffix(".untagged.tif")
    img.tiffsave(str(tmp))
    # Tag georef via gdal_translate -a_srs / -a_ullr (upper-left, lower-right).
    lrx, lry = ulx + n * gsd, uly - n * gsd
    subprocess.run(
        ["gdal_translate", "-q", "-a_srs", "EPSG:2180",
         "-a_ullr", str(ulx), str(uly), str(lrx), str(lry), str(tmp), str(path)],
        check=True,
    )
    tmp.unlink()


@pytest.mark.skipif(not _has_gdal(), reason="GDAL CLI required")
def test_raw_export_end_to_end(tmp_path):
    download_root = tmp_path / "downloads_poznan"
    raw_root = tmp_path / "sat_data_raw"
    artifacts = tmp_path / "artifacts_poznan"
    # 2 years, one 64px tile each at the same origin/GSD (a single 1:1 native cell).
    for year in (2018, 2021):
        _write_geotiff(download_root / str(year) / f"tile_{year}.tif",
                       ulx=500000.0, uly=600000.0, gsd=0.25)

    cfg = RawExportConfig(
        provider="geoportal",
        area="poznan",
        download_root=download_root,
        raw_root=raw_root,
        artifacts_dir=artifacts,
        output_json=artifacts / "raw_export_manifest.json",
    )
    exit_code, artifact = raw_export.run(cfg)
    assert exit_code == 0, artifact

    # Exported native tiles
    assert (raw_root / "geoportal" / "poznan" / "2018" / "tile_2018.tif").exists()
    assert (raw_root / "geoportal" / "poznan" / "2021" / "tile_2021.tif").exists()

    # Ingested cell with sidecars (one cell, two seasons, native 1:1 symlinks)
    cell_dirs = list((raw_root / "geoportal" / "poznan").glob("e*_n*"))
    assert len(cell_dirs) == 1
    cell = cell_dirs[0]
    for year in (2018, 2021):
        assert (cell / f"year_{year}.tif").exists()
        assert (cell / f"year_{year}.tfw").exists()
        assert (cell / f"year_{year}.prj").exists()

    # Per-area manifest.yaml
    import yaml
    area_manifest = yaml.safe_load((raw_root / "geoportal" / "poznan" / "manifest.yaml").read_text())
    assert area_manifest["provider"] == "geoportal"
    assert area_manifest["epsg"] == 2180
    assert area_manifest["locations"]

    # Stage JSON artifact
    payload = json.loads(artifact.read_text())
    assert payload["kind"] == "raw_export_manifest"
    assert payload["passed"] is True
    assert payload["provider"] == "geoportal"
    assert payload["epsg"] == 2180
    assert payload["epsg_provider_mismatch"] is False
    assert payload["cells_produced"] == 1
    assert payload["exported_tile_counts_by_year"]["2018"] == 1


@pytest.mark.skipif(not _has_gdal(), reason="GDAL CLI required")
def test_raw_export_reuse_skips_second_run(tmp_path):
    download_root = tmp_path / "downloads_poznan"
    raw_root = tmp_path / "sat_data_raw"
    artifacts = tmp_path / "artifacts_poznan"
    for year in (2018, 2021):
        _write_geotiff(download_root / str(year) / f"tile_{year}.tif", 500000.0, 600000.0, 0.25)
    cfg = RawExportConfig(provider="geoportal", area="poznan", download_root=download_root,
                          raw_root=raw_root, artifacts_dir=artifacts,
                          output_json=artifacts / "raw_export_manifest.json")
    raw_export.run(cfg)
    first = json.loads((artifacts / "raw_export_manifest.json").read_text())["generated_at"]
    exit_code, artifact = raw_export.run(cfg)
    assert exit_code == 0
    second = json.loads(artifact.read_text())["generated_at"]
    assert first == second  # reused, manifest not rewritten
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_raw_export_stage.py -v`
Expected: FAIL with `ModuleNotFoundError: satmap_dataset.pipeline.raw_export`.

- [ ] **Step 3: Implement the stage**

Create `src/satmap_dataset/pipeline/raw_export.py`:

```python
"""Raw-tile export + ingest stage.

Lays native download tiles into <raw_root>/<provider>/<area>/<year>/*.tif, then
ingests co-located season-cell stacks (+ .tfw/.prj) and the per-area
manifest.yaml via the ported `raw_tiles` core. Writes raw_export_manifest.json.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from satmap_dataset.config import RawExportConfig
from satmap_dataset.models import RawExportManifest
from satmap_dataset.raw_tiles import core as rt


def _export_native_tiles(config: RawExportConfig) -> dict[int, int]:
    """Lay download_root/<year>/*.tif into <raw_root>/<provider>/<area>/<year>/.

    Returns a per-year exported-tile count. Symlinks by default; copies when
    link_mode == 'copy'.
    """
    src_root = Path(config.download_root)
    out_area = Path(config.raw_root) / config.provider / config.area
    counts: dict[int, int] = {}
    for year_dir in sorted(src_root.glob("*")):
        if not (year_dir.is_dir() and rt._YEAR_DIR_RE.match(year_dir.name)):
            continue
        year = int(year_dir.name)
        tiles = sorted(year_dir.glob("*.tif"))
        if not tiles:
            continue
        dest_dir = out_area / year_dir.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for tile in tiles:
            dest = dest_dir / tile.name
            if dest.is_symlink() or dest.exists():
                dest.unlink()
            if config.link_mode == "copy":
                shutil.copy2(tile, dest)
            else:
                dest.symlink_to(tile.resolve())
        counts[year] = len(tiles)
    return counts


def _can_reuse_raw_export(config: RawExportConfig, prior: RawExportManifest) -> bool:
    if not prior.passed or prior.stage != "raw_export":
        return False
    if (prior.provider, prior.area) != (config.provider, config.area):
        return False
    if str(prior.raw_root) != str(Path(config.raw_root)):
        return False
    if prior.link_mode != config.link_mode:
        return False
    if prior.min_coverage != config.min_coverage:
        return False
    out_area = Path(config.raw_root) / config.provider / config.area
    if not (out_area / "manifest.yaml").exists():
        return False
    return all((out_area / cell).exists() for cell in prior.cell_dirs)


def run(config: RawExportConfig) -> tuple[int, Path]:
    output_json = Path(config.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    # Idempotent reuse: if a prior artifact still matches and all outputs exist, skip.
    if output_json.exists():
        try:
            prior = RawExportManifest.model_validate_json(output_json.read_text())
        except Exception:
            prior = None
        if prior is not None and _can_reuse_raw_export(config, prior):
            print(str(output_json))
            return 0, output_json

    warnings: list[str] = []
    errors: list[str] = []

    # 1. Export native tiles into the canonical layout.
    exported = _export_native_tiles(config)
    if not exported:
        errors.append(f"No <year>/*.tif tiles found under {config.download_root}")
        manifest = RawExportManifest(
            provider=config.provider, area=config.area, raw_root=str(Path(config.raw_root)),
            link_mode=config.link_mode, min_coverage=config.min_coverage,
            passed=False, errors=errors,
            run_parameters={"download_root": str(config.download_root)},
        )
        output_json.write_text(manifest.model_dump_json(indent=2))
        print(str(output_json))
        return 1, output_json

    # 2. Ingest cells via the ported core.
    registry = rt.load_provider_registry()
    src_area = Path(config.raw_root) / config.provider / config.area
    area_manifest = rt.ingest_area(
        src_area, Path(config.raw_root), registry,
        cell_size_m=config.cell_size_m, min_coverage=config.min_coverage,
    )

    # EPSG cross-check (warning only).
    epsg = area_manifest.get("epsg")
    mismatch = False
    if epsg is not None:
        detected = rt.provider_for_epsg(epsg, registry)
        if detected != config.provider:
            mismatch = True
            warnings.append(
                f"EPSG:{epsg} maps to provider '{detected}' but configured provider "
                f"is '{config.provider}'"
            )

    # 3. Write the per-area manifest.yaml (handoff contract).
    out_provider_area = Path(config.raw_root) / area_manifest["provider"] / area_manifest["area"]
    out_provider_area.mkdir(parents=True, exist_ok=True)
    per_area_path = out_provider_area / "manifest.yaml"
    per_area_path.write_text(yaml.safe_dump(area_manifest, sort_keys=False))

    # 4. Tally + write the stage artifact.
    locations = area_manifest.get("locations", {})
    seasons_kept = sum(
        sum(1 for s in loc["seasons"] if not s.get("dropped")) for loc in locations.values()
    )
    seasons_dropped = sum(
        sum(1 for s in loc["seasons"] if s.get("dropped")) for loc in locations.values()
    )
    manifest = RawExportManifest(
        provider=config.provider,
        area=config.area,
        raw_root=str(Path(config.raw_root)),
        epsg=epsg,
        epsg_provider_mismatch=mismatch,
        link_mode=config.link_mode,
        min_coverage=config.min_coverage,
        cell_size_m=area_manifest.get("cell_size_m"),
        exported_tile_counts_by_year=exported,
        cells_produced=len(locations),
        seasons_kept=seasons_kept,
        seasons_dropped=seasons_dropped,
        per_area_manifest_path=str(per_area_path),
        cell_dirs=sorted(locations.keys()),
        source_download_manifest=str(config.download_manifest) if config.download_manifest else None,
        passed=True,
        warnings=warnings,
        errors=errors,
        run_parameters={"download_root": str(config.download_root)},
    )
    output_json.write_text(manifest.model_dump_json(indent=2))
    print(str(output_json))
    return 0, output_json
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_raw_export_stage.py -v`
Expected: PASS (both end-to-end and reuse tests). If GDAL CLI is absent the tests skip — run on a machine with `gdalinfo`/`gdal_translate`.

- [ ] **Step 5: Commit**

```bash
git add src/satmap_dataset/pipeline/raw_export.py tests/test_raw_export_stage.py
git commit -m "feat(pipeline): add raw-export stage (export + ingest + manifest)"
```

---

## Task 6: CLI command family + base/location builder

**Files:**
- Modify: `src/satmap_dataset/cli.py` (add 5 commands + `_build_raw_export_config_from_base_and_location`; reuse `_load_params_json_dict`, `_apply_location_paths_policy`, `_slugify_location_name`, `_finish`, `_print_validation_error`)
- Test: `tests/test_raw_export_cli.py`

**Interfaces:**
- Consumes: `RawExportConfig` (Task 3), `raw_export.run` (Task 5), `split_manifest.build_test_manifest` (Task 2).
- Produces CLI commands: `raw-export`, `raw-export-json`, `raw-export-location-json`, `raw-export-all-location-json`, `raw-test-manifest`; and `_build_raw_export_config_from_base_and_location(*, base_json: Path, location_json: Path) -> RawExportConfig` which derives `area = _slugify_location_name(location_name)` and `download_root`/`artifacts_dir` via `_apply_location_paths_policy`, and adds `raw_root` from env/default (a single shared root, NOT per-location).

- [ ] **Step 1: Write the failing test**

Create `tests/test_raw_export_cli.py`:

```python
import json
from pathlib import Path

from typer.testing import CliRunner

from satmap_dataset import cli
from satmap_dataset.config import RawExportConfig

runner = CliRunner()


def _write_base_and_location(tmp_path):
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"year_start": 2015, "year_end": 2021, "provider": "geoportal"}))
    loc = tmp_path / "location_poznan.json"
    loc.write_text(json.dumps({"location_name": "Poznań", "center_lat": 52.4, "center_lon": 16.9}))
    return base, loc


def test_builder_derives_area_and_shared_raw_root(tmp_path, monkeypatch):
    monkeypatch.setenv("SATMAP_RAW_ROOT", str(tmp_path / "sat_data_raw"))
    base, loc = _write_base_and_location(tmp_path)
    cfg = cli._build_raw_export_config_from_base_and_location(base_json=base, location_json=loc)
    assert isinstance(cfg, RawExportConfig)
    assert cfg.provider == "geoportal"
    assert cfg.area == "poznan"
    assert "downloads_poznan" in str(cfg.download_root)
    assert str(cfg.raw_root) == str(tmp_path / "sat_data_raw")  # shared, not per-location


def test_raw_export_json_invokes_run(tmp_path, monkeypatch):
    captured = {}

    def _fake_run(config):
        captured["config"] = config
        out = Path(config.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{}")
        return 0, out

    monkeypatch.setattr(cli.raw_export, "run", _fake_run)
    params = {
        "provider": "geoportal", "area": "poznan",
        "download_root": str(tmp_path / "downloads_poznan"),
        "raw_root": str(tmp_path / "sat_data_raw"),
        "artifacts_dir": str(tmp_path / "artifacts_poznan"),
        "output_json": str(tmp_path / "artifacts_poznan" / "raw_export_manifest.json"),
    }
    p = tmp_path / "params.json"
    p.write_text(json.dumps(params))
    result = runner.invoke(cli.app, ["raw-export-json", str(p)])
    assert result.exit_code == 0, result.stdout
    assert isinstance(captured["config"], RawExportConfig)
    assert captured["config"].area == "poznan"


def test_raw_export_json_rejects_sentinel2(tmp_path):
    params = {"provider": "sentinel2", "area": "x", "download_root": str(tmp_path)}
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(params))
    result = runner.invoke(cli.app, ["raw-export-json", str(p)])
    assert result.exit_code == 2


def test_raw_export_help_smoke():
    for cmd in ("raw-export", "raw-export-json", "raw-export-location-json",
                "raw-export-all-location-json", "raw-test-manifest"):
        result = runner.invoke(cli.app, [cmd, "--help"])
        assert result.exit_code == 0, result.stdout
        assert "Usage" in result.stdout
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_raw_export_cli.py -v`
Expected: FAIL (commands/builder not defined).

- [ ] **Step 3: Implement the CLI surface**

In `cli.py`, add the import near the other pipeline imports:

```python
from satmap_dataset.pipeline import raw_export
from satmap_dataset.raw_tiles.split_manifest import build_test_manifest
```

Add the base+location builder near the other `_build_*_config_from_base_and_location` helpers:

```python
def _build_raw_export_config_from_base_and_location(*, base_json: Path, location_json: Path) -> RawExportConfig:
    base_payload = _load_params_json_dict(base_json)
    location_payload = _load_params_json_dict(location_json)
    merged: dict[str, object] = dict(base_payload)
    merged.update(location_payload)
    repo_root = base_json.resolve().parents[2] if len(base_json.resolve().parents) >= 3 else Path.cwd().resolve()
    merged = _apply_location_paths_policy(merged, repo_root)
    location_name = merged.get("location_name")
    if location_name is None:
        raise typer.BadParameter("location JSON must set 'location_name'")
    merged.setdefault("area", _slugify_location_name(str(location_name)))
    artifacts_dir = Path(str(merged.get("artifacts_dir")))
    merged.setdefault("download_manifest", str(artifacts_dir / "dataset_manifest_download.json"))
    merged.setdefault("output_json", str(artifacts_dir / "raw_export_manifest.json"))
    # Drop keys that are not RawExportConfig fields (base.json carries many).
    allowed = set(RawExportConfig.model_fields)
    cleaned = {k: v for k, v in merged.items() if k in allowed}
    return RawExportConfig.model_validate(cleaned)
```

Add the five commands (place near the other stage commands). Flag form:

```python
@app.command("raw-export")
def raw_export_command(
    provider: str = typer.Option("geoportal", help="geoportal|lantmateriet|nls (sentinel2 rejected)."),
    area: str = typer.Option(..., help="Area slug (output namespace under <raw_root>/<provider>/)."),
    download_root: Path = typer.Option(..., help="downloads_<slug> root with <year>/*.tif."),
    raw_root: Path | None = typer.Option(None, help="Shared sat_data_raw root (default: $SATMAP_RAW_ROOT or ~/Github/sat_data_raw)."),
    download_manifest: Path | None = typer.Option(None, help="Optional download manifest for provenance."),
    min_coverage: float | None = typer.Option(None, help="Override per-provider coverage gate (0,1]."),
    link_mode: str = typer.Option("symlink", help="symlink|copy for exported native tiles."),
    cell_size_m: float | None = typer.Option(None, help="Override cell size in metres."),
    artifacts_dir: Path = typer.Option(Path("artifacts"), help="Where raw_export_manifest.json is written."),
    output_json: Path | None = typer.Option(None, help="Stage artifact path."),
) -> None:
    payload: dict[str, object] = {
        "provider": provider, "area": area, "download_root": str(download_root),
        "link_mode": link_mode, "artifacts_dir": str(artifacts_dir),
        "output_json": str(output_json) if output_json else str(artifacts_dir / "raw_export_manifest.json"),
    }
    if raw_root is not None:
        payload["raw_root"] = str(raw_root)
    if download_manifest is not None:
        payload["download_manifest"] = str(download_manifest)
    if min_coverage is not None:
        payload["min_coverage"] = min_coverage
    if cell_size_m is not None:
        payload["cell_size_m"] = cell_size_m
    try:
        config = RawExportConfig.model_validate(payload)
    except ValidationError as error:
        _print_validation_error(error)
        raise typer.Exit(code=2) from error
    exit_code, artifact_path = raw_export.run(config)
    _finish(exit_code, artifact_path)


@app.command("raw-export-json")
def raw_export_json_command(
    params_json: Path = typer.Argument(..., help="JSON file with RawExportConfig fields."),
) -> None:
    try:
        payload = _load_params_json_dict(params_json)
        config = RawExportConfig.model_validate(payload)
    except ValidationError as error:
        _print_validation_error(error)
        raise typer.Exit(code=2) from error
    exit_code, artifact_path = raw_export.run(config)
    _finish(exit_code, artifact_path)


@app.command("raw-export-location-json")
def raw_export_location_json_command(
    location_json: Path = typer.Argument(..., help="Location JSON (location_name, center_lat, center_lon)."),
    base_json: Path = typer.Option(Path("configs/run/base.json"), "--base-json"),
) -> None:
    try:
        config = _build_raw_export_config_from_base_and_location(base_json=base_json, location_json=location_json)
    except typer.BadParameter as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=2) from error
    except ValidationError as error:
        _print_validation_error(error)
        raise typer.Exit(code=2) from error
    exit_code, artifact_path = raw_export.run(config)
    _finish(exit_code, artifact_path)


@app.command("raw-export-all-location-json")
def raw_export_all_location_json_command(
    locations_dir: Path = typer.Option(Path("configs/run/locations"), "--locations-dir"),
    base_json: Path = typer.Option(Path("configs/run/base.json"), "--base-json"),
    continue_on_error: bool = typer.Option(False, "--continue-on-error"),
) -> None:
    location_files = sorted(locations_dir.glob("*.json"))
    if not location_files:
        console.print(f"[red]No location JSONs under {locations_dir}[/red]")
        raise typer.Exit(code=2)
    last_path: Path | None = None
    failures = 0
    for loc in location_files:
        try:
            config = _build_raw_export_config_from_base_and_location(base_json=base_json, location_json=loc)
            exit_code, last_path = raw_export.run(config)
            if exit_code != 0:
                failures += 1
                if not continue_on_error:
                    raise typer.Exit(code=1)
        except (typer.BadParameter, ValidationError) as error:
            failures += 1
            console.print(f"[red]{loc.name}: {error}[/red]")
            if not continue_on_error:
                raise typer.Exit(code=2) from error
    if last_path is not None:
        typer.echo(str(last_path))
    raise typer.Exit(code=1 if failures else 0)


@app.command("raw-test-manifest")
def raw_test_manifest_command(
    raw_root: Path | None = typer.Option(None, help="Shared sat_data_raw root (default: $SATMAP_RAW_ROOT or ~/Github/sat_data_raw)."),
    out: Path | None = typer.Option(None, help="Output split manifest path (default: <raw_root>/test_manifest.yaml)."),
    min_years: int = typer.Option(2, help="Minimum seasons per kept cell."),
) -> None:
    from satmap_dataset.config import _default_raw_root
    root = Path(raw_root) if raw_root is not None else _default_raw_root()
    out_path = Path(out) if out is not None else root / "test_manifest.yaml"
    build_test_manifest(root, out_path, min_years=min_years)
    typer.echo(str(out_path.resolve()))
```

If `ValidationError` / `console` / `typer` are not already imported in `cli.py`, they are (used throughout). Confirm `_default_raw_root` is exported from `config.py` (Task 3 defines it at module level).

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_raw_export_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite (no regressions)**

Run: `pytest -q`
Expected: PASS (existing tests unaffected; new tests green or skipped where GDAL absent).

- [ ] **Step 6: Commit**

```bash
git add src/satmap_dataset/cli.py tests/test_raw_export_cli.py
git commit -m "feat(cli): add raw-export command family + raw-test-manifest"
```

---

## Task 7: Justfile recipes + docs note

**Files:**
- Modify: `Justfile` (add 3 recipes mirroring the existing `*-location-json` recipes)
- Modify: `CLAUDE.md` (one line documenting the new opt-in stage under "Pipeline stages")

- [ ] **Step 1: Add Justfile recipes**

Append, mirroring the existing `*-location-json` / `*-all-json` recipe style:

```makefile
raw-export-location-json location_json="configs/run/locations/poznan.json" base_json="configs/run/base.json":
  loc="{{location_json}}"; loc="${loc#location_json=}"; base="{{base_json}}"; base="${base#base_json=}"; python -m satmap_dataset.cli raw-export-location-json "$loc" --base-json "$base"

raw-export-all-json locations_dir="configs/run/locations" base_json="configs/run/base.json" continue_on_error="--continue-on-error":
  root="${SATMAP_LOCATIONS_ROOT:-./configs/run}"; dir="{{locations_dir}}"; dir="${dir#locations_dir=}"; base="{{base_json}}"; base="${base#base_json=}"; if [[ "$dir" != /* && "$dir" != */* ]]; then dir="$root/$dir"; fi; python -m satmap_dataset.cli raw-export-all-location-json --locations-dir "$dir" --base-json "$base" {{continue_on_error}}

raw-test-manifest min_years="2":
  my="{{min_years}}"; my="${my#min_years=}"; python -m satmap_dataset.cli raw-test-manifest --min-years "$my"
```

- [ ] **Step 2: Verify recipes parse**

Run: `just --list | grep raw`
Expected: the three `raw-*` recipes appear.

- [ ] **Step 3: Document in CLAUDE.md**

Under the pipeline/architecture section, add one line:

> Opt-in `raw-export` stage (`pipeline/raw_export.py`, not in `run-all`): turns native download tiles into `sat_data_raw/<provider>/<area>/{<year>,<cellkey>}` + per-area `manifest.yaml`; `raw-test-manifest` builds the cross-location split `test_manifest.yaml` consumed by sat_roma's `raw_tile_pipeline`. Ingestion core is a ported copy of sat_roma `romatch/datasets/raw_tiles.py` under `src/satmap_dataset/raw_tiles/`.

- [ ] **Step 4: Commit**

```bash
git add Justfile CLAUDE.md
git commit -m "feat(just): raw-export recipes + docs"
```

---

## Verification (end-to-end)

1. **Unit + stage tests:** `pytest tests/test_raw_tiles_core.py tests/test_raw_tiles_split_manifest.py tests/test_raw_export_config.py tests/test_raw_export_manifest.py tests/test_raw_export_stage.py tests/test_raw_export_cli.py -v` — all pass (stage/CLI tests requiring GDAL run on a machine with `gdalinfo`/`gdal_translate`).
2. **Full suite:** `pytest -q` — no regressions.
3. **Real run against existing downloads** (manual smoke, requires a populated `downloads_<slug>`):
   ```bash
   just run-location-json location_json=configs/run/locations/poznan.json   # ensure downloads exist
   just raw-export-location-json location_json=configs/run/locations/poznan.json
   ls "${SATMAP_RAW_ROOT:-$HOME/Github/sat_data_raw}/geoportal/poznan"       # <year>/, e*_n*/, manifest.yaml
   cat "${SATMAP_RAW_ROOT:-$HOME/Github/sat_data_raw}/geoportal/poznan/manifest.yaml"
   just raw-test-manifest
   cat "${SATMAP_RAW_ROOT:-$HOME/Github/sat_data_raw}/test_manifest.yaml"
   ```
   Confirm: native tiles symlinked under `<year>/`, ingested `year_YYYY.tif`/`.tfw`/`.prj` under `e*_n*/`, valid `manifest.yaml` (provider/epsg/locations), and a `test_manifest.yaml` with `roots:` + per-location `test.query`/`test.ref`.
4. **Idempotency:** re-run `just raw-export-location-json …` and confirm the stage reports reuse (artifact `generated_at` unchanged).
5. **sentinel2 rejection:** a JSON with `"provider": "sentinel2"` → `raw-export-json` exits `2`.

## Self-Review notes (spec coverage)

- §4.1 ported core + registry + drift test → Task 1; split manifest → Task 2. ✅
- §4.2 stage (export→ingest→manifest, reuse, EPSG cross-check, link_mode) → Task 5. ✅
- §4.3 `RawExportConfig` + `RawExportManifest` → Tasks 3, 4. ✅
- §4.4 3-flavour CLI + `raw-test-manifest` + Justfile → Tasks 6, 7. ✅
- §6 no-resampling / provider-aware coverage / EPSG warning / sentinel2 reject → enforced in Tasks 1, 3, 5. ✅
- §8 open points resolved: (1) export discovers tiles by glob of `download_root/<year>/*.tif`, manifest recorded for provenance; (2) single shared `raw_root` via `SATMAP_RAW_ROOT`; (3) `_can_reuse_raw_export` checks provider/area/raw_root/link_mode/min_coverage + on-disk cells + manifest.yaml; (4) tests in Tasks 1/5/6. ✅
