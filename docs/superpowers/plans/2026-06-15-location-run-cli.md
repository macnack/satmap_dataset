# `location-run-json` CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the already-built `pipeline/location_run.py:run_location` aligned RGB+DEM+OSM orchestrator as a single CLI command `location-run-json` plus a matching `just` target.

**Architecture:** Add one Typer command to `cli.py` that builds `RunConfig`/`DemConfig`/`OsmConfig` from base+location via the existing `_build_*_config_from_base_and_location` helpers, derives `artifacts_dir` from `run_config.artifacts_dir`, and calls `location_run.run_location(...)`. `--no-dem`/`--no-osm`/`--no-validate` flags gate the optional modalities; the DEM/OSM configs are only built when their flag is on. No new orchestration logic — `run_location` already exists and is tested by `tests/test_location_run.py`.

**Tech Stack:** Python 3.10+, Typer (CLI), Pydantic v2 (configs), pytest + `typer.testing.CliRunner`, `just`.

---

## File Structure

- **Modify** `src/satmap_dataset/cli.py`
  - Extend the pipeline import (line 30) to include `location_run`.
  - Add the `location-run-json` command (place it immediately after the existing `run_location_json_command`, ~line 1160, so the RGB-only and full-stack location commands sit together).
- **Modify** `Justfile` — add a `location-run-json` target alongside `run-location-json`.
- **Create** `tests/test_location_run_cli.py` — CLI wiring test using `CliRunner` with `run_location` monkeypatched.

The orchestrator (`pipeline/location_run.py`), the three config builders, and the layers are unchanged.

---

## Task 1: Add the `location-run-json` CLI command (TDD)

**Files:**
- Modify: `src/satmap_dataset/cli.py:30` (import) and after `src/satmap_dataset/cli.py:1159` (new command)
- Test: `tests/test_location_run_cli.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_location_run_cli.py`:

```python
import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from satmap_dataset.cli import app
from satmap_dataset.config import DemConfig, OsmConfig, RunConfig

runner = CliRunner()


def _write_base_and_location(tmp_path):
    base = tmp_path / "base.json"
    base.write_text(json.dumps({
        "year_start": 2015, "year_end": 2016, "mode": "hybrid",
        "profile": "reference", "srs": "EPSG:2180", "area_km2": 4.0,
    }))
    loc = tmp_path / "location_poznan.json"
    loc.write_text(json.dumps({
        "location_name": "Poznań", "center_lat": 52.4, "center_lon": 16.9,
    }))
    return base, loc


def test_location_run_json_wires_all_three_layers(tmp_path, monkeypatch):
    captured = {}

    def _fake_run_location(**kwargs):
        captured.update(kwargs)
        out = Path(kwargs["artifacts_dir"]) / "rgb_layer_manifest.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{}")
        return (0, out)

    monkeypatch.setattr(
        "satmap_dataset.cli.location_run.run_location", _fake_run_location
    )
    base, loc = _write_base_and_location(tmp_path)

    result = runner.invoke(app, ["location-run-json", str(loc), "--base-json", str(base)])

    assert result.exit_code == 0, result.stdout
    assert isinstance(captured["rgb_config"], RunConfig)
    assert isinstance(captured["dem_config"], DemConfig)
    assert isinstance(captured["osm_config"], OsmConfig)
    assert captured["run_dem"] is True
    assert captured["run_osm"] is True
    assert captured["validate"] is True
    # artifacts_dir is the RGB config's, populated by the location-paths policy.
    assert captured["artifacts_dir"] == captured["rgb_config"].artifacts_dir
    assert "artifacts_poznan" in str(captured["artifacts_dir"])
    # Last stdout line is the RGB manifest path (shell-composition contract).
    assert result.stdout.strip().splitlines()[-1].endswith("rgb_layer_manifest.json")


def test_location_run_json_no_dem_no_osm(tmp_path, monkeypatch):
    captured = {}

    def _fake_run_location(**kwargs):
        captured.update(kwargs)
        out = Path(kwargs["artifacts_dir"]) / "rgb_layer_manifest.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{}")
        return (0, out)

    monkeypatch.setattr(
        "satmap_dataset.cli.location_run.run_location", _fake_run_location
    )
    base, loc = _write_base_and_location(tmp_path)

    result = runner.invoke(app, [
        "location-run-json", str(loc), "--base-json", str(base),
        "--no-dem", "--no-osm", "--no-validate",
    ])

    assert result.exit_code == 0, result.stdout
    assert captured["dem_config"] is None
    assert captured["osm_config"] is None
    assert captured["run_dem"] is False
    assert captured["run_osm"] is False
    assert captured["validate"] is False


def test_location_run_json_bad_location_exit_2(tmp_path):
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"year_start": 2015, "year_end": 2016}))
    loc = tmp_path / "bad.json"
    loc.write_text(json.dumps({"location_name": "X"}))  # no center -> bbox cannot resolve

    result = runner.invoke(app, ["location-run-json", str(loc), "--base-json", str(base)])
    assert result.exit_code == 2
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_location_run_cli.py -v`
Expected: FAIL — the `location-run-json` command does not exist yet, so `runner.invoke` returns a non-zero exit (Typer "no such command") and the assertions fail / `captured` is empty.

- [ ] **Step 3: Add the `location_run` import**

In `src/satmap_dataset/cli.py`, change line 30 from:

```python
from satmap_dataset.pipeline import dem, dem_availability, downloader, index_builder, render, run_all, validator, osm as osm_pipeline
```

to:

```python
from satmap_dataset.pipeline import dem, dem_availability, downloader, index_builder, location_run, render, run_all, validator, osm as osm_pipeline
```

- [ ] **Step 4: Add the command**

Insert immediately after the end of `run_location_json_command` (after its `_finish(...)` call, before the `@app.command("render-location-json")` decorator, around line 1160):

```python
@app.command("location-run-json")
def location_run_json_command(
    location_json: Path = typer.Argument(..., help="Path to location JSON (location_name, center_lat, center_lon)."),
    base_json: Path = typer.Option(
        Path("configs/run/base.json"),
        "--base-json",
        help="Path to base JSON with shared run parameters.",
    ),
    dem: bool = typer.Option(
        True, "--dem/--no-dem", help="Produce the DEM layer aligned to the RGB grid."
    ),
    osm: bool = typer.Option(
        True, "--osm/--no-osm", help="Produce the OSM label layer aligned to the RGB grid."
    ),
    validate: bool = typer.Option(
        True, "--validate/--no-validate", help="Run the validator on the RGB layer manifest."
    ),
) -> None:
    """Produce the aligned RGB + DEM + OSM stack for one location in one pass.

    Unlike ``run-location-json`` (RGB only), this drives the layer orchestrator:
    the RGB layer defines the shared ReferenceGrid and the DEM/OSM layers align
    to it in memory (no re-reading a render manifest from disk).
    """
    try:
        rgb_config = _build_run_config_from_base_and_location(
            base_json=base_json, location_json=location_json
        )
        dem_config = (
            _build_dem_config_from_base_and_location(
                base_json=base_json, location_json=location_json
            )
            if dem
            else None
        )
        osm_config = (
            _build_osm_config_from_base_and_location(
                base_json=base_json, location_json=location_json
            )
            if osm
            else None
        )
    except typer.BadParameter as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=2) from error
    except ValidationError as error:
        _print_validation_error(error)
        raise typer.Exit(code=2) from error

    exit_code, artifact_path = location_run.run_location(
        rgb_config=rgb_config,
        dem_config=dem_config,
        osm_config=osm_config,
        artifacts_dir=rgb_config.artifacts_dir,
        run_dem=dem,
        run_osm=osm,
        validate=validate,
    )
    _finish(exit_code, artifact_path)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `pytest tests/test_location_run_cli.py -v`
Expected: PASS (all three tests).

- [ ] **Step 6: Run the full suite to confirm no regressions**

Run: `pytest -q`
Expected: all tests pass (the prior baseline was 270 passed; this adds 3).

- [ ] **Step 7: Commit**

```bash
git add src/satmap_dataset/cli.py tests/test_location_run_cli.py
git commit -m "feat(cli): location-run-json drives the aligned RGB+DEM+OSM orchestrator"
```

---

## Task 2: Add the `just` target

**Files:**
- Modify: `Justfile` (after the `run-location-json` recipe, ~line 28)

- [ ] **Step 1: Add the recipe**

In `Justfile`, immediately after the `run-location-json` recipe (the two lines at 27–28), add:

```
# Full aligned stack (RGB + DEM + OSM) for one location, one grid.
location-run-json location_json="configs/run/locations/poznan.json" base_json="configs/run/base.json":
  loc="{{location_json}}"; loc="${loc#location_json=}"; base="{{base_json}}"; base="${base#base_json=}"; python -m satmap_dataset.cli location-run-json "$loc" --base-json "$base"
```

- [ ] **Step 2: Verify the recipe parses and the command is reachable**

Run: `just --list | grep location-run-json`
Expected: the `location-run-json` recipe is listed.

Run: `python -m satmap_dataset.cli location-run-json --help`
Expected: help text shows `--dem/--no-dem`, `--osm/--no-osm`, `--validate/--no-validate`, and `--base-json`.

- [ ] **Step 3: Commit**

```bash
git add Justfile
git commit -m "feat(just): add location-run-json target for the aligned stack"
```

---

## Self-Review Notes

- **Spec coverage:** CLI command (Task 1), `--no-dem`/`--no-osm`/`--no-validate` flags (Task 1, command + tests), `just` target (Task 2), CLI wiring test (Task 1). Batch / atomic-writes / registry-CLI explicitly out of scope per the spec. ✓
- **Contract preservation:** exit code via `run_location`'s `max()`; last stdout line is the RGB manifest path, asserted in `test_location_run_json_wires_all_three_layers`. ✓
- **Type consistency:** `run_location` is called with the exact kwargs from its signature (`rgb_config`, `dem_config`, `osm_config`, `artifacts_dir`, `run_dem`, `run_osm`, `validate`); `rgb_config.artifacts_dir` is a real `RunConfig` field (`config.py:245`). ✓
- **No placeholders:** every code/command step is concrete. ✓
```
