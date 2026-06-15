# Design — Expose the aligned-stack orchestrator via CLI (`location-run-json`)

**Date:** 2026-06-15
**Branch:** `feat/layer-abstraction`
**Status:** approved design, pending implementation

## Problem

The Layer-abstraction refactor (`4455aaf`) introduced
`pipeline/location_run.py:run_location`, which produces the aligned
RGB + DEM + OSM stack for one location on a single shared `ReferenceGrid`.
It is fully implemented and unit-tested (`tests/test_location_run.py`) but
**not reachable from the CLI or `just`**. Today a user still runs three
separate manual commands (`run-location-json` for RGB, then
`dem-location-json`, then `osm-location-json`), each re-deriving alignment
from a manifest on disk.

This is the keystone open item from
`docs/tech_debt/architecture_review.md` §3 ("No single orchestrator for the
aligned multimodal output"). The MVP wires the existing orchestrator to a CLI
command — no new orchestration logic.

## Scope

In scope (MVP):
- One new single-location CLI command `location-run-json`.
- One `just` target of the same name.
- A CLI-level test.

Explicitly out of scope (per scoping decision):
- Batch `*-all-location-json` variant.
- Atomic `.part` download writes (architecture_review §5).
- Layer-registry-driven CLI collapse (architecture_review §6).

## Design

### CLI command: `location-run-json`

Lives in `cli.py`, mirroring the existing `*-location-json` command family.

Signature (Typer):
- `location_json: Path` (positional) — location JSON
  (`location_name`, `center_lat`, `center_lon`).
- `--base-json: Path = configs/run/base.json`.
- `--dem/--no-dem` (default: on).
- `--osm/--no-osm` (default: on).
- `--validate/--no-validate` (default: on).

Body:
1. Build `RunConfig` via `_build_run_config_from_base_and_location`.
2. If DEM enabled, build `DemConfig` via
   `_build_dem_config_from_base_and_location`; else `None`.
3. If OSM enabled, build `OsmConfig` via
   `_build_osm_config_from_base_and_location`; else `None`.
4. `artifacts_dir = run_config.artifacts_dir` (already populated by the
   location-paths policy inside the builders).
5. Call:
   ```python
   exit_code, artifact_path = location_run.run_location(
       rgb_config=run_config,
       dem_config=dem_config,
       osm_config=osm_config,
       artifacts_dir=artifacts_dir,
       run_dem=dem,
       run_osm=osm,
       validate=validate,
   )
   ```
6. `_finish(exit_code, artifact_path)`.

Config-build errors (`typer.BadParameter`, `ValidationError`) → exit 2,
identical to sibling commands. Building DEM/OSM configs only when their flag is
on means a `base.json` lacking that modality's params still works with the flag
off.

### Contract preservation

- Exit code: most-severe (highest) across RGB / validation / DEM / OSM — already
  implemented by `run_location` via `max()`.
- Last stdout line: absolute path of the RGB layer manifest
  (`<artifacts_dir>/rgb_layer_manifest.json`), printed by `_finish`.
- A grid/CRS mismatch raises `ValueError` from the DEM/OSM layer (existing
  guard); it surfaces as an error rather than being swallowed.

### `just` target

```
location-run-json location_json base_json="configs/run/base.json":
  python -m satmap_dataset.cli location-run-json {{location_json}} --base-json "{{base_json}}"
```
Mirrors `dem-location-json` / `osm-location-json`.

## Testing

New `tests/test_location_run_cli.py` using Typer's `CliRunner`:
- Monkeypatch `cli.location_run.run_location` to capture kwargs and return
  `(0, tmp_path / "rgb_layer_manifest.json")`.
- Assert: the three configs are the right types and carry the location's bbox /
  artifacts_dir; `run_dem`/`run_osm`/`validate` follow the flags.
- Assert `--no-dem` yields `dem_config=None` and `run_dem=False` (and the same
  for OSM), and that the command exits 0 and prints the manifest path last.

The orchestrator's stage-sequencing and exit-code logic is already covered by
`tests/test_location_run.py`; this test only verifies the CLI wiring.

## Naming note

`run-location-json` (existing, RGB-only) and `location-run-json` (new, full
stack) coexist. The distinction is intentional and documented in each command's
help text.
