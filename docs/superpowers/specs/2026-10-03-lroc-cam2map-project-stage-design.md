# LROC NAC projection stage (ISIS cam2map) — design

**Status:** approved for vertical slice (parent task brief)  
**Date:** 2026-10-03  
**Depends on:** `docs/superpowers/specs/2026-06-25-lroc-nac-provider-design.md` (index + download)

## Problem

`provider="lroc_nac"` downloads unprojected PDS NAC camera-geometry frames
(typically `.IMG`). Those assets are not yet map-projected, so the existing
Earth-oriented `render` stage cannot mosaic them onto a shared grid. The prior
provider design deferred ISIS `cam2map` to a follow-up; this spec is that
follow-up’s **first vertical slice**.

## Goals (this slice)

1. Optional post-download **project** stage with Stage API
   `run(config) -> (exit_code, Path)` writing one JSON manifest.
2. Invoke ISIS tools when present on `PATH`: `lronac2isis` → `spiceinit` →
   `cam2map` (per downloaded frame).
3. Clear, CI-friendly failure when ISIS is missing (no hard CI dependency;
   unit tests mock subprocess / `which`).
4. Keep lunar CRS `IAU_2015:30100` (ocentric lon/lat) as the stage default.
5. Minimal CLI: flag form + JSON form (not location-batch / run-all).
6. Document ISIS expectation + sample config in README / CLAUDE.md.

## Non-goals (deferred)

- Wiring into `run-all` / Studio DAG / unified orchestrator.
- Full `render` of projected cubes to NN-ready GeoTIFF stacks.
- `isis2std` / GDAL export to GeoTIFF (may follow once cubes exist).
- Docker/conda packaging of ISIS itself.
- Photogrammetric bundle adjustment / multi-epoch co-registration QC.
- Support for non-`lroc_nac` providers.

## Approaches considered

| | Approach | Pros | Cons |
|---|----------|------|------|
| **A (chosen)** | Opt-in `lroc-project` stage after download | Matches Stage API; no CI ISIS dep; clear skip; mirrors raw-export | Extra CLI command; not yet in run-all |
| B | Project inside `LrocNacProvider.download` | Fewer commands | Couples ISIS to download; hard for CI; slower reuse |
| C | Extend `render` to call cam2map | One less stage name | Mixes pyvips Earth path with ISIS lunar path |

**Recommendation:** A.

## Architecture

```
downloads_*/<year>/<pdsid>.IMG
        │
        ▼
 pipeline/lroc_project.run(LrocProjectConfig)
        │  uses providers/lroc_nac/isis.py (argv + PATH checks)
        ▼
 project_root/<year>/<pdsid>_map.cub
 artifacts_*/project_manifest.json
```

### Config — `LrocProjectConfig`

| Field | Default | Notes |
|-------|---------|-------|
| `download_manifest` | required | LayerManifest from download stage |
| `download_root` | optional | If set, used only for provenance; assets come from manifest |
| `project_root` | `projected` | Output cubes root |
| `srs` | `IAU_2015:30100` | Must be lunar `IAU_2015:301xx` |
| `map_file` | `None` | Optional ISIS map template for `cam2map map=` |
| `overwrite` | `False` | Skip existing non-empty `*_map.cub` when false |
| `keep_work_cubes` | `False` | Keep intermediate `.cub` from `lronac2isis` |
| `artifacts_dir` | `artifacts` | |
| `output_json` | `artifacts/project_manifest.json` | |

Invalid config → CLI exit `2` via Pydantic (same as other stages).

### Manifest — `LrocProjectManifest`

On-disk contract (`kind="lroc_project_manifest"`, `stage="lroc_project"`):

- `srs`, `project_root`, `source_download_manifest`
- `assets_projected: list[str]` (absolute or relative paths of `*_map.cub`)
- `assets_skipped: list[str]` (reuse / missing source)
- `assets_failed: list[{source, step, message}]`
- `isis_tools: dict[str, str | null]` (resolved binaries)
- `passed`, `warnings`, `errors`, `run_parameters`

### ISIS command builder (`providers/lroc_nac/isis.py`)

Pure helpers (easy to unit-test):

- `REQUIRED_TOOLS = ("lronac2isis", "spiceinit", "cam2map")`
- `resolve_isis_tools(which=shutil.which) -> dict[str, Path | None]`
- `isis_available(tools) -> bool`
- `build_lronac2isis_cmd(from_img, to_cub) -> list[str]`
- `build_spiceinit_cmd(from_cub) -> list[str]`
- `build_cam2map_cmd(from_cub, to_map, *, map_file=None) -> list[str]`

Runtime check: if any required tool is missing, stage writes manifest with
`passed=false`, `errors=["ISIS required but not on PATH: …"]`, exit `1`.
Does **not** raise an uncaught exception.

### Stage behavior (`pipeline/lroc_project.py`)

1. Validate download manifest exists and `provider == "lroc_nac"` (else error,
   exit `1`).
2. Resolve ISIS tools; if missing → fail as above.
3. For each path in `download_manifest.assets` that exists and is non-empty:
   - Derive year from parent dir name when numeric, else `"unknown"`.
   - Work cube: `project_root/<year>/<stem>.cub`
   - Map cube: `project_root/<year>/<stem>_map.cub`
   - If map cube exists and `not overwrite` → skip (count as projected).
   - Else run: `lronac2isis` → `spiceinit` → `cam2map` via injectable
     `run_cmd(argv) -> CompletedProcess`-shaped callable (tests mock this).
   - On failure of any step: record in `assets_failed`, continue others.
4. `passed = bool(assets_projected) and not assets_failed and not errors`.
5. Write manifest; print absolute `output_json`; return `(0|1, path)`.

Exit codes: `0` success, `1` policy/data/ISIS/tool failure, `2` only from CLI
config validation (stage itself returns 0/1).

### CLI

- `lroc-project` — flag form
- `lroc-project-json` — JSON form mapped onto `LrocProjectConfig`

No `*-location-json` / run-all integration in this slice.

### Sample config

`configs/run/lroc_nac_apollo17.project.json` pointing at the existing Apollo 17
download artifact paths.

### Testing

- Unit tests mock `resolve_isis_tools` / `run_cmd`; assert argv sequences and
  manifest fields for happy path.
- Unit test: missing ISIS → exit `1`, error string mentions ISIS / PATH.
- Config test: non-lunar `srs` rejected; non-`lroc_nac` download provider →
  stage exit `1`.
- No live ISIS requirement in CI.

### Documentation

- README provider table: `lroc_nac` note → index/download + optional project.
- CLAUDE.md LROC section: how to enable, ISIS on PATH, sample commands.
- This design doc + short plan under `docs/superpowers/`.

## Success criteria

- `pytest` green without ISIS installed.
- With ISIS on PATH and a real download tree, `lroc-project-json` produces
  `*_map.cub` files and a `passed=true` manifest (manual / researcher machine).
