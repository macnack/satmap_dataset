# Unified pipeline orchestrator

## Problem

Two entry points (`run_all.run` vs `location_run.run_location`) with different
return artifacts, different validation targets, and duplicated stage wiring.
Newcomers cannot tell which command builds a full location stack.

## Decision

**One orchestrator** driven by `PipelineConfig`:

1. RGB layer (defines `ReferenceGrid`)
2. Optional DEM / OSM layers (consume that grid)
3. Optional RGB validate (`run_validate`)
4. Always write `pipeline_manifest.json` summarizing the run

`run_all.run` and `location_run.run_location` become thin wrappers that preserve
their existing return-path contracts for CLI/studio callers.

RGB core lives in `pipeline/rgb_pipeline.py` (reuse predicates + index/download/render).
Geoportal index reuse rejects WMS-only stubs when `mode != wms_tiled` and vice versa.

## Alternatives considered

| Approach | Pros | Cons |
|----------|------|------|
| A. Layer-list orchestrator (chosen) | Matches existing Layer ABC; one mental model | Slight migration of RGB-only path |
| B. Keep both, shared private helper only | Minimal churn | Dual public APIs remain confusing |
| C. Full CLI rewrite around layers | Ideal long-term | Out of scope for this change |

## Contracts

- `orchestrator.run(PipelineConfig) -> (exit_code, Path)` → `pipeline_manifest.json`
- Exit code: RGB failure short-circuits; DEM/OSM/validate use `max(code)` so
  later failures are not masked
- Index reuse for geoportal rejects WMS-only stubs when `mode != wms_tiled` and
  vice versa
