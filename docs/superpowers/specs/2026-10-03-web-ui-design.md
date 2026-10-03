# satmap-web — lightweight browser UI (design)

**Date:** 2026-10-03  
**Branch:** `feat/web-ui`  
**Status:** Approved by parent-agent brief (no interactive user gate); documented here for PR review.

## Context

`satmap-studio` (Streamlit) already covers interactive runs. PR #16 adds year-timeline and pipeline-DAG panels there. A second Streamlit surface would fight that branch. This work adds a **separate** stack that sits alongside studio.

## Decision

**Stack:** FastAPI (API + static serving) + Vite/React SPA (`web-ui/`).

**Why not extend Streamlit:** studio remains the full form-heavy operator console; web is a lighter first viewport for location pick → status → run trigger / CLI deeplink. Timeline/DAG status derivation is vendored as pure helpers under `satmap_dataset.web.status` (mirror of PR #16 pure functions, no Streamlit HTML) so merges stay friendly until those helpers land on main.

## Scope (v1)

- List / select location JSON under `configs/run/locations`
- Show merged summary (provider, year range, paths, SRS)
- Derive on-disk pipeline status: year timeline + stage DAG
- Trigger `index` or full `run` in a background thread (reuses Pydantic configs + stage `run()`)
- Or copy a documented CLI deeplink
- `just web` / `just install-web` recipes; API smoke tests; frontend production build

## Out of scope

- Full studio form parity (map search, DEM/OSM toggles, secret writers)
- Reimplementing providers
- Auth / multi-user

## Architecture

```
Browser (React)  --REST-->  FastAPI (satmap_dataset.web)
                              |-- studio.config_builders (merge)
                              |-- web.status (timeline + DAG)
                              |-- pipeline.run_all / index_builder (jobs)
                              `-- serves web-ui/dist in prod
```

## Design language

Satellite-ops aesthetic: deep slate/teal atmosphere, brand-first **satmap**, Syne + IBM Plex Mono (not Inter/system), purposeful motion on timeline chips and running DAG nodes. Avoid purple gradients, cream+terracotta, broadsheet clutter.
