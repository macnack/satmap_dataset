# Architecture

`satmap_dataset` builds year-aware orthophoto (and optional DEM/OSM) stacks for
machine-learning research. Every stage writes one JSON manifest and returns
`(exit_code, artifact_path)`.

## Stages

| Stage | Module | Artifact |
|-------|--------|----------|
| Index | `pipeline/index_builder.py` (via provider) | `index_manifest.json`, `year_availability_report.json` |
| Download | `pipeline/downloader.py` (via provider) | `dataset_manifest_download.json` + TIFF assets |
| Render | `pipeline/render.py` | `dataset_manifest_render.json` + `year_YYYY.tif` |
| Validate | `pipeline/validator.py` | `validation_report.json` |
| DEM | `pipeline/dem.py` | `dem_manifest.json` |
| OSM | `pipeline/osm.py` | `osm_manifest.json` |
| Raw export | `pipeline/raw_export.py` | `raw_export_manifest.json` (opt-in) |

Exit codes: `0` success, `1` policy/data failure, `2` invalid CLI/config.
CLI last stdout line: absolute path of the artifact written.

## Orchestration

Preferred mental model: **layers on a shared grid**.

1. RGB layer runs index → download → render and defines a `ReferenceGrid`.
2. Optional DEM / OSM layers align to that grid.
3. Optional RGB validation.

On current `main`, RGB-only orchestration is `pipeline/run_all.py`; multimodal
is `pipeline/location_run.py`. A unified `PipelineConfig` orchestrator may land
via a follow-up PR (`feat/unified-orchestrator`).

## Providers

RGB acquisition goes through `providers.get_provider(name)`:

| Provider | Status | Notes |
|----------|--------|-------|
| `geoportal` | Supported | Polish PZGiK WFS + WMS; default `EPSG:2180` |
| `lantmateriet` | Supported | Swedish STAC (+ optional paid WMS); `EPSG:3006` |
| `nls` | Supported (download-first) | Finnish WCS/OAPIF; needs API key; `EPSG:3067` |
| `sentinel2` | Experimental | STAC COGs; set `target_srs` carefully |
| `esri_wayback` | Experimental | Esri Wayback WMTS; capture-year versions; EPSG:3857 tiles reprojected at render; restrictive Esri terms |
| `lroc_nac` | Deferred | Lunar PDS index/download; projection/render out of scope |

DEM/OSM today are Geoportal/Overpass-oriented stages, not full provider plugins.

## Configs and paths

- Defaults: `configs/run/base.json` (+ provider-specific bases).
- Locations: `configs/run/locations/<name>.json` (`location_name`, center, area).
- With `location_name` set, download/render/artifact roots are derived from a
  slug under the repo (e.g. `downloads_poznan`).

## Idempotent reuse

Index/download reuse skips work when manifests match the request and assets
exist. Treat any non-empty download file as complete — interrupted downloads
must be deleted manually (writers should use `.part` then rename).

## Further reading

- Agent contracts: [`CLAUDE.md`](../CLAUDE.md)
- Internal design archive: [`docs/superpowers/`](superpowers/) (not user docs)
- Tech debt notes: [`docs/tech_debt/architecture_review.md`](tech_debt/architecture_review.md)
