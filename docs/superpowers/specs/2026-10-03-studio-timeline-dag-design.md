# Studio year timeline + pipeline DAG

## Goal

Make satmap-studio runs scannable: a compact year timeline (requested vs available vs downloaded/rendered) and a pipeline DAG with stage status derived from on-disk manifests.

## Approach (chosen)

**Streamlit-native HTML/CSS** via `st.markdown(..., unsafe_allow_html=True)`. No new dependencies (no plotly/graphviz).

Alternatives considered:
- Plotly/Altair charts — heavier, not in studio extras.
- Graphviz — nice DAG edges, extra system/Python dep.

## Modules

- `studio/timeline.py` — pure `derive_year_timeline(...)` + `timeline_html(...)`
- `studio/pipeline_dag.py` — pure `derive_pipeline_dag(...)` + `dag_html(...)`
- Thin hooks in `studio/app.py` under Availability (timeline) and Run & status (DAG + timeline from disk)

## Year timeline status (per year in range)

| Level | Meaning |
|-------|---------|
| `rendered` | Year in render manifest `years_included` or `year_YYYY.tif(f)` on disk |
| `downloaded` | Year in download manifest `years_included` or non-empty `download_root/<year>/` |
| `available` | Index/availability says `has_features` or year in `years_available_wfs` / `years_included` |
| `missing` | Requested but not available |
| `out_of_range` | Not used (range is closed) |

Color hierarchy: rendered > downloaded > available > missing.

## Pipeline DAG stages

Core: `index → download → render → validate`

Optional (config flags): `dem`, `osm` (after render), `raw_export` (after download).

Status from artifacts:
- `done` / `failed` from manifest `passed` (or presence + errors)
- `pending` if artifact missing and stage enabled
- `skipped` if config disabled the stage
- `running` when an active studio job maps to that stage (progress label / first incomplete stage)

Artifact paths (slug from location name):
- `artifacts_<slug>/{index_manifest,year_availability_report,dataset_manifest_download,dataset_manifest_render,validation_report,raw_export_manifest}.json`
- `dem_<slug>/dem_manifest.json`, `osm_<slug>/osm_manifest.json`

## UI placement

- **Availability** tab: timeline above the existing year table when a report/manifest is present; also load from disk for current location.
- **Run & status** tab: “Run status” section with DAG + timeline from disk for the selected location (visible even before clicking Run).

## Testing

Unit tests for derivation helpers with fixture JSON dicts / temp dirs. No Streamlit e2e.
