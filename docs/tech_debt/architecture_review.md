# Architecture Review — Scalability & Adding New Services

Whole-project review focused on one question: **how cheap is it to add a new
map service (another Geoportal product, another country's orthophoto, another
OSM-style label source) and have it land in the aligned multi-band output?**

Scope: `src/satmap_dataset/{providers,pipeline,cli.py,config.py,models.py}`,
the `Justfile`, and the `docs/superpowers/plans/` direction. Ranked
most-structural first.

---

## Verdict

The foundation is sound — the provider registry, the uniform stage contract,
and the Pydantic manifest schemas are the right bones. But **the abstraction
stops at RGB acquisition.** The project has grown three parallel silos —
*provider RGB* / *DEM* / *OSM* — that share no common notion of "a layer that
contributes aligned bands on a shared grid." That missing abstraction is
exactly what makes adding a new service expensive today. The
`aligned-multiband-ml-stack` plan confirms the real target abstraction is a
**Layer / Modality**, not a *Provider*.

---

## What is right (keep)

- **Provider registry** — `providers/__init__.py:get_provider` + the
  `Provider` ABC (`providers/base.py`). RGB acquisition is genuinely
  swappable: `geoportal`, `lantmateriet`, `sentinel2`.
- **Uniform stage contract** — every stage is `run(config) -> (exit_code, Path)`
  writing one JSON manifest, with load-bearing exit codes. Composable and
  testable.
- **`provider_options` + ENV fallback** — the `_option(options, key, env_var,
  default)` pattern (lantmateriet/sentinel2) is a clean way to carry
  per-service config without bloating the shared schema.
- **Reuse predicates now check provider** — `run_all.py:102` (`_can_reuse_index`)
  and `run_all.py:148` (`_can_reuse_download`) both gate on
  `manifest.provider == config.provider`. This closed a `nls_provider_review.md`
  item.
- **Manifest schemas as the on-disk contract** (`models.py`, Pydantic v2) and
  the slug → directory convention, extended cleanly to `dem_<slug>` /
  `osm_<slug>`.

---

## Structural (raise the cost of every new service)

### 1. The `Provider` ABC is too thin and is bypassed
**Files:** `providers/base.py:15-21`, `providers/geoportal.py:20`,
`cli.py:1428,1480,1501,1541`, `run_all.py:317`, `cli.py:834,853,1097,1180`

The ABC declares only `index` + `download`. Everything else routes around it:

- `dem()` exists **only** on `GeoportalProvider` (`providers/geoportal.py:20`),
  not in the ABC. The CLI calls `dem.run()` directly, never via the registry.
- OSM does **not** go through providers at all — `osm.run()` is called directly.
- `render` is **not** in the provider interface — `run_all.py:317` and four CLI
  sites call `render.run()` unconditionally.

**Consequence:** "providers" abstract only *RGB acquisition*, never
transformation. `render.py` (1281 lines) is implicitly Geoportal- /
EPSG:2180-centric — the Sentinel-2 provider docstring itself admits the render
stage does not reproject across CRS. A new service with a different CRS or
spectral layout has no clean hook.

**Direction:** widen the contract so transformation (`render`/`harmonize`) and
elevation/label modalities are first-class, or — better — replace `Provider`
with `Layer` (see Recommendation).

---

### 2. Two taxonomies that do not compose
**Files:** `config.py` (`RunConfig`, `DemConfig`, `OsmConfig`,
`DemAvailabilityConfig`), `models.py` (`DatasetManifest`, `DemManifest`,
`OsmManifest`), `cli.py`, `Justfile`

`provider` (RGB) and the hardcoded DEM/OSM stages are conceptually the same
thing — *layers that contribute bands to a shared grid* — but implemented as
disjoint silos: separate config classes, separate manifests, separate CLI
command families (`run-*` / `dem-*` / `osm-*`), separate `just` targets.

**Adding one service touches:** `config.py` + `models.py` + `cli.py` (×3
command flavors) + `Justfile` + `scripts/manage_location_roots.py` + the reuse
predicates. High N-touch cost, and easy to get partially wrong.

---

### 3. No single orchestrator for the aligned multimodal output
**Files:** `run_all.py` (no `dem`/`osm` references), plan
`docs/superpowers/plans/2026-05-30-aligned-multiband-ml-stack.md`

`run_all` orchestrates RGB only: index → download → render → validate. DEM and
OSM are separate manual invocations. The actual *product* — the aligned
multi-band stack — exists only as a plan, assembled post-hoc by a future
`dem_stack.py` that **auto-discovers files by naming convention**
(`rendered_<slug>/year_<year>.tiff`, `dem_<slug>/skorowidz/...`). That is
coupling through hardcoded paths instead of through manifests, and it breaks
the moment a path convention changes.

**Direction:** a per-location orchestrator driven by a list of requested
layers, composing manifests (not globbed paths) into the stack.

---

### 4. Dispatch logic is fragmented
**File:** `dem.py:250`

`dem.run` carries its own mini-dispatcher (`if config.transport ==
"skorowidz": return dem_skorowidz.run(...)`), a second selection mechanism
living beside the provider registry. Each such ad-hoc branch is one more place
a new service has to be wired by hand.

---

## Robustness (bites at batch scale)

### 5. Downloads are not atomic
**Files:** `providers/lantmateriet/provider.py:305-307` (and the equivalent
geoportal/sentinel2 write paths)

Assets are streamed straight to the final path. An interrupted download
(SIGINT, OOM, network reset) leaves a truncated `.tif`; the next run's
existence check (`output_path.exists() and st_size > 0`) treats it as complete
and skips it, propagating a corrupt tile into render. Same issue was flagged
for NLS in `nls_provider_review.md:#3`.

**Fix:** stream to `output_path.with_suffix(".part")`, then `Path.rename()`
atomically on success; unlink `.part` on failure. Cheap, and necessary before
`run-all` over many locations is trustworthy.

---

### 6. CLI boilerplate scales linearly with services
**File:** `cli.py` (1777 lines)

Three command flavors (flag / `*-json` / `*-location-json` + `*-all-*`) ×
every stage, each with a hand-written `_build_*_config_from_base_and_location`
helper. Most of the file is repetition. Every new modality multiplies it.

**Direction:** drive the CLI from a layer registry/table — one generic
`acquire <layer>` family resolving the layer by name — instead of N
hand-coded command sets.

---

## Recommended direction (for the follow-up plan)

Generalize the existing `Provider` into a **`Layer` (modality)**. Minimal
conceptual change, resolves items 1–3 together:

```
Layer.produce(config, grid: ReferenceGrid) -> (exit_code, LayerManifest)
```

Each layer (`rgb-geoportal`, `rgb-lantmateriet`, `dem-nmt`, `osm-buildings`, …)
declares: which bands it contributes, its native CRS, whether it needs
reprojection.

- **One reference-grid resolver** computes the grid once; every layer aligns to
  it — removing the discovery-by-filename coupling the stack plan relies on.
- **Adding a service = one `Layer` subclass + a registry entry**, not six files.
- **Render becomes a layer hook** (`harmonize`/`prepare`); grid alignment stays
  generic, so Geoportal's color-norm stops leaking onto every provider.
- **The multi-band stack is the natural composition** of a layer list, not a
  bolt-on.

Independent of the refactor, also: atomic `.part`-then-rename writes across all
providers (#5), and a single per-location `run-all` orchestrating the full
stack from a requested-layers list (#3).

---

## Quick reference: cost of "add a new service" today vs. target

| Step | Today | With `Layer` |
|------|-------|--------------|
| Acquisition logic | new provider class (if RGB) **or** new stage module | one `Layer` subclass |
| Config | new `*Config` or extend `RunConfig` | `provider_options` on shared config |
| Manifest | new `*Manifest` model | shared `LayerManifest` |
| CLI | 3 command flavors hand-written | registry entry |
| `just` | new targets | none (generic command) |
| Orchestration | manual; not in `run_all` | declare in layer list |
| Reuse predicate | extend by hand | generic, keyed on layer + params |
