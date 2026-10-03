# LROC cam2map Project Stage Implementation Plan

> **For agentic workers:** Implement task-by-task. Steps use checkbox syntax.

**Goal:** Ship an optional post-download LROC projection stage that invokes ISIS `lronac2isis` → `spiceinit` → `cam2map` when available, with mocked CI tests and clear failure when ISIS is missing.

**Architecture:** `LrocProjectConfig` + `LrocProjectManifest` + pure `isis.py` argv helpers + `pipeline/lroc_project.run` + CLI `lroc-project` / `lroc-project-json`.

**Tech Stack:** Python 3.10+, Pydantic v2, subprocess, typer (existing CLI).

## Global Constraints

- Stage API: `run(config) -> tuple[int, Path]`; exit 0/1 from stage; CLI exit 2 on bad config; last stdout line = artifact path.
- No hard ISIS dependency in CI; mock `which` / `run_cmd`.
- Lunar CRS default `IAU_2015:30100`.
- Do not wire into run-all / Studio in this slice.
- Work only in `.worktrees/lroc-cam2map` on `feat/lroc-cam2map`.

---

## Task 1: Models + config

- [ ] Add `LrocProjectManifest` to `models.py`
- [ ] Add `LrocProjectConfig` to `config.py` (lunar SRS validation)
- [ ] Tests: config accepts lunar SRS; rejects Earth SRS

## Task 2: ISIS helpers

- [ ] Create `providers/lroc_nac/isis.py` with resolve + argv builders
- [ ] Tests: argv shape; missing-tool detection

## Task 3: Stage + CLI + sample + docs

- [ ] Create `pipeline/lroc_project.py`
- [ ] Wire `lroc-project` / `lroc-project-json` in `cli.py`
- [ ] Sample `configs/run/lroc_nac_apollo17.project.json`
- [ ] Update README + CLAUDE.md
- [ ] Stage tests with mocked subprocess (happy path + missing ISIS)

## Task 4: Verify + PR

- [ ] `pytest` focused + related suite
- [ ] Commit, push, `gh pr create` vs main
