# Task 5 Report: `LrocNacProvider._download_async`

## Files Modified
- `src/satmap_dataset/providers/lroc_nac/provider.py` — added imports, module-level helpers, replaced stub
- `tests/test_lroc_nac_download.py` — created (new file)

## Test Commands and Output

### Step 2 — Failing test (before implementation)
```
pytest tests/test_lroc_nac_download.py -v
```
Result: `FAILED` — `AttributeError: module 'satmap_dataset.providers.lroc_nac.provider' has no attribute '_download_asset_with_retry'`

### Step 4 — Passing test (after implementation)
```
pytest tests/test_lroc_nac_download.py -v
```
```
============================= test session starts ==============================
collected 1 item
tests/test_lroc_nac_download.py .                                        [100%]
============================== 1 passed in 0.35s ==============================
```

### Task 4 regression check
```
pytest tests/test_lroc_nac_index.py -v
```
```
============================= test session starts ==============================
collected 3 items
tests/test_lroc_nac_index.py ...                                         [100%]
============================== 3 passed in 0.26s ==============================
```

## Commit Hash
`df0d249`

## `role="rgb"` — Accepted
`LayerManifest.role` is `Literal["rgb", "dem", "labels"]` — `"rgb"` is valid, no change needed.

## `mode="ode"` — Changed to `"stac"`
`LayerManifest.mode` is `Literal["wms_tiled", "wfs_render", "hybrid", "stac"]`. The brief specified `mode="ode"` which is not in the allowed set and would raise a Pydantic `ValidationError`. Changed to `"stac"` as the closest semantic match (ODE is a catalog endpoint analogous to STAC). This is the only deviation from the brief.

## Self-Review Notes
- `_download_asset_with_retry` is module-level (not a method) and called by bare name in `_download_async` — monkeypatch works correctly.
- No duplicate imports: `asyncio` and `Path` were already present; added only `random`, `urlparse`, `aiofiles`, `httpx`, `LayerManifest`.
- Minimal scope: no YAGNI additions. `_ext_for_url` and `_NON_RETRYABLE_STATUSES` are exactly as specified.
- Worker pattern (queue + sentinel None) matches lantmateriet provider pattern per design decision.

## Concerns
None. One intentional deviation: `mode="stac"` instead of `mode="ode"` to satisfy Pydantic's Literal constraint on `LayerManifest.mode`.

---

## Fix Report (Task 5 follow-up correctness fix)

### Two edits made

**1. `src/satmap_dataset/models.py`, line 317 — `LayerManifest.mode` Literal extended**

```python
# before
mode: Literal["wms_tiled", "wfs_render", "hybrid", "stac"] = "hybrid"
# after
mode: Literal["wms_tiled", "wfs_render", "hybrid", "stac", "ode"] = "hybrid"
```

Note: the worktree's `LayerManifest` is defined at line 271 (mode field at line 317), not line ~126 as the task brief estimated. Line 126 is `DatasetManifest.mode` and was left unchanged.

**2. `src/satmap_dataset/providers/lroc_nac/provider.py`, `_download_async` — mode corrected**

```python
# before
mode="stac",
# after
mode="ode",
```

### pytest — lroc_nac tests

```
python -m pytest tests/test_lroc_nac_download.py tests/test_lroc_nac_index.py -v
============================= test session starts ==============================
collected 4 items
tests/test_lroc_nac_download.py .                                        [ 25%]
tests/test_lroc_nac_index.py ...                                         [100%]
============================== 4 passed in 0.21s ===============================
```

### pytest — Lantmateriet + EPSG-3067 regression

```
python -m pytest tests/test_lantmateriet_config.py tests/test_validator_epsg_3067.py -v
============================= test session starts ==============================
collected 10 items
tests/test_lantmateriet_config.py ........                               [ 80%]
tests/test_validator_epsg_3067.py ..                                     [100%]
============================== 10 passed in 0.13s ===============================
```

### Commit hash

`422dbb6`
