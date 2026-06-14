"""Regression tests for `.prj` sidecar validation.

Render writes a `.prj` for every CRS it has WKT for; a missing one for such a
CRS is a real defect and must be flagged. For CRSes render has no WKT for, the
`.prj` is intentionally absent and must not fail validation.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.pipeline import validator
from satmap_dataset.pipeline.render import EPSG_2180_WKT, EPSG_3067_WKT


def _write_worldfile(asset: Path) -> None:
    asset.with_suffix(".tfw").write_text(
        "0.25\n0.0\n0.0\n-0.25\n351089.125\n6673972.875\n", encoding="ascii"
    )


def test_missing_prj_for_known_epsg_is_flagged(tmp_path: Path):
    asset = tmp_path / "year_2018.tiff"
    _write_worldfile(asset)
    errors = validator._validate_sidecars(asset, "EPSG:2180")
    assert any("Missing projection file" in e for e in errors)


def test_missing_prj_for_unknown_epsg_is_not_flagged(tmp_path: Path):
    asset = tmp_path / "year_2018.tiff"
    _write_worldfile(asset)
    # 9999 is not a known projected CRS and not a UTM zone, so render would skip
    # the .prj and the validator must not flag its absence.
    errors = validator._validate_sidecars(asset, "EPSG:9999")
    assert not any("projection file" in e.lower() for e in errors)


def test_present_prj_with_matching_authority_passes(tmp_path: Path):
    asset = tmp_path / "year_2018.tiff"
    _write_worldfile(asset)
    asset.with_suffix(".prj").write_text(EPSG_3067_WKT, encoding="ascii")
    errors = validator._validate_sidecars(asset, "EPSG:3067")
    assert not any("projection file" in e.lower() for e in errors)


def test_present_prj_with_wrong_authority_is_flagged(tmp_path: Path):
    asset = tmp_path / "year_2018.tiff"
    _write_worldfile(asset)
    # A CS92/EPSG:2180 .prj declared against an EPSG:3006 target shares no
    # authority code or fallback marker, so it must be flagged.
    asset.with_suffix(".prj").write_text(EPSG_2180_WKT, encoding="ascii")
    errors = validator._validate_sidecars(asset, "EPSG:3006")
    assert any("does not indicate EPSG:3006" in e for e in errors)
