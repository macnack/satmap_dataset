"""Validator must enforce non-EPSG:2180 CRS (e.g. NLS Finland's EPSG:3067)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import tifffile

from satmap_dataset.config import ValidateConfig
from satmap_dataset.models import DatasetManifest, ValidationReport
from satmap_dataset.pipeline import validator


def _write_geotiff(
    out_path: Path,
    *,
    width: int,
    height: int,
    target_bbox: tuple[float, float, float, float],
    epsg: int,
) -> None:
    """Write a minimal RGB GeoTIFF with georef tags for the given EPSG."""
    arr = np.full((height, width, 3), 128, dtype=np.uint8)
    xmin, ymin, xmax, ymax = target_bbox
    px = (xmax - xmin) / width
    py = (ymax - ymin) / height
    scale = (float(px), float(py), 0.0)
    tie = (0.0, 0.0, 0.0, float(xmin), float(ymax), 0.0)
    geokey = (
        1, 1, 0, 4,
        1024, 0, 1, 1,
        1025, 0, 1, 1,
        3072, 0, 1, int(epsg),
        3076, 0, 1, 9001,
    )
    extratags = [
        (33550, "d", 3, scale, False),
        (33922, "d", 6, tie, False),
        (34735, "H", len(geokey), geokey, False),
    ]
    tifffile.imwrite(out_path, arr, photometric="rgb", extratags=extratags)
    out_path.with_suffix(".tfw").write_text(
        f"{px}\n0.0\n0.0\n{-py}\n{xmin + px / 2}\n{ymax - py / 2}\n",
        encoding="ascii",
    )


def test_validator_flags_wrong_epsg_for_3067_target(tmp_path: Path):
    """A GeoTIFF tagged with EPSG:2180 must fail validation when target_srs=EPSG:3067."""
    asset = tmp_path / "year_2018.tiff"
    bbox = (351089.0, 6671973.0, 353089.0, 6673973.0)
    _write_geotiff(asset, width=64, height=64, target_bbox=bbox, epsg=2180)

    manifest = DatasetManifest(
        provider="nls",
        stage="render",
        mode="wcs",
        years_requested=[2018],
        years_included=[2018],
        assets=[str(asset)],
        target_bbox="351089,6671973,353089,6673973",
        target_srs="EPSG:3067",
        target_width=64,
        target_height=64,
        passed=True,
    )
    manifest_path = tmp_path / "dataset_manifest_render.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    cfg = ValidateConfig(
        dataset_manifest=manifest_path,
        requested_years=[2018],
        output_json=tmp_path / "validation_report.json",
    )
    exit_code, report_path = validator.run(cfg)
    assert exit_code != 0
    report = ValidationReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    assert any("expected 3067" in err for err in report.errors), report.errors


def test_validator_accepts_correct_epsg_3067(tmp_path: Path):
    asset = tmp_path / "year_2018.tiff"
    bbox = (351089.0, 6671973.0, 353089.0, 6673973.0)
    _write_geotiff(asset, width=64, height=64, target_bbox=bbox, epsg=3067)

    manifest = DatasetManifest(
        provider="nls",
        stage="render",
        mode="wcs",
        years_requested=[2018],
        years_included=[2018],
        assets=[str(asset)],
        target_bbox="351089,6671973,353089,6673973",
        target_srs="EPSG:3067",
        target_width=64,
        target_height=64,
        passed=True,
    )
    manifest_path = tmp_path / "dataset_manifest_render.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    cfg = ValidateConfig(
        dataset_manifest=manifest_path,
        requested_years=[2018],
        output_json=tmp_path / "validation_report.json",
    )
    exit_code, report_path = validator.run(cfg)
    report = ValidationReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    # The asset has the right CRS, so no EPSG-related error should appear.
    assert not any("EPSG" in err for err in report.errors), report.errors
