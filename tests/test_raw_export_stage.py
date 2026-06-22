import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyvips
import pytest

from satmap_dataset.config import RawExportConfig
from satmap_dataset.pipeline import raw_export


def _has_gdal():
    return shutil.which("gdalinfo") is not None and shutil.which("gdal_translate") is not None


def _write_geotiff(path: Path, ulx: float, uly: float, gsd: float, n: int = 64):
    """Write an n×n RGB GeoTIFF at (ulx, uly) with the given GSD in EPSG:2180."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((n, n, 3), 200, np.uint8)
    img = pyvips.Image.new_from_memory(arr.tobytes(), n, n, 3, "uchar")
    tmp = path.with_suffix(".untagged.tif")
    img.tiffsave(str(tmp))
    # Tag georef via gdal_translate -a_srs / -a_ullr (upper-left, lower-right).
    lrx, lry = ulx + n * gsd, uly - n * gsd
    subprocess.run(
        ["gdal_translate", "-q", "-a_srs", "EPSG:2180",
         "-a_ullr", str(ulx), str(uly), str(lrx), str(lry), str(tmp), str(path)],
        check=True,
    )
    tmp.unlink()


@pytest.mark.skipif(not _has_gdal(), reason="GDAL CLI required")
def test_raw_export_end_to_end(tmp_path):
    download_root = tmp_path / "downloads_poznan"
    raw_root = tmp_path / "sat_data_raw"
    artifacts = tmp_path / "artifacts_poznan"
    # 2 years, one 64px tile each at the same origin/GSD (a single 1:1 native cell).
    for year in (2018, 2021):
        _write_geotiff(download_root / str(year) / f"tile_{year}.tif",
                       ulx=500000.0, uly=600000.0, gsd=0.25)

    cfg = RawExportConfig(
        provider="geoportal",
        area="poznan",
        download_root=download_root,
        raw_root=raw_root,
        artifacts_dir=artifacts,
        output_json=artifacts / "raw_export_manifest.json",
    )
    exit_code, artifact = raw_export.run(cfg)
    assert exit_code == 0, artifact

    # Exported native tiles
    assert (raw_root / "geoportal" / "poznan" / "2018" / "tile_2018.tif").exists()
    assert (raw_root / "geoportal" / "poznan" / "2021" / "tile_2021.tif").exists()

    # Ingested cell with sidecars (one cell, two seasons, native 1:1 symlinks)
    cell_dirs = list((raw_root / "geoportal" / "poznan").glob("e*_n*"))
    assert len(cell_dirs) == 1
    cell = cell_dirs[0]
    for year in (2018, 2021):
        assert (cell / f"year_{year}.tif").exists()
        assert (cell / f"year_{year}.tfw").exists()
        assert (cell / f"year_{year}.prj").exists()

    # Per-area manifest.yaml
    import yaml
    area_manifest = yaml.safe_load((raw_root / "geoportal" / "poznan" / "manifest.yaml").read_text())
    assert area_manifest["provider"] == "geoportal"
    assert area_manifest["epsg"] == 2180
    assert area_manifest["locations"]

    # Stage JSON artifact
    payload = json.loads(artifact.read_text())
    assert payload["kind"] == "raw_export_manifest"
    assert payload["passed"] is True
    assert payload["provider"] == "geoportal"
    assert payload["epsg"] == 2180
    assert payload["epsg_provider_mismatch"] is False
    assert payload["cells_produced"] == 1
    assert payload["exported_tile_counts_by_year"]["2018"] == 1


@pytest.mark.skipif(not _has_gdal(), reason="GDAL CLI required")
def test_raw_export_reuse_skips_second_run(tmp_path):
    download_root = tmp_path / "downloads_poznan"
    raw_root = tmp_path / "sat_data_raw"
    artifacts = tmp_path / "artifacts_poznan"
    for year in (2018, 2021):
        _write_geotiff(download_root / str(year) / f"tile_{year}.tif", 500000.0, 600000.0, 0.25)
    cfg = RawExportConfig(provider="geoportal", area="poznan", download_root=download_root,
                          raw_root=raw_root, artifacts_dir=artifacts,
                          output_json=artifacts / "raw_export_manifest.json")
    raw_export.run(cfg)
    first = json.loads((artifacts / "raw_export_manifest.json").read_text())["generated_at"]
    exit_code, artifact = raw_export.run(cfg)
    assert exit_code == 0
    second = json.loads(artifact.read_text())["generated_at"]
    assert first == second  # reused, manifest not rewritten


@pytest.mark.skipif(not _has_gdal(), reason="GDAL CLI required")
def test_raw_export_no_tiles_fails(tmp_path):
    cfg = RawExportConfig(provider="geoportal", area="empty",
                          download_root=tmp_path / "downloads_empty",
                          raw_root=tmp_path / "sat_data_raw",
                          artifacts_dir=tmp_path / "artifacts_empty",
                          output_json=tmp_path / "artifacts_empty" / "raw_export_manifest.json")
    (tmp_path / "downloads_empty").mkdir()
    exit_code, artifact = raw_export.run(cfg)
    assert exit_code == 1
    payload = json.loads(artifact.read_text())
    assert payload["passed"] is False
    assert payload["errors"]
