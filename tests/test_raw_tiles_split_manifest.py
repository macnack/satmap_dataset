import numpy as np
import pyvips
import yaml

from satmap_dataset.raw_tiles.split_manifest import build_test_manifest


def _write_tif(path, w=8, h=8):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((h, w, 3), 200, np.uint8)
    img = pyvips.Image.new_from_memory(arr.tobytes(), w, h, 3, "uchar")
    img.tiffsave(str(path))


def test_build_test_manifest_picks_richest_cell(tmp_path):
    root = tmp_path / "sat_data_raw"
    area = root / "geoportal" / "poznan"
    # rich cell: 3 equal-dim years; poor cell: 1 year
    for y in (2015, 2018, 2021):
        _write_tif(area / "e500_n600" / f"year_{y}.tif")
    _write_tif(area / "e700_n800" / "year_2019.tif")

    out = root / "test_manifest.yaml"
    manifest = build_test_manifest(root, out, min_years=2)

    loc_name = "geoportal_poznan_e500_n600"
    assert loc_name in manifest
    assert manifest[loc_name]["root"] == "locs"
    assert manifest[loc_name]["test"]["query"] == 2021
    assert manifest[loc_name]["test"]["ref"] == [2018, 2015]
    # poor cell excluded (below min_years)
    assert "geoportal_poznan_e700_n800" not in manifest
    # YAML written and round-trips
    on_disk = yaml.safe_load(out.read_text())
    assert on_disk[loc_name]["test"]["query"] == 2021
    # symlink location dir materialised
    link = root / "_manifest_locs" / loc_name / "year_2021.tif"
    assert link.is_symlink()
