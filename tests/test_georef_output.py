from __future__ import annotations

import sys
from pathlib import Path

import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.pipeline import render


def test_apply_geotiff_tags_and_sidecars(tmp_path: Path) -> None:
    out = tmp_path / "year_2023.tif"
    data = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
    tifffile.imwrite(out, data, photometric="rgb", compression="deflate")

    bbox = render.BBox(min_x=210300.0, min_y=521900.0, max_x=210302.0, max_y=521902.0)
    render._ensure_georeferenced_output(
        out_path=out,
        target_bbox=bbox,
        target_width=2,
        target_height=2,
        srs="EPSG:2180",
    )

    with tifffile.TiffFile(out) as tif:
        page = tif.pages[0]
        assert page.tags.get("ModelPixelScaleTag") is not None
        assert page.tags.get("ModelTiepointTag") is not None
        assert page.tags.get("GeoKeyDirectoryTag") is not None

    tfw = out.with_suffix(".tfw")
    prj = out.with_suffix(".prj")
    assert tfw.exists()
    assert prj.exists()

    tfw_lines = [line.strip() for line in tfw.read_text(encoding="ascii").splitlines() if line.strip()]
    assert len(tfw_lines) == 6
    assert "EPSG\",\"2180" in prj.read_text(encoding="ascii")
