from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.pipeline.downloader import BBox, _tag_wms_tile_as_geotiff


def test_tag_wms_tile_as_geotiff_supports_epsg_3006(tmp_path: Path) -> None:
    path = tmp_path / "wms_2020.tiff"
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    arr[:, :, 0] = 10
    tifffile.imwrite(path, arr, photometric="rgb")

    bbox = BBox(min_x=536000.0, min_y=6426000.0, max_x=538000.0, max_y=6428000.0)
    _tag_wms_tile_as_geotiff(path, bbox, 32, 32, "EPSG:3006")

    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        tags = {tag.name: tag.value for tag in page.tags.values()}
    assert "ModelPixelScaleTag" in tags or 33550 in page.tags
    # ProjectedCSTypeGeoKey holds the EPSG code (3072 key → value 3006).
    geokeys = page.tags[34735].value
    assert 3006 in geokeys
