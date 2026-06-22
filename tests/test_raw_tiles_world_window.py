import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyvips
import pytest

from satmap_dataset.raw_tiles.core import GeoTransform, TileInfo, load_provider_registry, world_window_to_pixel
from satmap_dataset.raw_tiles.world_window import common_window, ingest_area_world_window


def _has_gdal():
    return shutil.which("gdalinfo") is not None and shutil.which("gdal_translate") is not None


def _tile(ulx, uly, gsd, n, year):
    return TileInfo(None, n, n, GeoTransform(ulx, gsd, 0.0, uly, 0.0, -gsd), 2180, "wkt", year)


# ---- pure window math (mirrors the real Poznań translation: fine grid offset +0.05,-0.15) ----

def test_common_window_snaps_to_coarse_grid_and_aligns_both():
    # coarse 0.25 m tile and fine 0.05 m tile (offset +0.05 E, -0.15 S) over ~the same 10 m spot.
    coarse = _tile(1000.0, 2000.0, 0.25, 40, 2020)    # footprint 10 m, top 2000, bottom 1990
    fine = _tile(1000.05, 1999.85, 0.05, 200, 2021)   # footprint 10 m, top 1999.85, bottom 1989.85
    win = common_window({2020: coarse, 2021: fine})
    assert win is not None
    ulx, uly, w_m, h_m, res = win
    assert res == 0.25  # coarsest defines the grid
    # window sits on the coarse grid (origin 1000.0, 2000.0)
    assert abs((ulx - 1000.0) / 0.25 - round((ulx - 1000.0) / 0.25)) < 1e-9
    assert abs((2000.0 - uly) / 0.25 - round((2000.0 - uly) / 0.25)) < 1e-9
    # and crops losslessly (integer windows) from BOTH tiles
    for t in (coarse, fine):
        x, y, w, h = world_window_to_pixel(t.gt, ulx, uly, w_m, h_m)
        assert x >= 0 and y >= 0 and x + w <= t.width and y + h <= t.height
    # equal-dimension after decimation: coarse px == fine px / 5
    cx = world_window_to_pixel(coarse.gt, ulx, uly, w_m, h_m)
    fx = world_window_to_pixel(fine.gt, ulx, uly, w_m, h_m)
    assert fx[2] == cx[2] * 5 and fx[3] == cx[3] * 5


def test_common_window_single_gsd_is_full_footprint():
    a = _tile(1000.0, 2000.0, 0.25, 40, 2018)
    b = _tile(1000.0, 2000.0, 0.25, 40, 2020)
    ulx, uly, w_m, h_m, res = common_window({2018: a, 2020: b})
    assert (ulx, uly, w_m, h_m, res) == (1000.0, 2000.0, 10.0, 10.0, 0.25)


# ---- end-to-end ingest on tiny real GeoTIFFs ----

_WKT_2180 = (
    'PROJCS["ETRS89 / Poland CS92",GEOGCS["ETRS89",'
    'DATUM["European_Terrestrial_Reference_System_1989",'
    'SPHEROID["GRS 1980",6378137,298.257222101]],PRIMEM["Greenwich",0],'
    'UNIT["degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],'
    'PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",19],'
    'PARAMETER["scale_factor",0.9993],PARAMETER["false_easting",500000],'
    'PARAMETER["false_northing",-5300000],UNIT["metre",1],AUTHORITY["EPSG","2180"]]'
)


def _write_geotiff(path: Path, ulx, uly, gsd, n):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((n, n, 3), 200, np.uint8)
    img = pyvips.Image.new_from_memory(arr.tobytes(), n, n, 3, "uchar")
    tmp = path.with_suffix(".untagged.tif")
    img.tiffsave(str(tmp))
    lrx, lry = ulx + n * gsd, uly - n * gsd
    subprocess.run(["gdal_translate", "-q", "-a_srs", "EPSG:2180",
                    "-a_ullr", str(ulx), str(uly), str(lrx), str(lry), str(tmp), str(path)], check=True)
    tmp.unlink()


@pytest.mark.skipif(not _has_gdal(), reason="GDAL CLI required")
def test_world_window_ingest_equalizes_dims(tmp_path):
    src = tmp_path / "geoportal" / "poznan_mix"
    # 2 coarse years @0.25 m (800px=200m) and 1 fine year @0.05 m (4000px=200m),
    # fine grid offset +0.05 E, -0.15 S (the real Poznań translation). Sizes are large
    # enough for tiled/pyramided tiffsave (as on real gigapixel tiles).
    _write_geotiff(src / "2018" / "a.tif", 1000.0, 2000.0, 0.25, 800)
    _write_geotiff(src / "2020" / "b.tif", 1000.0, 2000.0, 0.25, 800)
    _write_geotiff(src / "2021" / "c.tif", 1000.05, 1999.85, 0.05, 4000)

    out_root = tmp_path / "out"
    reg = load_provider_registry()
    manifest = ingest_area_world_window(src, out_root, reg)

    assert manifest["provider"] == "geoportal" and manifest["epsg"] == 2180
    assert len(manifest["locations"]) == 1
    (key, loc), = manifest["locations"].items()
    years = {s["year"]: s for s in loc["seasons"] if not s.get("dropped")}
    assert set(years) == {2018, 2020, 2021}
    # all three co-registered to ONE equal-dimension stack at the coarse GSD
    dims = {tuple(s["dims"]) for s in years.values()}
    assert len(dims) == 1, dims
    assert years[2021]["downsampled"] is True
    assert years[2020]["downsampled"] is False
    # on disk: equal-dimension year tifs with sidecars
    cell = out_root / "geoportal" / "poznan_mix" / key
    files = sorted(p.name for p in cell.glob("year_*.tif"))
    assert files == ["year_2018.tif", "year_2020.tif", "year_2021.tif"]
    sizes = {(pyvips.Image.new_from_file(str(cell / f)).width,
              pyvips.Image.new_from_file(str(cell / f)).height) for f in files}
    assert len(sizes) == 1, sizes
    for y in (2018, 2020, 2021):
        assert (cell / f"year_{y}.tfw").exists() and (cell / f"year_{y}.prj").exists()
