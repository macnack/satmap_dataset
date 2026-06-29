import math

import numpy as np
import pyvips
import pytest

from satmap_dataset.raw_tiles.core import (
    Cell,
    GeoTransform,
    TileInfo,
    _epsg_from_wkt,
    cell_key,
    derive_cell_grid,
    detect_year,
    geotransform_to_tfw_lines,
    load_provider_registry,
    min_coverage_for_epsg,
    provider_for_epsg,
    resolve_season_tile,
    tile_covers_cell,
    valid_pixel_fraction,
    world_window_to_pixel,
    write_prj_wkt,
    write_tfw,
    _YEAR_TOKEN_RE,
)


def test_geotransform_to_tfw_lines_corner_to_center():
    gt = GeoTransform(ulx=383394.0, xres=0.5, xskew=0.0, uly=6674897.5, yskew=0.0, yres=-0.5)
    a, d, b, e, c, f = geotransform_to_tfw_lines(gt)
    assert (a, d, b, e) == (0.5, 0.0, 0.0, -0.5)
    assert math.isclose(c, 383394.25, abs_tol=1e-9)
    assert math.isclose(f, 6674897.25, abs_tol=1e-9)


def test_write_tfw_reads_back_six_lines(tmp_path):
    gt = GeoTransform(717500.0, 0.16, 0.0, 7537500.0, 0.0, -0.16)
    p = tmp_path / "year_2025.tfw"
    write_tfw(gt, p)
    vals = [float(x) for x in p.read_text().splitlines()]
    assert len(vals) == 6
    assert vals[0] == pytest.approx(0.16) and vals[3] == pytest.approx(-0.16)
    assert vals[4] == pytest.approx(717500.08) and vals[5] == pytest.approx(7537499.92)


def test_epsg_from_wkt_top_level_authority_only():
    wkt = ('PROJCRS["x", BASEGEOGCRS["b", DATUM["d", ID["EPSG",6326]]], '
           'CONVERSION["c", METHOD["m", ID["EPSG",9807]]], CS["xy", ID["EPSG",9001]], '
           'ID["EPSG",2180]]')
    assert _epsg_from_wkt(wkt) == 2180
    generic = 'PROJCRS["Transverse Mercator; WGS84", CS["xy", AXIS["e", ID["EPSG",9001]]]]'
    assert _epsg_from_wkt(generic) is None


def test_write_prj_wkt_verbatim(tmp_path):
    p = tmp_path / "year_2017.prj"
    write_prj_wkt('PROJCS["x"]', p)
    assert p.read_text() == 'PROJCS["x"]'


def test_registry_maps_three_providers():
    reg = load_provider_registry()
    assert provider_for_epsg(2180, reg) == "geoportal"
    assert provider_for_epsg(3006, reg) == "lantmateriet"
    assert provider_for_epsg(3067, reg) == "nls"


def test_provider_for_unknown_epsg_raises_helpful():
    with pytest.raises(KeyError, match="9999"):
        provider_for_epsg(9999, {2180: {"provider": "geoportal", "min_coverage": None}})


def test_min_coverage_per_provider_override():
    reg = load_provider_registry()
    assert min_coverage_for_epsg(2180, reg) == 0.5
    assert min_coverage_for_epsg(3006, reg) == 0.95
    assert min_coverage_for_epsg(3067, reg, default=0.9) == 0.9


def test_detect_year_prefers_parent_dir(tmp_path):
    d = tmp_path / "2014"
    d.mkdir()
    f = d / "nls_2020_0_0.tif"
    f.write_bytes(b"")
    assert detect_year(f) == 2014


def test_detect_year_filename_token_and_coordinate_guard(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    f = d / "nls_2014_0_0.tif"
    f.write_bytes(b"")
    assert detect_year(f) == 2014
    assert _YEAR_TOKEN_RE.search("e717500_n7535000") is None
    assert _YEAR_TOKEN_RE.search("o75350_7175_25_fj08") is None


def _tile(ulx, uly, gsd, size_m, year=2008, ny=None):
    nx = round(size_m / gsd)
    return TileInfo(None, nx, ny if ny is not None else nx,
                    GeoTransform(ulx, gsd, 0.0, uly, 0.0, -gsd), 3006, "wkt", year)


def test_cell_key_sw_corner():
    assert cell_key(Cell(717500, 7537500, 2500, 2500)) == "e717500_n7535000"


def test_derive_cell_grid_uses_smallest_footprint():
    fine = _tile(717500, 7537500, 0.25, 2500)
    coarse = _tile(715000, 7540000, 0.50, 5000)
    cells = derive_cell_grid([fine, coarse])
    assert all(c.w_m == 2500 and c.h_m == 2500 for c in cells)
    assert Cell(717500, 7537500, 2500, 2500) in cells


def test_tile_covers_and_window():
    coarse = _tile(715000, 7540000, 0.50, 5000)
    cell = Cell(717500, 7537500, 2500, 2500)
    assert tile_covers_cell(coarse, cell)
    assert world_window_to_pixel(coarse.gt, 717500, 7537500, 2500, 2500) == (5000, 5000, 5000, 5000)


def test_derive_cell_grid_nonsquare_tile_covers_its_own_cell():
    gt = GeoTransform(383394.0, 0.5, 0.0, 6674897.5, 0.0, -0.5)
    t = TileInfo(None, 3200, 2987, gt, 3067, "wkt", 2014)
    cells = derive_cell_grid([t])
    assert len(cells) == 1
    c = cells[0]
    assert c.w_m == pytest.approx(1600.0) and c.h_m == pytest.approx(1493.5)
    assert tile_covers_cell(t, c)
    assert cell_key(c) == "e383394_n6673404"


def _img(arr):
    h, w, c = arr.shape
    return pyvips.Image.new_from_memory(arr.tobytes(), w, h, c, "uchar")


def test_valid_pixel_fraction_half_nodata():
    arr = np.zeros((10, 20, 3), np.uint8)
    arr[:, :10, :] = 200
    assert valid_pixel_fraction(_img(arr)) == pytest.approx(0.5, abs=1e-6)


def test_resolve_prefers_finest_gsd():
    cell = Cell(717500, 7537500, 2500, 2500)
    fine = _tile(717500, 7537500, 0.16, 2500, year=2019)
    coarse = _tile(715000, 7540000, 0.50, 5000, year=2019)
    assert resolve_season_tile(cell, [coarse, fine]) is fine
    assert resolve_season_tile(cell, [coarse]) is coarse
    assert resolve_season_tile(Cell(720000, 7532500, 2500, 2500), [fine]) is None
