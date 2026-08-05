"""The NMPT WCS answers FORMAT=image/tiff with a 3-band Byte raster (elevation
rounded to 1 m) instead of the single-band Float32 the DEM_F32 pixel profile
promises. The pipeline normalises that shape so consumers can trust the profile.
"""
from pathlib import Path

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

from satmap_dataset.pipeline.dem import _normalise_elevation_raster


def _write_tiff(path: Path, array: np.ndarray) -> None:
    tifffile.imwrite(str(path), array)


def test_multiband_byte_response_is_collapsed_to_single_float32_band(tmp_path):
    base = np.arange(64, dtype=np.uint8).reshape(8, 8) + 130
    rgb = np.stack([base, base, base], axis=-1)
    path = tmp_path / "nmpt_kron86.tif"
    _write_tiff(path, rgb)

    warning = _normalise_elevation_raster(path)

    out = np.asarray(tifffile.imread(str(path)))
    assert out.ndim == 2, "expected a single elevation band"
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, base.astype(np.float32))
    assert warning is not None
    assert "1 m" in warning


def test_single_band_float_raster_is_left_alone(tmp_path):
    arr = (np.arange(64, dtype=np.float32).reshape(8, 8) + 130.25)
    path = tmp_path / "nmt_kron86.tif"
    _write_tiff(path, arr)

    assert _normalise_elevation_raster(path) is None

    out = np.asarray(tifffile.imread(str(path)))
    np.testing.assert_array_equal(out, arr)


def test_multiband_raster_with_differing_bands_is_not_collapsed(tmp_path):
    a = np.full((8, 8), 130, dtype=np.uint8)
    rgb = np.stack([a, a + 1, a], axis=-1)
    path = tmp_path / "odd.tif"
    _write_tiff(path, rgb)

    warning = _normalise_elevation_raster(path)

    out = np.asarray(tifffile.imread(str(path)))
    assert out.shape == rgb.shape, "a genuine RGB raster must be left untouched"
    assert warning is not None
    assert "bands differ" in warning
