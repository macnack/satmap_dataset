import zipfile

import pytest
from pathlib import Path

from satmap_dataset.config import DemConfig
from satmap_dataset.models import DemManifest, YearStatus
from satmap_dataset.pipeline import dem_skorowidz


def _patch(monkeypatch, *, tiles_by_year):
    """tiles_by_year: {year: {godlo: url}} ; empty dict -> skipped year."""

    async def _fake_year_typenames(product, datum, options=None, *, timeout=45.0, retry_policy=None):
        return {y: f"gugik:SkorowidzNMT{y}" for y in tiles_by_year}

    async def _fake_tiles_for_year(product, datum, year, bbox, srs, *, year_to_typename, options=None, timeout=45.0, retry_policy=None):
        tiles = tiles_by_year[year]
        status = YearStatus(year=year, typename_exists=True, feature_count=len(tiles),
                            status="has_features" if tiles else "zero_features")
        return status, dict(tiles), {}, {}

    async def _fake_download(urls, dest_dir, config):
        out = []
        for i, _u in enumerate(urls):
            p = Path(dest_dir) / f"t{i}.asc"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("ncols 1\n")
            out.append(p)
        return out

    def _fake_mosaic(tiles, out_path, bbox):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"NATIVE")

    def _fake_align(native, out_path, **kwargs):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"ALIGNED")

    monkeypatch.setattr(dem_skorowidz.dem_skorowidz_client, "year_typenames", _fake_year_typenames)
    monkeypatch.setattr(dem_skorowidz.dem_skorowidz_client, "tiles_for_year", _fake_tiles_for_year)
    monkeypatch.setattr(dem_skorowidz, "_download_tiles", _fake_download)
    monkeypatch.setattr(dem_skorowidz, "_mosaic_asc_to_native", _fake_mosaic)
    monkeypatch.setattr(dem_skorowidz.dem, "_align_to_grid", _fake_align)
    monkeypatch.setattr(dem_skorowidz.dem, "_raster_dims", lambda path: (10, 10))


def _cfg(tmp_path, **kw):
    base = dict(
        bbox="0,0,100,100", transport="skorowidz", year_start=2012, year_end=2019,
        products=["nmt"], vertical_datum="kron86", align_to_render=False,
        dem_root=tmp_path / "dem_x", output_json=tmp_path / "dem_x" / "dem_manifest.json",
    )
    base.update(kw)
    return DemConfig(**base)


def test_run_year_keyed_native(tmp_path, monkeypatch):
    _patch(monkeypatch, tiles_by_year={2012: {"g1": "u1", "g2": "u2"}, 2019: {"g3": "u3"}})
    code, path = dem_skorowidz.run(_cfg(tmp_path))
    assert code == 0
    m = DemManifest.model_validate_json(Path(path).read_text())
    assert m.transport == "skorowidz"
    nmt = m.products[0]
    years = {y.year: y for y in nmt.years}
    assert set(years) == {2012, 2019}
    assert years[2012].tile_count == 2 and years[2012].passed
    assert years[2012].godla == ["g1", "g2"]
    assert Path(years[2012].native_path).exists()
    assert years[2019].aligned_path is None


def test_run_skips_empty_year(tmp_path, monkeypatch):
    _patch(monkeypatch, tiles_by_year={2012: {"g1": "u1"}, 2015: {}})
    code, path = dem_skorowidz.run(_cfg(tmp_path))
    assert code == 0
    m = DemManifest.model_validate_json(Path(path).read_text())
    assert 2015 in m.years_skipped
    assert {y.year for y in m.products[0].years} == {2012}


def test_run_aligns_when_enabled(tmp_path, monkeypatch):
    _patch(monkeypatch, tiles_by_year={2012: {"g1": "u1"}})
    cfg = _cfg(tmp_path, align_to_render=True, target_bbox="0,0,100,100", target_width=100, target_height=100)
    code, path = dem_skorowidz.run(cfg)
    assert code == 0
    m = DemManifest.model_validate_json(Path(path).read_text())
    y = m.products[0].years[0]
    assert Path(y.aligned_path).exists() and y.aligned_width == 100


def test_run_fails_when_no_years_available(tmp_path, monkeypatch):
    _patch(monkeypatch, tiles_by_year={})
    code, path = dem_skorowidz.run(_cfg(tmp_path, year_start=2012, year_end=2012))
    assert code == 1
    m = DemManifest.model_validate_json(Path(path).read_text())
    assert m.passed is False


def test_run_reuses_existing_native_without_download(tmp_path, monkeypatch):
    _patch(monkeypatch, tiles_by_year={2012: {"g1": "u1"}})
    calls = {"download": 0, "mosaic": 0}

    async def _spy_download(urls, dest_dir, config):
        calls["download"] += 1
        return []

    def _spy_mosaic(tiles, out_path, bbox):
        calls["mosaic"] += 1

    monkeypatch.setattr(dem_skorowidz, "_download_tiles", _spy_download)
    monkeypatch.setattr(dem_skorowidz, "_mosaic_asc_to_native", _spy_mosaic)

    cfg = _cfg(tmp_path)
    native = cfg.dem_root / "skorowidz" / "nmt_kron86" / "native" / "year_2012.tif"
    native.parent.mkdir(parents=True, exist_ok=True)
    native.write_bytes(b"PREEXISTING")

    code, path = dem_skorowidz.run(cfg)
    assert code == 0
    assert calls["download"] == 0 and calls["mosaic"] == 0
    m = DemManifest.model_validate_json(Path(path).read_text())
    y = m.products[0].years[0]
    assert y.passed is True
    assert y.godla == ["g1"]  # godla set even on the reuse path


def test_run_capabilities_failure_isolates_product(tmp_path, monkeypatch):
    _patch(monkeypatch, tiles_by_year={2012: {"g1": "u1"}})

    async def _boom(product, datum, options=None, *, timeout=45.0, retry_policy=None):
        raise RuntimeError("capabilities down")

    monkeypatch.setattr(dem_skorowidz.dem_skorowidz_client, "year_typenames", _boom)

    code, path = dem_skorowidz.run(_cfg(tmp_path))
    assert code == 1  # no years produced
    m = DemManifest.model_validate_json(Path(path).read_text())
    assert m.products[0].years == []
    assert m.products[0].passed is False


def test_select_grid_urls_prefers_nonxyz_and_keeps_only_xyz():
    tiles = {
        "a": "https://x/70_85_M-34-27-C-c-4-4.asc",
        "a_xyz": "https://x/70_85_M-34-27-C-c-4-4.xyz",
        "b_xyz": "https://x/70_85_M-34-27-C-d-3-3.xyz",          # only xyz -> kept
        "c_zip": "https://x/72_88_M-34-27-C-e-1-1.zip",          # non-xyz zip
    }
    sel = dem_skorowidz._select_grid_urls(tiles)
    assert len(sel) == 3  # three distinct godla
    # for godlo c-4-4: .asc chosen, .xyz dropped
    assert "70_85_M-34-27-C-c-4-4.asc" in sel
    assert "70_85_M-34-27-C-c-4-4.xyz" not in sel
    # godlo d-3-3 has only xyz -> kept
    assert "70_85_M-34-27-C-d-3-3.xyz" in sel
    # zip kept
    assert "72_88_M-34-27-C-e-1-1.zip" in sel


def test_extract_if_zip_returns_inner_raster(tmp_path):
    z = tmp_path / "72_88_M-34-27-C-d-3-3.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("M-34-27-C-d-3-3.asc", "ncols 2\nnrows 2\n0 0 0 0\n")
        zf.writestr("readme.txt", "ignore me")
    out = dem_skorowidz._extract_if_zip(z, tmp_path)
    assert len(out) == 1 and out[0].name.endswith(".asc")
    assert out[0].exists()


def test_extract_if_zip_passthrough_nonzip(tmp_path):
    a = tmp_path / "x.asc"
    a.write_text("x")
    assert dem_skorowidz._extract_if_zip(a, tmp_path) == [a]


def test_extract_if_zip_no_raster_raises(tmp_path):
    z = tmp_path / "bad.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("notes.txt", "nothing")
    with pytest.raises(RuntimeError):
        dem_skorowidz._extract_if_zip(z, tmp_path)


def test_run_swaps_bbox_axes_for_epsg2180_wfs_query(tmp_path, monkeypatch):
    _patch(monkeypatch, tiles_by_year={2012: {"g1": "u1"}})
    captured = {}

    async def _capture(product, datum, year, bbox, srs, *, year_to_typename, options=None, timeout=45.0, retry_policy=None):
        captured["bbox"] = bbox
        status = YearStatus(year=year, typename_exists=True, feature_count=1, status="has_features")
        return status, {"g1": "u1"}, {}, {}

    monkeypatch.setattr(dem_skorowidz.dem_skorowidz_client, "tiles_for_year", _capture)
    cfg = _cfg(tmp_path, bbox="348967.353,508503.706,349967.353,509503.706")
    dem_skorowidz.run(cfg)
    # EPSG:2180 WFS expects (ymin,xmin,ymax,xmax)
    assert captured["bbox"] == "508503.706,348967.353,509503.706,349967.353"


def test_normalize_xyz_sorts_to_row_major(tmp_path):
    raw = tmp_path / "t.xyz"
    raw.write_text("0 0 1\n1 1 2\n0 1 3\n1 0 4\n")  # serpentine / unsorted (X Y Z)
    out = dem_skorowidz._normalize_xyz(raw, tmp_path)
    lines = [ln.split() for ln in out.read_text().splitlines() if ln.strip()]
    ys = [float(ln[1]) for ln in lines]
    assert ys == sorted(ys, reverse=True)  # Y descending (top-to-bottom rows)
    # within the first row (Y=1), X ascending
    assert lines[0][1] == "1" and lines[1][1] == "1"
    assert float(lines[0][0]) <= float(lines[1][0])


def test_normalize_xyz_passthrough_nonxyz(tmp_path):
    a = tmp_path / "x.asc"
    a.write_text("x")
    assert dem_skorowidz._normalize_xyz(a, tmp_path) == a
