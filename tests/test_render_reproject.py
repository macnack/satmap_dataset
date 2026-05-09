from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.pipeline import render


@pytest.fixture(autouse=True)
def _isolate_subprocess(monkeypatch):
    """Capture gdalwarp invocations and pretend gdalwarp succeeded."""
    captured: list[list[str]] = []

    def fake_run(cmd, *_args, **kwargs):
        captured.append(cmd)
        # Simulate gdalwarp creating the output file.
        out_path = Path(cmd[-1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"\x00" * 1024)

        class _Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Result()

    monkeypatch.setattr(render.subprocess, "run", fake_run)
    monkeypatch.setattr(render, "_gdalwarp_path", lambda: "/usr/bin/gdalwarp")
    return captured


def test_reproject_short_circuits_when_crs_matches(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "tile.tif"
    src.write_bytes(b"x")
    monkeypatch.setattr(render, "_read_source_epsg", lambda _p: 3006)

    out = render._reproject_asset_to_target_crs(
        src, target_srs="EPSG:3006", resample_method="bilinear"
    )

    assert out == src
    assert not (tmp_path / "_reprojected").exists()


def test_reproject_invokes_gdalwarp_when_crs_differs(
    tmp_path: Path, monkeypatch, _isolate_subprocess
) -> None:
    src = tmp_path / "scene.tif"
    src.write_bytes(b"x")
    monkeypatch.setattr(render, "_read_source_epsg", lambda _p: 32633)

    out = render._reproject_asset_to_target_crs(
        src, target_srs="EPSG:3006", resample_method="bilinear"
    )

    assert out.parent == tmp_path / "_reprojected"
    assert out.name.startswith("scene__epsg3006_")
    assert out.name.endswith(".tif")
    assert out.exists()
    assert len(_isolate_subprocess) == 1
    cmd = _isolate_subprocess[0]
    assert "gdalwarp" in cmd[0]
    assert "-t_srs" in cmd and "EPSG:3006" in cmd
    assert "-r" in cmd and "bilinear" in cmd
    assert cmd[-2] == str(src)
    assert cmd[-1] == str(out)
    # Full-extent warp (no target_bbox kwarg) should NOT pass -te / -tr.
    assert "-te" not in cmd
    assert "-tr" not in cmd


def test_reproject_uses_cached_output_on_repeat_call(
    tmp_path: Path, monkeypatch, _isolate_subprocess
) -> None:
    src = tmp_path / "scene.tif"
    src.write_bytes(b"x")
    monkeypatch.setattr(render, "_read_source_epsg", lambda _p: 32633)

    first = render._reproject_asset_to_target_crs(
        src, target_srs="EPSG:3006", resample_method="bilinear"
    )
    second = render._reproject_asset_to_target_crs(
        src, target_srs="EPSG:3006", resample_method="bilinear"
    )

    assert first == second
    assert len(_isolate_subprocess) == 1  # second call hit cache, no gdalwarp


def test_reproject_uses_nearest_resampling_when_requested(
    tmp_path: Path, monkeypatch, _isolate_subprocess
) -> None:
    src = tmp_path / "scene.tif"
    src.write_bytes(b"x")
    monkeypatch.setattr(render, "_read_source_epsg", lambda _p: 32633)

    render._reproject_asset_to_target_crs(
        src, target_srs="EPSG:3006", resample_method="nearest"
    )
    cmd = _isolate_subprocess[0]
    assert "near" in cmd  # gdalwarp's flag for nearest neighbor


def test_reproject_raises_when_gdalwarp_missing(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "scene.tif"
    src.write_bytes(b"x")
    monkeypatch.setattr(render, "_read_source_epsg", lambda _p: 32633)
    monkeypatch.setattr(render, "_gdalwarp_path", lambda: None)

    with pytest.raises(RuntimeError, match="Install GDAL"):
        render._reproject_asset_to_target_crs(
            src, target_srs="EPSG:3006", resample_method="bilinear"
        )


def test_reproject_grouped_assets_substitutes_paths(
    tmp_path: Path, monkeypatch, _isolate_subprocess
) -> None:
    a = tmp_path / "a.tif"; a.write_bytes(b"x")
    b = tmp_path / "b.tif"; b.write_bytes(b"x")
    monkeypatch.setattr(render, "_read_source_epsg", lambda p: 32633 if p == a else 3006)

    out = render._reproject_grouped_assets_to_target_crs(
        {2024: [a, b]},
        target_srs="EPSG:3006",
        resample_method="bilinear",
    )

    assert out[2024][0] != a  # reprojected
    assert out[2024][1] == b  # short-circuited


def test_reproject_with_target_bbox_uses_te_te_srs_tr(
    tmp_path: Path, monkeypatch, _isolate_subprocess
) -> None:
    src = tmp_path / "scene.tif"
    src.write_bytes(b"x")
    monkeypatch.setattr(render, "_read_source_epsg", lambda _p: 32633)
    bbox = render.BBox(min_x=536194.91, min_y=6426213.28, max_x=538194.91, max_y=6428213.28)

    out = render._reproject_asset_to_target_crs(
        src,
        target_srs="EPSG:3006",
        resample_method="bilinear",
        target_bbox=bbox,
        target_pixel_size=10.0,
        margin_meters=500.0,
    )

    assert out.exists()
    cmd = _isolate_subprocess[0]
    # -te <minx miny maxx maxy> with margin baked in.
    assert "-te" in cmd
    te_index = cmd.index("-te")
    minx, miny, maxx, maxy = (float(cmd[te_index + i]) for i in range(1, 5))
    assert minx == pytest.approx(bbox.min_x - 500.0, abs=1e-3)
    assert miny == pytest.approx(bbox.min_y - 500.0, abs=1e-3)
    assert maxx == pytest.approx(bbox.max_x + 500.0, abs=1e-3)
    assert maxy == pytest.approx(bbox.max_y + 500.0, abs=1e-3)
    # -te_srs == target CRS (so we describe the bbox in target_srs).
    te_srs_index = cmd.index("-te_srs")
    assert cmd[te_srs_index + 1] == "EPSG:3006"
    # -tr <px> <px> at the requested pixel size.
    tr_index = cmd.index("-tr")
    assert float(cmd[tr_index + 1]) == 10.0
    assert float(cmd[tr_index + 2]) == 10.0


def test_reproject_cache_key_differs_per_bbox_and_pixel_size(
    tmp_path: Path, monkeypatch, _isolate_subprocess
) -> None:
    src = tmp_path / "scene.tif"
    src.write_bytes(b"x")
    monkeypatch.setattr(render, "_read_source_epsg", lambda _p: 32633)
    bbox_a = render.BBox(min_x=0, min_y=0, max_x=2000, max_y=2000)
    bbox_b = render.BBox(min_x=10000, min_y=10000, max_x=12000, max_y=12000)

    a = render._reproject_asset_to_target_crs(
        src, target_srs="EPSG:3006", resample_method="bilinear",
        target_bbox=bbox_a, target_pixel_size=10.0,
    )
    b = render._reproject_asset_to_target_crs(
        src, target_srs="EPSG:3006", resample_method="bilinear",
        target_bbox=bbox_b, target_pixel_size=10.0,
    )
    c = render._reproject_asset_to_target_crs(
        src, target_srs="EPSG:3006", resample_method="bilinear",
        target_bbox=bbox_a, target_pixel_size=2.0,
    )

    # All three cache keys must be distinct: bbox change ↔ pixel-size change.
    assert len({a.name, b.name, c.name}) == 3
    assert len(_isolate_subprocess) == 3
