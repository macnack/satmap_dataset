from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.pipeline import render


def test_bbox_parse_roundtrip() -> None:
    bbox = render._parse_bbox("210300,521900,210500,522100")
    assert bbox.min_x == 210300
    assert bbox.min_y == 521900
    assert bbox.max_x == 210500
    assert bbox.max_y == 522100


def test_target_pixel_mapping() -> None:
    target = render.BBox(min_x=0, min_y=0, max_x=100, max_y=100)
    bounds = render.BBox(min_x=25, min_y=25, max_x=75, max_y=75)
    x0, y0, x1, y1 = render._to_target_pixels(bounds, target, 1000, 1000)
    assert (x0, y0, x1, y1) == (250, 250, 750, 750)


def test_source_pixel_mapping() -> None:
    src = render.GeoRef(
        origin_x=10,
        origin_y=110,
        pixel_size_x=1,
        pixel_size_y=1,
        width=100,
        height=100,
    )
    bounds = render.BBox(min_x=20, min_y=30, max_x=40, max_y=60)
    sx0, sy0, sx1, sy1 = render._to_source_pixels(bounds, src)
    assert (sx0, sy0, sx1, sy1) == (10, 50, 30, 80)


def test_group_assets_is_deterministic(tmp_path: Path) -> None:
    a = tmp_path / "downloads" / "2019" / "b.tif"
    b = tmp_path / "downloads" / "2019" / "a.tif"
    c = tmp_path / "downloads" / "2021" / "z.tif"
    for p in (a, b, c):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")

    grouped = render._group_assets_by_year(
        assets=[str(a), str(c), str(b)],
        dataset_manifest_path=tmp_path / "artifacts" / "dataset_manifest_download.json",
    )
    assert [p.name for p in grouped[2019]] == ["a.tif", "b.tif"]
    assert [p.name for p in grouped[2021]] == ["z.tif"]


def test_reference_dimensions_from_bbox() -> None:
    bbox = render.BBox(min_x=210300.0, min_y=521900.0, max_x=210500.0, max_y=522100.0)
    width, height = render._compute_reference_dimensions(bbox, px_per_meter=15.0)
    assert width == 3000
    assert height == 3000


def test_reference_dimensions_from_2km_bbox() -> None:
    bbox = render.BBox(min_x=359700.0, min_y=504900.0, max_x=361700.0, max_y=506900.0)
    width, height = render._compute_reference_dimensions(bbox, px_per_meter=15.0)
    assert width == 30000
    assert height == 30000


def test_resolve_target_dimensions_auto_size() -> None:
    bbox = render.BBox(min_x=0.0, min_y=0.0, max_x=100.0, max_y=200.0)
    config = render.RenderConfig(
        dataset_manifest=Path("artifacts/dataset_manifest_download.json"),
        px_per_meter=10.0,
        auto_size_from_bbox=True,
    )
    width, height = render._resolve_target_dimensions(config, bbox)
    assert width == 1000
    assert height == 2000


def test_resolve_target_dimensions_override() -> None:
    bbox = render.BBox(min_x=0.0, min_y=0.0, max_x=100.0, max_y=200.0)
    config = render.RenderConfig(
        dataset_manifest=Path("artifacts/dataset_manifest_download.json"),
        px_per_meter=10.0,
        target_width=321,
        target_height=654,
        auto_size_from_bbox=True,
    )
    width, height = render._resolve_target_dimensions(config, bbox)
    assert (width, height) == (321, 654)


def test_translation_matrix_structure() -> None:
    matrix = render._translation_matrix(3.5, -2.0)
    assert matrix == (1.0, 0.0, 3.5, 0.0, 1.0, -2.0)
