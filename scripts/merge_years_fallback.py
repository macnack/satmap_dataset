#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.models import DatasetManifest
from satmap_dataset.pipeline import render

Image.MAX_IMAGE_PIXELS = None

LOG = logging.getLogger("merge_fallback")


def _read_manifest(path: Path) -> DatasetManifest:
    return DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))


def _as_rgb_u8(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.shape[2] > 3:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _read_tile_rgb(path: Path) -> np.ndarray:
    try:
        with tifffile.TiffFile(path) as tif:
            arr = tif.pages[0].asarray()
    except Exception:
        with Image.open(path) as img:
            arr = np.asarray(img)
    return _as_rgb_u8(arr)


def _resolve_asset_path(asset: str, dataset_manifest_path: Path) -> Path:
    p = Path(asset)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    return (dataset_manifest_path.parent / p).resolve()


def _extract_year(asset_path: Path) -> int:
    parent = asset_path.parent.name
    if parent.isdigit() and len(parent) == 4:
        return int(parent)
    for part in reversed(asset_path.parts):
        if part.isdigit() and len(part) == 4:
            return int(part)
    raise ValueError(f"Cannot infer year from asset path: {asset_path}")


def _group_assets_by_year(assets: list[str], dataset_manifest_path: Path) -> dict[int, list[Path]]:
    grouped: dict[int, list[Path]] = {}
    for asset in assets:
        asset_path = _resolve_asset_path(asset, dataset_manifest_path)
        year = _extract_year(asset_path)
        grouped.setdefault(year, []).append(asset_path)
    for year in grouped:
        grouped[year] = sorted(grouped[year])
    return grouped


def _resize_rgb(tile: np.ndarray, width: int, height: int, resample: Image.Resampling) -> np.ndarray:
    if tile.shape[1] == width and tile.shape[0] == height:
        return tile
    img = Image.fromarray(tile, mode="RGB")
    img = img.resize((width, height), resample=resample)
    return np.asarray(img, dtype=np.uint8)


def _render_year_fallback(
    *,
    year: int,
    assets: list[Path],
    out_path: Path,
    target_bbox: render.BBox,
    target_width: int,
    target_height: int,
    target_srs: str,
    source_axis_mode: str,
    tile_size: int,
) -> dict[str, float]:
    raw_path = out_path.with_suffix(".raw.memmap")
    if raw_path.exists():
        raw_path.unlink()

    canvas = np.memmap(raw_path, dtype=np.uint8, mode="w+", shape=(target_height, target_width, 3))
    canvas[:] = 0

    covered_area = 0.0
    bbox_area = max(1e-9, (target_bbox.max_x - target_bbox.min_x) * (target_bbox.max_y - target_bbox.min_y))
    bilinear = Image.Resampling.BILINEAR
    processed = 0

    for asset in assets:
        georef = render._read_georef(asset)
        src_bbox = (
            render._bbox_from_swapped_axis_georef(georef)
            if source_axis_mode == "swapped"
            else render._bbox_from_georef(georef)
        )
        inter = render._intersection(src_bbox, target_bbox)
        if inter is None:
            continue
        covered_area += (inter.max_x - inter.min_x) * (inter.max_y - inter.min_y)

        if source_axis_mode == "swapped":
            sx0, sy0, sx1, sy1 = render._to_source_pixels_swapped(inter, georef)
        else:
            sx0, sy0, sx1, sy1 = render._to_source_pixels(inter, georef)
        dx0, dy0, dx1, dy1 = render._to_target_pixels(inter, target_bbox, target_width, target_height)

        src_w = sx1 - sx0
        src_h = sy1 - sy0
        dst_w = dx1 - dx0
        dst_h = dy1 - dy0
        if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
            continue

        tile = _read_tile_rgb(asset)
        crop = tile[sy0:sy1, sx0:sx1, :]
        if crop.size == 0:
            continue
        if source_axis_mode == "swapped":
            crop = np.transpose(crop, (1, 0, 2))[::-1, ::-1, :].copy()
        crop = _resize_rgb(crop, dst_w, dst_h, bilinear)
        canvas[dy0:dy1, dx0:dx1, :] = crop
        processed += 1

    canvas.flush()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        out_path,
        canvas,
        photometric="rgb",
        compression="deflate",
        tile=(tile_size, tile_size),
        bigtiff=True,
        metadata=None,
    )
    mean_rgb = [float(canvas[:, :, i].mean()) for i in range(3)]
    del canvas
    raw_path.unlink(missing_ok=True)
    render._ensure_georeferenced_output(out_path, target_bbox, target_width, target_height, target_srs)
    return {
        "coverage_ratio": float(min(1.0, covered_area / bbox_area)),
        "tiles_used": float(processed),
        "mean_r": mean_rgb[0],
        "mean_g": mean_rgb[1],
        "mean_b": mean_rgb[2],
    }


def run(
    *,
    dataset_manifest: Path,
    output_dir: Path,
    output_manifest: Path,
    target_bbox_str: str | None,
    target_srs: str,
    px_per_meter: float,
    tile_size: int,
) -> Path:
    source_manifest = _read_manifest(dataset_manifest)
    if not source_manifest.years_included:
        raise ValueError("No years_included in dataset manifest.")

    target_bbox = render._parse_bbox(target_bbox_str or source_manifest.target_bbox or "")
    target_width, target_height = render._compute_reference_dimensions(target_bbox, px_per_meter)
    years_source_map = dict(source_manifest.years_source_map)
    grouped_assets = _group_assets_by_year(source_manifest.assets, dataset_manifest)

    global_wfs_axis_mode = render._infer_global_wfs_axis_mode(
        grouped_assets=grouped_assets,
        years_source_map=years_source_map,
        target_bbox=target_bbox,
        reference_year=None,
    )
    LOG.info(
        "Merge: target=%sx%s bbox=%s,%s,%s,%s global_wfs_axis_mode=%s",
        target_width,
        target_height,
        target_bbox.min_x,
        target_bbox.min_y,
        target_bbox.max_x,
        target_bbox.max_y,
        global_wfs_axis_mode,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    merged_assets: list[str] = []
    years_report: list[dict[str, object]] = []

    for year in source_manifest.years_included:
        assets = grouped_assets.get(year, [])
        if not assets:
            years_report.append({"year": year, "status": "missing_assets"})
            continue
        source_type = years_source_map.get(year, "wfs")
        axis_mode = global_wfs_axis_mode if source_type == "wfs" else "normal"
        out_path = output_dir / f"year_{year}.tiff"
        LOG.info(
            "Merge: year=%s source=%s tiles=%s axis_mode=%s -> %s",
            year,
            source_type,
            len(assets),
            axis_mode,
            out_path,
        )
        stats = _render_year_fallback(
            year=year,
            assets=assets,
            out_path=out_path,
            target_bbox=target_bbox,
            target_width=target_width,
            target_height=target_height,
            target_srs=target_srs,
            source_axis_mode=axis_mode,
            tile_size=tile_size,
        )
        merged_assets.append(str(out_path))
        years_report.append(
            {
                "year": year,
                "source": source_type,
                "axis_mode": axis_mode,
                "tiles": len(assets),
                "output": str(out_path),
                "stats": stats,
            }
        )

    payload = {
        "kind": "fallback_merge_manifest",
        "source_manifest": str(dataset_manifest),
        "target_bbox": [target_bbox.min_x, target_bbox.min_y, target_bbox.max_x, target_bbox.max_y],
        "target_srs": target_srs,
        "target_width": target_width,
        "target_height": target_height,
        "px_per_meter": px_per_meter,
        "years_requested": source_manifest.years_requested,
        "years_included": source_manifest.years_included,
        "years_source_map": {str(k): v for k, v in years_source_map.items()},
        "assets": merged_assets,
        "years": years_report,
        "wfs_axis_mode": global_wfs_axis_mode,
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOG.info("Merge: wrote manifest %s", output_manifest)
    return output_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge yearly ortho maps without pyvips.")
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--target-bbox", default=None, help="xmin,ymin,xmax,ymax (optional)")
    parser.add_argument("--target-srs", default="EPSG:2180")
    parser.add_argument("--px-per-meter", type=float, default=15.0)
    parser.add_argument("--tile-size", type=int, default=512)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run(
        dataset_manifest=args.dataset_manifest,
        output_dir=args.output_dir,
        output_manifest=args.output_manifest,
        target_bbox_str=args.target_bbox,
        target_srs=args.target_srs,
        px_per_meter=args.px_per_meter,
        tile_size=args.tile_size,
    )


if __name__ == "__main__":
    main()
