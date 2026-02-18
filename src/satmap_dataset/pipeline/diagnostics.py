from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import tifffile


@dataclass(frozen=True)
class BBox:
    min_x: float
    min_y: float
    max_x: float
    max_y: float


def parse_bbox(value: str) -> BBox:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must have 4 numeric values")
    min_x, min_y, max_x, max_y = parts
    if min_x >= max_x or min_y >= max_y:
        raise ValueError("bbox must satisfy min_x<max_x and min_y<max_y")
    return BBox(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y)


def _read_georef_bbox(path: Path) -> BBox | None:
    try:
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            scale_tag = page.tags.get("ModelPixelScaleTag")
            tie_tag = page.tags.get("ModelTiepointTag")
            if scale_tag is None or tie_tag is None:
                return None
            scale = scale_tag.value
            tie = tie_tag.value
            if len(scale) < 2 or len(tie) < 6:
                return None
            px_x = float(scale[0])
            px_y = abs(float(scale[1]))
            origin_x = float(tie[3])
            origin_y = float(tie[4])
            width = int(page.imagewidth)
            height = int(page.imagelength)
        return BBox(
            min_x=origin_x,
            min_y=origin_y - height * px_y,
            max_x=origin_x + width * px_x,
            max_y=origin_y,
        )
    except Exception:
        return None


def _bbox_to_dict(bbox: BBox | None) -> dict[str, float] | None:
    if bbox is None:
        return None
    return {
        "xmin": float(bbox.min_x),
        "ymin": float(bbox.min_y),
        "xmax": float(bbox.max_x),
        "ymax": float(bbox.max_y),
    }


def _bbox_union(boxes: list[BBox]) -> BBox | None:
    if not boxes:
        return None
    return BBox(
        min_x=min(box.min_x for box in boxes),
        min_y=min(box.min_y for box in boxes),
        max_x=max(box.max_x for box in boxes),
        max_y=max(box.max_y for box in boxes),
    )


def _ensure_rgb_u8(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.shape[2] > 3:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _write_quicklook(input_path: Path, output_path: Path, max_dim: int = 1024) -> bool:
    try:
        import pyvips

        image = pyvips.Image.new_from_file(str(input_path), access="sequential")
        if image.bands == 1:
            image = image.bandjoin([image, image])
        elif image.bands >= 4:
            image = image.extract_band(0, n=3)
        elif image.bands == 2:
            b = image.extract_band(0)
            image = b.bandjoin([b, b])

        if image.format != "uchar":
            image = image.cast("uchar")

        scale = min(1.0, float(max_dim) / float(max(image.width, image.height)))
        if scale < 1.0:
            image = image.resize(scale, kernel="linear")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.tiffsave(
            str(output_path),
            compression="deflate",
            tile=True,
            tile_width=256,
            tile_height=256,
            pyramid=False,
            bigtiff=False,
        )
        return True
    except Exception:
        try:
            with tifffile.TiffFile(input_path) as tif:
                arr = tif.pages[0].asarray()
            arr = _ensure_rgb_u8(arr)
            h, w = arr.shape[:2]
            scale = min(1.0, float(max_dim) / float(max(h, w)))
            if scale < 1.0:
                out_h = max(1, int(round(h * scale)))
                out_w = max(1, int(round(w * scale)))
                y_idx = np.linspace(0, h - 1, out_h).astype(np.int32)
                x_idx = np.linspace(0, w - 1, out_w).astype(np.int32)
                arr = arr[np.ix_(y_idx, x_idx)]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(output_path, arr, photometric="rgb", compression="deflate")
            return True
        except Exception:
            return False


def _quicklook_diff(a_path: Path, b_path: Path, diff_path: Path) -> bool:
    try:
        with tifffile.TiffFile(a_path) as ta:
            a = _ensure_rgb_u8(ta.pages[0].asarray())
        with tifffile.TiffFile(b_path) as tb:
            b = _ensure_rgb_u8(tb.pages[0].asarray())

        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        if h <= 0 or w <= 0:
            return False

        a = a[:h, :w, :]
        b = b[:h, :w, :]
        diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.uint8)
        diff_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(diff_path, diff, photometric="rgb", compression="deflate")
        return True
    except Exception:
        return False


def build_mismatch_report(
    *,
    output_json_path: Path,
    years: list[int],
    rendered_assets: dict[int, Path],
    source_assets_by_year: dict[int, list[Path]],
    years_source_map: dict[int, str],
    target_bbox: str,
    target_width: int,
    target_height: int,
) -> tuple[Path, Path]:
    bbox = parse_bbox(target_bbox)
    px_x = (bbox.max_x - bbox.min_x) / float(max(1, target_width))
    px_y = (bbox.max_y - bbox.min_y) / float(max(1, target_height))

    quicklook_dir = output_json_path.parent / "diagnostic_quicklooks"
    quicklook_dir.mkdir(parents=True, exist_ok=True)

    years_payload: list[dict[str, object]] = []

    for year in years:
        output_asset = rendered_assets.get(year)
        source_assets = source_assets_by_year.get(year, [])
        source_type = years_source_map.get(year, "wfs")

        output_bbox = _read_georef_bbox(output_asset) if output_asset else None
        source_boxes = [box for box in (_read_georef_bbox(path) for path in source_assets) if box is not None]
        source_union = _bbox_union(source_boxes)

        shift_x = None
        shift_y = None
        if source_union is not None:
            source_cx = (source_union.min_x + source_union.max_x) / 2.0
            source_cy = (source_union.min_y + source_union.max_y) / 2.0
            target_cx = (bbox.min_x + bbox.max_x) / 2.0
            target_cy = (bbox.min_y + bbox.max_y) / 2.0
            shift_x = (source_cx - target_cx) / px_x
            shift_y = (target_cy - source_cy) / px_y

        output_quicklook = None
        if output_asset is not None:
            out_preview = quicklook_dir / f"year_{year}_output.tiff"
            if _write_quicklook(output_asset, out_preview):
                output_quicklook = str(out_preview)

        years_payload.append(
            {
                "year": year,
                "source": source_type,
                "input_bbox": _bbox_to_dict(bbox),
                "output_asset": str(output_asset) if output_asset is not None else None,
                "output_extent": _bbox_to_dict(output_bbox),
                "source_extents": [_bbox_to_dict(box) for box in source_boxes],
                "source_union_extent": _bbox_to_dict(source_union),
                "pixel_shift_x": shift_x,
                "pixel_shift_y": shift_y,
                "quicklook_output": output_quicklook,
            }
        )

    payload = {
        "kind": "wfs_wms_mismatch_report",
        "target_bbox": _bbox_to_dict(bbox),
        "target_width": target_width,
        "target_height": target_height,
        "years": years_payload,
    }
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_json_path, quicklook_dir
