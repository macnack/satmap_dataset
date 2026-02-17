from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import logging
import hashlib
import json
import math
import shutil
import subprocess
import tempfile

import tifffile
import numpy as np

from satmap_dataset.config import RenderConfig
from satmap_dataset.models import DatasetManifest, IndexManifest
from satmap_dataset.pipeline import diagnostics

logger = logging.getLogger("satmap_dataset.render")

EPSG_2180_WKT = (
    'PROJCS["ETRS89 / Poland CS92",GEOGCS["ETRS89",'
    'DATUM["European Terrestrial Reference System 1989",'
    'SPHEROID["GRS 1980",6378137,298.257222101]],'
    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
    'PROJECTION["Transverse_Mercator"],'
    'PARAMETER["latitude_of_origin",0],'
    'PARAMETER["central_meridian",19],'
    'PARAMETER["scale_factor",0.9993],'
    'PARAMETER["false_easting",500000],'
    'PARAMETER["false_northing",-5300000],UNIT["metre",1],'
    'AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","2180"]]'
)


@dataclass(frozen=True)
class GeoRef:
    origin_x: float
    origin_y: float
    pixel_size_x: float
    pixel_size_y: float
    width: int
    height: int

    @property
    def min_x(self) -> float:
        return self.origin_x

    @property
    def max_x(self) -> float:
        return self.origin_x + self.width * self.pixel_size_x

    @property
    def max_y(self) -> float:
        return self.origin_y

    @property
    def min_y(self) -> float:
        return self.origin_y - self.height * self.pixel_size_y


@dataclass(frozen=True)
class BBox:
    min_x: float
    min_y: float
    max_x: float
    max_y: float


@dataclass(frozen=True)
class Calibration:
    enabled: bool
    reference_year: int | None
    transform_type: str | None
    source_axis_mode: str
    matrix: tuple[float, ...] | None
    fit_error_px: float | None
    report_path: Path | None


def _bbox_from_georef(src: GeoRef) -> BBox:
    return BBox(min_x=src.min_x, min_y=src.min_y, max_x=src.max_x, max_y=src.max_y)


def _bbox_from_swapped_axis_georef(src: GeoRef) -> BBox:
    return BBox(
        min_x=src.origin_y - src.height * src.pixel_size_y,
        min_y=src.origin_x,
        max_x=src.origin_y,
        max_y=src.origin_x + src.width * src.pixel_size_x,
    )


def _parse_bbox(bbox: str) -> BBox:
    parts = [float(part.strip()) for part in bbox.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must contain 4 numeric values")
    min_x, min_y, max_x, max_y = parts
    if min_x >= max_x or min_y >= max_y:
        raise ValueError("bbox must satisfy min_x<max_x and min_y<max_y")
    return BBox(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y)


def _read_dataset_manifest(path: Path) -> DatasetManifest:
    return DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))


def _read_index_manifest(path: Path) -> IndexManifest:
    return IndexManifest.model_validate_json(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: DatasetManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")


def _resolve_manifest_path(reference: str, dataset_manifest_path: Path) -> Path:
    candidate = Path(reference)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    return (dataset_manifest_path.parent / candidate).resolve()


def _resolve_target_bbox(config: RenderConfig, source_manifest: DatasetManifest, dataset_manifest_path: Path) -> BBox:
    if config.target_bbox is not None:
        return _parse_bbox(config.target_bbox)
    if source_manifest.target_bbox is not None:
        return _parse_bbox(source_manifest.target_bbox)

    if not source_manifest.source_manifest:
        raise ValueError("target_bbox is required when source index manifest is unavailable")

    index_manifest_path = _resolve_manifest_path(source_manifest.source_manifest, dataset_manifest_path)
    index_manifest = _read_index_manifest(index_manifest_path)
    return _parse_bbox(index_manifest.bbox)


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


def _group_assets_by_year(assets: Iterable[str], dataset_manifest_path: Path) -> dict[int, list[Path]]:
    grouped: dict[int, list[Path]] = {}
    for asset in assets:
        asset_path = _resolve_asset_path(asset, dataset_manifest_path)
        year = _extract_year(asset_path)
        grouped.setdefault(year, []).append(asset_path)

    for year in grouped:
        grouped[year] = sorted(grouped[year])
    return grouped


def _compute_reference_dimensions(target_bbox: BBox, px_per_meter: float) -> tuple[int, int]:
    width = int(round((target_bbox.max_x - target_bbox.min_x) * px_per_meter))
    height = int(round((target_bbox.max_y - target_bbox.min_y) * px_per_meter))
    return max(1, width), max(1, height)


def _resolve_target_dimensions(config: RenderConfig, target_bbox: BBox) -> tuple[int, int]:
    if config.target_width is not None and config.target_height is not None:
        return config.target_width, config.target_height
    if not config.auto_size_from_bbox:
        raise ValueError("target_width and target_height are required when auto_size_from_bbox is disabled")
    return _compute_reference_dimensions(target_bbox, config.px_per_meter)


def _read_georef(path: Path) -> GeoRef:
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        scale_tag = page.tags.get("ModelPixelScaleTag")
        tie_tag = page.tags.get("ModelTiepointTag")
        if scale_tag is None or tie_tag is None:
            raise ValueError(f"Missing georeferencing tags in {path}")

        scale = scale_tag.value
        tie = tie_tag.value
        if len(scale) < 2 or len(tie) < 6:
            raise ValueError(f"Invalid georeferencing tags in {path}")

        pixel_size_x = float(scale[0])
        pixel_size_y = abs(float(scale[1]))
        origin_x = float(tie[3])
        origin_y = float(tie[4])

        return GeoRef(
            origin_x=origin_x,
            origin_y=origin_y,
            pixel_size_x=pixel_size_x,
            pixel_size_y=pixel_size_y,
            width=page.imagewidth,
            height=page.imagelength,
        )


def _intersection(a: BBox, b: BBox) -> BBox | None:
    min_x = max(a.min_x, b.min_x)
    min_y = max(a.min_y, b.min_y)
    max_x = min(a.max_x, b.max_x)
    max_y = min(a.max_y, b.max_y)
    if min_x >= max_x or min_y >= max_y:
        return None
    return BBox(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y)


def _to_target_pixels(bounds: BBox, target: BBox, target_width: int, target_height: int) -> tuple[int, int, int, int]:
    px_x = (target.max_x - target.min_x) / target_width
    px_y = (target.max_y - target.min_y) / target_height

    x0 = int(round((bounds.min_x - target.min_x) / px_x))
    x1 = int(round((bounds.max_x - target.min_x) / px_x))
    y0 = int(round((target.max_y - bounds.max_y) / px_y))
    y1 = int(round((target.max_y - bounds.min_y) / px_y))

    x0 = max(0, min(target_width, x0))
    x1 = max(0, min(target_width, x1))
    y0 = max(0, min(target_height, y0))
    y1 = max(0, min(target_height, y1))
    return x0, y0, x1, y1


def _to_source_pixels(bounds: BBox, src: GeoRef) -> tuple[int, int, int, int]:
    sx0 = int((bounds.min_x - src.min_x) / src.pixel_size_x)
    sx1 = int((bounds.max_x - src.min_x) / src.pixel_size_x)
    sy0 = int((src.max_y - bounds.max_y) / src.pixel_size_y)
    sy1 = int((src.max_y - bounds.min_y) / src.pixel_size_y)

    sx0 = max(0, min(src.width, sx0))
    sx1 = max(0, min(src.width, sx1))
    sy0 = max(0, min(src.height, sy0))
    sy1 = max(0, min(src.height, sy1))
    return sx0, sy0, sx1, sy1


def _to_source_pixels_swapped(bounds: BBox, src: GeoRef) -> tuple[int, int, int, int]:
    sx0 = int((bounds.min_y - src.origin_x) / src.pixel_size_x)
    sx1 = int((bounds.max_y - src.origin_x) / src.pixel_size_x)
    sy0 = int((src.origin_y - bounds.max_x) / src.pixel_size_y)
    sy1 = int((src.origin_y - bounds.min_x) / src.pixel_size_y)
    sx0 = max(0, min(src.width, sx0))
    sx1 = max(0, min(src.width, sx1))
    sy0 = max(0, min(src.height, sy0))
    sy1 = max(0, min(src.height, sy1))
    return sx0, sy0, sx1, sy1


def _harmonize_rgb_u8(image):
    if image.bands == 1:
        image = image.bandjoin([image, image])
    elif image.bands >= 4:
        image = image.extract_band(0, n=3)
    elif image.bands == 2:
        image = image.extract_band(0).bandjoin([image.extract_band(0), image.extract_band(0)])

    if image.bands > 3:
        image = image.extract_band(0, n=3)
    if image.format != "uchar":
        image = image.cast("uchar")
    return image


def _force_srgb_if_requested(image, enabled: bool):
    if not enabled:
        return image
    try:
        if str(image.interpretation).lower() != "srgb":
            image = image.colourspace("srgb")
    except Exception:
        return image
    return image


def _grayworld_normalize(image):
    means = []
    for band in range(min(3, image.bands)):
        means.append(float(image.extract_band(band).avg()))
    if len(means) != 3:
        return image

    target = sum(means) / 3.0
    if target <= 0.0:
        return image

    scales = []
    for m in means:
        if m <= 0.0:
            scales.append(1.0)
        else:
            scales.append(target / m)

    image = image.cast("float").linear(scales, [0.0, 0.0, 0.0]).cast("uchar")
    return image


def _mean_rgb(image) -> list[float]:
    return [float(image.extract_band(i).avg()) for i in range(min(3, image.bands))]


def _mean_rgb_from_file(path: Path) -> list[float]:
    with tifffile.TiffFile(path) as tif:
        arr = tif.pages[0].asarray()
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.shape[2] > 3:
        arr = arr[:, :, :3]
    return [float(arr[:, :, i].mean()) for i in range(3)]


def _parse_year_from_output_name(path: Path) -> int | None:
    stem = path.stem
    if stem.startswith("year_"):
        tail = stem.split("_", 1)[1]
        if tail.isdigit() and len(tail) == 4:
            return int(tail)
    return None


def _manifest_by_year(manifest: DatasetManifest) -> dict[int, str]:
    by_year: dict[int, str] = {}
    for asset in manifest.assets:
        year = _parse_year_from_output_name(Path(asset))
        if year is not None:
            by_year[year] = asset
    return by_year


def _resolve_wms_reference_asset(
    grouped_assets: dict[int, list[Path]],
    years_source_map: dict[int, str],
    preferred_year: int | None,
) -> tuple[int, Path] | None:
    candidates = sorted(year for year, source in years_source_map.items() if source != "wfs")
    if preferred_year is not None and preferred_year in candidates:
        candidates = [preferred_year] + [y for y in candidates if y != preferred_year]
    for year in candidates:
        for asset in grouped_assets.get(year, []):
            if asset.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
                return year, asset
    return None


def _to_gray_f32(path: Path, max_dim: int = 512) -> np.ndarray:
    with tifffile.TiffFile(path) as tif:
        arr = tif.pages[0].asarray()
    if arr.ndim == 2:
        gray = arr.astype(np.float32)
    else:
        if arr.shape[2] > 3:
            arr = arr[:, :, :3]
        gray = arr.astype(np.float32).mean(axis=2)
    h, w = gray.shape
    scale = min(1.0, float(max_dim) / float(max(h, w)))
    if scale < 1.0:
        out_h = max(1, int(round(h * scale)))
        out_w = max(1, int(round(w * scale)))
        y_idx = (np.linspace(0, h - 1, out_h)).astype(np.int32)
        x_idx = (np.linspace(0, w - 1, out_w)).astype(np.int32)
        gray = gray[np.ix_(y_idx, x_idx)]
    return gray


def _phase_correlation_shift(reference_gray: np.ndarray, moving_gray: np.ndarray) -> tuple[float, float]:
    a = reference_gray - reference_gray.mean()
    b = moving_gray - moving_gray.mean()
    if a.shape != b.shape:
        raise ValueError("phase correlation requires equal image shapes")
    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)
    cps = fa * np.conj(fb)
    denom = np.abs(cps)
    denom[denom == 0] = 1.0
    cps = cps / denom
    corr = np.fft.ifft2(cps)
    peak_idx = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)
    y, x = int(peak_idx[0]), int(peak_idx[1])
    h, w = a.shape
    if x > w // 2:
        x -= w
    if y > h // 2:
        y -= h
    return float(x), float(y)


def _translation_matrix(dx: float, dy: float) -> tuple[float, ...]:
    return (1.0, 0.0, dx, 0.0, 1.0, dy)


def _apply_translation_vips(image, dx: float, dy: float):
    shift_x = int(round(dx))
    shift_y = int(round(dy))
    if shift_x == 0 and shift_y == 0:
        return image
    pad_left = max(0, shift_x)
    pad_top = max(0, shift_y)
    pad_right = max(0, -shift_x)
    pad_bottom = max(0, -shift_y)
    expanded = image.embed(
        pad_left,
        pad_top,
        image.width + pad_left + pad_right,
        image.height + pad_top + pad_bottom,
        extend="black",
    )
    crop_x = pad_right
    crop_y = pad_bottom
    return expanded.crop(crop_x, crop_y, image.width, image.height)


def _apply_swapped_axis_orientation(tile):
    # For swapped georef, standard target orientation requires axis swap + inversion.
    try:
        import pyvips
    except Exception as exc:
        raise RuntimeError(
            "pyvips is required for render stage. Install pyvips in the active environment."
        ) from exc

    arr = np.ndarray(
        buffer=tile.write_to_memory(),
        dtype=np.uint8,
        shape=(tile.height, tile.width, tile.bands),
    )
    arr_t = np.transpose(arr, (1, 0, 2))
    arr_t = arr_t[::-1, ::-1, :].copy()
    return pyvips.Image.new_from_memory(
        arr_t.tobytes(),
        int(arr_t.shape[1]),
        int(arr_t.shape[0]),
        int(arr_t.shape[2]),
        "uchar",
    )


def _intersection_area(a: BBox, b: BBox) -> float:
    inter = _intersection(a, b)
    if inter is None:
        return 0.0
    return max(0.0, (inter.max_x - inter.min_x) * (inter.max_y - inter.min_y))


def _infer_global_wfs_axis_mode(
    grouped_assets: dict[int, list[Path]],
    years_source_map: dict[int, str],
    target_bbox: BBox,
    reference_year: int | None,
) -> str:
    wfs_years = sorted(year for year, src in years_source_map.items() if src == "wfs")
    if not wfs_years:
        return "normal"
    candidate_years = list(wfs_years)
    if reference_year is not None and reference_year in candidate_years:
        candidate_years = [reference_year] + [year for year in candidate_years if year != reference_year]

    normal_area = 0.0
    swapped_area = 0.0
    for year in candidate_years:
        assets = grouped_assets.get(year, [])
        for asset in assets:
            if asset.suffix.lower() not in {".tif", ".tiff"}:
                continue
            georef = _read_georef(asset)
            normal_area += _intersection_area(_bbox_from_georef(georef), target_bbox)
            swapped_area += _intersection_area(_bbox_from_swapped_axis_georef(georef), target_bbox)
    if swapped_area > normal_area:
        return "swapped"
    return "normal"


def _estimate_global_calibration(
    config: RenderConfig,
    source_manifest: DatasetManifest,
    grouped_assets: dict[int, list[Path]],
    target_bbox: BBox,
    target_width: int,
    target_height: int,
) -> Calibration:
    years_source_map = dict(source_manifest.years_source_map)
    axis_mode_no_cal = _infer_global_wfs_axis_mode(
        grouped_assets=grouped_assets,
        years_source_map=years_source_map,
        target_bbox=target_bbox,
        reference_year=config.calibration_reference_year,
    )
    enabled = bool(config.profile == "reference" and config.wfs_global_calibration)
    if not enabled:
        return Calibration(
            enabled=False,
            reference_year=None,
            transform_type=config.calibration_transform,
            source_axis_mode=axis_mode_no_cal,
            matrix=None,
            fit_error_px=None,
            report_path=None,
        )

    wfs_years = sorted(year for year, source in years_source_map.items() if source == "wfs")
    if not wfs_years:
        return Calibration(
            enabled=False,
            reference_year=None,
            transform_type=config.calibration_transform,
            source_axis_mode=axis_mode_no_cal,
            matrix=None,
            fit_error_px=None,
            report_path=None,
        )

    reference_year = config.calibration_reference_year or wfs_years[0]
    if reference_year not in wfs_years:
        raise ValueError(
            f"calibration_reference_year={reference_year} is not a WFS year in this run"
        )
    if config.calibration_transform == "homography":
        logger.warning(
            "Render: calibration_transform=homography requested; using translation-only estimate in current implementation"
        )
    source_axis_mode = _infer_global_wfs_axis_mode(
        grouped_assets=grouped_assets,
        years_source_map=years_source_map,
        target_bbox=target_bbox,
        reference_year=reference_year,
    )

    wms_ref = _resolve_wms_reference_asset(
        grouped_assets=grouped_assets,
        years_source_map=years_source_map,
        preferred_year=config.calibration_reference_year,
    )
    if wms_ref is None:
        raise ValueError(
            "WFS global calibration requires at least one WMS fallback asset in reference profile"
        )
    _, wms_asset = wms_ref

    reference_assets = grouped_assets.get(reference_year, [])
    if not reference_assets:
        raise ValueError(f"No assets for calibration reference year {reference_year}")

    with tempfile.TemporaryDirectory(prefix="satmap_cal_") as temp_dir:
        temp_out = Path(temp_dir) / f"calibration_wfs_year_{reference_year}.tif"
        coverage_ratio, _ = _render_year(
            year=reference_year,
            assets=reference_assets,
            out_path=temp_out,
            target_bbox=target_bbox,
            target_width=target_width,
            target_height=target_height,
            target_srs=config.target_srs,
            resample_method=config.resample_method,
            tile_size=config.tile_size,
            compression=config.compression,
            force_srgb_from_ycbcr=True,
            per_year_color_norm=False,
            calibration_matrix=None,
            source_axis_mode=source_axis_mode,
        )
        if coverage_ratio <= 0.0:
            raise ValueError(
                f"Calibration reference year {reference_year} has zero coverage after axis_mode={source_axis_mode}"
            )

        temp_wms = Path(temp_dir) / "calibration_wms_reference.tif"
        _render_reference_wms_year(
            asset=wms_asset,
            out_path=temp_wms,
            target_bbox=target_bbox,
            target_width=target_width,
            target_height=target_height,
            target_srs=config.target_srs,
        )

        gray_wms = _to_gray_f32(temp_wms)
        gray_wfs = _to_gray_f32(temp_out)
        dx_small, dy_small = _phase_correlation_shift(gray_wms, gray_wfs)
        scale_x = target_width / gray_wms.shape[1]
        scale_y = target_height / gray_wms.shape[0]
        dx = dx_small * scale_x
        dy = dy_small * scale_y

        try:
            import pyvips
        except Exception as exc:
            raise RuntimeError(
                "pyvips is required for render stage. Install pyvips in the active environment."
            ) from exc

        aligned = _apply_translation_vips(
            image=_harmonize_rgb_u8(pyvips.Image.new_from_file(str(temp_out), access="sequential")),
            dx=dx,
            dy=dy,
        )
        temp_aligned = Path(temp_dir) / "calibration_wfs_aligned.tif"
        aligned.tiffsave(str(temp_aligned), compression="deflate", bigtiff=True)
        aligned_gray = _to_gray_f32(temp_aligned)
        res_dx_small, res_dy_small = _phase_correlation_shift(gray_wms, aligned_gray)
        fit_error_px = math.sqrt((res_dx_small * scale_x) ** 2 + (res_dy_small * scale_y) ** 2)

    matrix = _translation_matrix(dx, dy)
    status = "ok"
    notes = "Single global translation estimated from one WFS reference year against WMS reference."
    if fit_error_px > config.calibration_max_error_px:
        status = "degraded_identity"
        notes = (
            "Estimated translation exceeded calibration_max_error_px; "
            "using identity transform while keeping inferred global source_axis_mode."
        )
        matrix = _translation_matrix(0.0, 0.0)
    report_path = config.output_json.parent / "georef_calibration_report.json"
    report_payload = {
        "reference_year": reference_year,
        "source_axis_mode": source_axis_mode,
        "transform_type": config.calibration_transform,
        "transform_matrix": list(matrix),
        "fit_error_px": float(fit_error_px),
        "status": status,
        "notes": notes,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    return Calibration(
        enabled=True,
        reference_year=reference_year,
        transform_type=config.calibration_transform,
        source_axis_mode=source_axis_mode,
        matrix=matrix,
        fit_error_px=float(fit_error_px),
        report_path=report_path,
    )


def _render_cache_signature(
    config: RenderConfig,
    target_bbox: BBox,
    target_width: int,
    target_height: int,
    calibration: Calibration,
) -> str:
    payload = {
        "mode": config.mode,
        "profile": config.profile,
        "target_bbox": [target_bbox.min_x, target_bbox.min_y, target_bbox.max_x, target_bbox.max_y],
        "target_size": [target_width, target_height],
        "target_srs": config.target_srs.upper(),
        "resample_method": config.resample_method,
        "compression": config.compression,
        "tile_size": config.tile_size,
        "wfs_global_calibration": config.wfs_global_calibration,
        "calibration_reference_year": calibration.reference_year,
        "calibration_transform": config.calibration_transform,
        "source_axis_mode": calibration.source_axis_mode,
        "calibration_matrix": list(calibration.matrix) if calibration.matrix else None,
        "calibration_max_error_px": config.calibration_max_error_px,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _extract_epsg_from_geokey(page: tifffile.TiffPage) -> int | None:
    geo_key = page.tags.get("GeoKeyDirectoryTag")
    if geo_key is None:
        return None
    raw = [int(v) for v in geo_key.value]
    if len(raw) < 4:
        return None
    key_count = raw[3]
    idx = 4
    for _ in range(key_count):
        if idx + 3 >= len(raw):
            break
        key_id, tiff_tag_location, count, value_offset = raw[idx : idx + 4]
        idx += 4
        if key_id == 3072 and count == 1 and tiff_tag_location == 0:
            return int(value_offset)
    return None


def _can_reuse_render_output(
    out_path: Path,
    target_bbox: BBox,
    target_width: int,
    target_height: int,
    target_srs: str,
) -> bool:
    if not out_path.exists():
        return False
    if not out_path.with_suffix(".tfw").exists():
        return False
    if target_srs.upper() == "EPSG:2180" and not out_path.with_suffix(".prj").exists():
        return False

    try:
        with tifffile.TiffFile(out_path) as tif:
            page = tif.pages[0]
            if int(page.imagewidth) != target_width or int(page.imagelength) != target_height:
                return False
            if page.tags.get("ModelPixelScaleTag") is None or page.tags.get("ModelTiepointTag") is None:
                return False
            if target_srs.upper() == "EPSG:2180":
                if _extract_epsg_from_geokey(page) != 2180:
                    return False
            # Guard against corrupted/empty cached renders (tiny all-black artifacts).
            arr = page.asarray()
            if arr.ndim == 2:
                arr = arr[:, :, None]
            if arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
            if arr.shape[2] > 3:
                arr = arr[:, :, :3]
            if float(arr.mean()) <= 0.0:
                return False
        georef = _read_georef(out_path)
        expected_px_x, expected_px_y = _pixel_size_from_bbox(target_bbox, target_width, target_height)
        tol_x = max(1e-9, expected_px_x)
        tol_y = max(1e-9, expected_px_y)
        if abs(georef.min_x - target_bbox.min_x) > tol_x:
            return False
        if abs(georef.max_x - target_bbox.max_x) > tol_x:
            return False
        if abs(georef.min_y - target_bbox.min_y) > tol_y:
            return False
        if abs(georef.max_y - target_bbox.max_y) > tol_y:
            return False
    except Exception:
        return False
    return True


def _pixel_size_from_bbox(target_bbox: BBox, target_width: int, target_height: int) -> tuple[float, float]:
    pixel_size_x = (target_bbox.max_x - target_bbox.min_x) / target_width
    pixel_size_y = (target_bbox.max_y - target_bbox.min_y) / target_height
    return pixel_size_x, pixel_size_y


def _geo_key_directory_for_epsg_2180() -> tuple[int, ...]:
    # GeoKeyDirectoryTag: header + entries (KeyID, TIFFTagLocation, Count, ValueOffset)
    return (
        1,
        1,
        0,
        4,
        1024,
        0,
        1,
        1,
        1025,
        0,
        1,
        1,
        3072,
        0,
        1,
        2180,
        3076,
        0,
        1,
        9001,
    )


def _write_worldfile_and_prj(out_path: Path, target_bbox: BBox, target_width: int, target_height: int, srs: str) -> None:
    pixel_size_x, pixel_size_y = _pixel_size_from_bbox(target_bbox, target_width, target_height)
    c = target_bbox.min_x + pixel_size_x / 2.0
    f = target_bbox.max_y - pixel_size_y / 2.0

    tfw_path = out_path.with_suffix(".tfw")
    tfw_path.write_text(
        "\n".join(
            [
                f"{pixel_size_x:.12f}",
                "0.0",
                "0.0",
                f"{-pixel_size_y:.12f}",
                f"{c:.12f}",
                f"{f:.12f}",
            ]
        )
        + "\n",
        encoding="ascii",
    )

    if srs.upper() == "EPSG:2180":
        out_path.with_suffix(".prj").write_text(EPSG_2180_WKT, encoding="ascii")


def _apply_geotiff_tags_exiftool(out_path: Path, target_bbox: BBox, target_width: int, target_height: int, srs: str) -> bool:
    exiftool = shutil.which("exiftool")
    if exiftool is None or srs.upper() != "EPSG:2180":
        return False

    pixel_size_x, pixel_size_y = _pixel_size_from_bbox(target_bbox, target_width, target_height)
    tie = f"0 0 0 {target_bbox.min_x} {target_bbox.max_y} 0"
    scale = f"{pixel_size_x} {pixel_size_y} 0"

    cmd = [
        exiftool,
        "-overwrite_original",
        f"-ModelPixelScaleTag={scale}",
        f"-ModelTiePointTag={tie}",
        "-GTModelTypeGeoKey=1",
        "-GTRasterTypeGeoKey=1",
        "-ProjectedCSTypeGeoKey=2180",
        "-ProjLinearUnitsGeoKey=9001",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return result.returncode == 0


def _apply_geotiff_tags_tifffile(out_path: Path, target_bbox: BBox, target_width: int, target_height: int, srs: str) -> None:
    if srs.upper() != "EPSG:2180":
        raise ValueError(f"Unsupported SRS for GeoTIFF tagging: {srs}")

    pixel_size_x, pixel_size_y = _pixel_size_from_bbox(target_bbox, target_width, target_height)
    scale = (float(pixel_size_x), float(pixel_size_y), 0.0)
    tie = (0.0, 0.0, 0.0, float(target_bbox.min_x), float(target_bbox.max_y), 0.0)
    geokey = _geo_key_directory_for_epsg_2180()

    temp_path = out_path.with_suffix(out_path.suffix + ".georef_tmp")
    if temp_path.exists():
        temp_path.unlink()

    with tifffile.TiffFile(out_path) as src, tifffile.TiffWriter(temp_path, bigtiff=True) as dst:
        for idx, page in enumerate(src.pages):
            data = page.asarray()
            tile_w_tag = page.tags.get("TileWidth")
            tile_h_tag = page.tags.get("TileLength")
            tile_w = int(tile_w_tag.value) if tile_w_tag is not None else int(page.imagewidth)
            tile_h = int(tile_h_tag.value) if tile_h_tag is not None else int(page.imagelength)
            tile = None
            if tile_w >= 16 and tile_h >= 16 and tile_w % 16 == 0 and tile_h % 16 == 0:
                tile = (tile_h, tile_w)

            compression_tag = page.tags.get("Compression")
            compression = "deflate"
            predictor = None
            if compression_tag is not None and int(compression_tag.value) == 7:
                compression = "jpeg"

            extratags = []
            if idx == 0:
                extratags.extend(
                    [
                        (33550, "d", 3, scale, False),
                        (33922, "d", 6, tie, False),
                        (34735, "H", len(geokey), geokey, False),
                    ]
                )

            photometric = None
            if data.ndim == 3 and data.shape[2] in {3, 4}:
                photometric = "rgb"

            dst.write(
                data,
                photometric=photometric,
                tile=tile,
                compression=compression,
                predictor=predictor,
                subfiletype=int(page.tags.get("SubfileType", 0).value) if page.tags.get("SubfileType") else 0,
                metadata=None,
                extratags=extratags,
            )

    temp_path.replace(out_path)


def _ensure_georeferenced_output(out_path: Path, target_bbox: BBox, target_width: int, target_height: int, srs: str) -> None:
    tagged = _apply_geotiff_tags_exiftool(out_path, target_bbox, target_width, target_height, srs)
    if not tagged:
        _apply_geotiff_tags_tifffile(out_path, target_bbox, target_width, target_height, srs)
    _write_worldfile_and_prj(out_path, target_bbox, target_width, target_height, srs)


def _render_year(
    year: int,
    assets: list[Path],
    out_path: Path,
    target_bbox: BBox,
    target_width: int,
    target_height: int,
    target_srs: str,
    resample_method: str,
    tile_size: int,
    compression: str,
    force_srgb_from_ycbcr: bool,
    per_year_color_norm: bool,
    calibration_matrix: tuple[float, ...] | None,
    source_axis_mode: str,
):
    try:
        import pyvips
    except Exception as exc:
        raise RuntimeError(
            "pyvips is required for render stage. Install pyvips in the active environment."
        ) from exc

    canvas = pyvips.Image.black(target_width, target_height, bands=3).copy(interpretation="srgb")
    covered_area = 0.0
    bbox_area = max(
        1e-9,
        (target_bbox.max_x - target_bbox.min_x) * (target_bbox.max_y - target_bbox.min_y),
    )

    for asset in assets:
        georef = _read_georef(asset)
        src_bbox = _bbox_from_swapped_axis_georef(georef) if source_axis_mode == "swapped" else _bbox_from_georef(georef)
        inter = _intersection(src_bbox, target_bbox)
        if inter is None:
            continue
        covered_area += (inter.max_x - inter.min_x) * (inter.max_y - inter.min_y)

        if source_axis_mode == "swapped":
            sx0, sy0, sx1, sy1 = _to_source_pixels_swapped(inter, georef)
        else:
            sx0, sy0, sx1, sy1 = _to_source_pixels(inter, georef)
        dx0, dy0, dx1, dy1 = _to_target_pixels(inter, target_bbox, target_width, target_height)

        src_w = sx1 - sx0
        src_h = sy1 - sy0
        dst_w = dx1 - dx0
        dst_h = dy1 - dy0
        if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
            continue

        tile = pyvips.Image.new_from_file(str(asset), access="sequential")
        tile = _force_srgb_if_requested(tile, force_srgb_from_ycbcr)
        tile = _harmonize_rgb_u8(tile)
        tile = tile.crop(sx0, sy0, src_w, src_h)
        if source_axis_mode == "swapped":
            tile = _apply_swapped_axis_orientation(tile)
            src_w = tile.width
            src_h = tile.height

        kernel = "linear" if resample_method == "bilinear" else "nearest"
        x_scale = dst_w / src_w
        y_scale = dst_h / src_h
        tile = tile.resize(x_scale, vscale=y_scale, kernel=kernel)

        if tile.width != dst_w or tile.height != dst_h:
            tile = tile.crop(0, 0, min(tile.width, dst_w), min(tile.height, dst_h))
            if tile.width != dst_w or tile.height != dst_h:
                pad_w = max(0, dst_w - tile.width)
                pad_h = max(0, dst_h - tile.height)
                tile = tile.embed(0, 0, tile.width + pad_w, tile.height + pad_h, extend="black")

        canvas = canvas.insert(tile, dx0, dy0, expand=False)

    if per_year_color_norm:
        canvas = _grayworld_normalize(canvas)
    if calibration_matrix is not None:
        dx = float(calibration_matrix[2])
        dy = float(calibration_matrix[5])
        canvas = _apply_translation_vips(canvas, dx=dx, dy=dy)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.tiffsave(
        str(out_path),
        tile=True,
        tile_width=tile_size,
        tile_height=tile_size,
        compression=compression,
        pyramid=True,
        bigtiff=True,
        predictor="horizontal",
    )

    _ensure_georeferenced_output(out_path, target_bbox, target_width, target_height, target_srs)
    return min(1.0, covered_area / bbox_area), _mean_rgb(canvas)


def _render_reference_wms_year(
    asset: Path,
    out_path: Path,
    target_bbox: BBox,
    target_width: int,
    target_height: int,
    target_srs: str,
):
    try:
        import pyvips
    except Exception as exc:
        raise RuntimeError(
            "pyvips is required for render stage. Install pyvips in the active environment."
        ) from exc

    image = pyvips.Image.new_from_file(str(asset), access="random")
    image = _force_srgb_if_requested(image, True)
    image = _harmonize_rgb_u8(image)
    if image.width != target_width or image.height != target_height:
        x_scale = target_width / max(1, image.width)
        y_scale = target_height / max(1, image.height)
        image = image.resize(x_scale, vscale=y_scale, kernel="linear")
        image = image.crop(0, 0, min(image.width, target_width), min(image.height, target_height))
        if image.width != target_width or image.height != target_height:
            image = image.embed(0, 0, target_width, target_height, extend="black")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.tiffsave(
        str(out_path),
        tile=True,
        tile_width=512,
        tile_height=512,
        compression="deflate",
        pyramid=True,
        bigtiff=True,
        predictor="horizontal",
    )

    _ensure_georeferenced_output(out_path, target_bbox, target_width, target_height, target_srs)
    return 1.0, _mean_rgb_from_file(out_path)


def _collect_stats(path: Path) -> dict[str, int | str | None]:
    try:
        import pyvips
    except Exception:
        return {"width": None, "height": None, "bands": None, "dtype": None}

    image = pyvips.Image.new_from_file(str(path), access="sequential")
    return {
        "width": int(image.width),
        "height": int(image.height),
        "bands": int(image.bands),
        "dtype": str(image.format),
    }


def run(config: RenderConfig) -> tuple[int, Path]:
    logger.info(
        "Render: start input=%s root=%s mode=%s profile=%s resample=%s auto_size_from_bbox=%s",
        config.dataset_manifest,
        config.render_root,
        config.mode,
        config.profile,
        config.resample_method,
        config.auto_size_from_bbox,
    )
    source_manifest = _read_dataset_manifest(config.dataset_manifest)
    if config.experimental_force_srgb_from_ycbcr:
        logger.warning("Render: experimental_force_srgb_from_ycbcr enabled")
    if config.experimental_per_year_color_norm:
        logger.warning("Render: experimental_per_year_color_norm enabled")

    if not source_manifest.years_included:
        manifest = DatasetManifest(
            stage="render",
            years_requested=source_manifest.years_requested,
            years_available_wfs=source_manifest.years_available_wfs,
            years_included=[],
            years_excluded_with_reason=source_manifest.years_excluded_with_reason,
            common_tile_ids=source_manifest.common_tile_ids,
            tile_sources_by_year=source_manifest.tile_sources_by_year,
            assets=[],
            source_manifest=str(config.dataset_manifest),
            mode=config.mode,
            target_width=config.target_width,
            target_height=config.target_height,
            target_bbox=config.target_bbox,
            target_srs=config.target_srs,
            profile=config.profile,
            px_per_meter=config.px_per_meter,
            resample_method=config.resample_method,
            render_backend="pyvips",
            pixel_profile="RGB_U8",
            passed=False,
            notes="No years_included available for render stage.",
        )
        _write_json(config.output_json, manifest)
        logger.error("Render: no years_included; aborting")
        return 1, config.output_json

    target_bbox = _resolve_target_bbox(config, source_manifest, config.dataset_manifest)
    grouped_assets = _group_assets_by_year(source_manifest.assets, config.dataset_manifest)
    target_width, target_height = _resolve_target_dimensions(config, target_bbox)
    years_source_map = dict(source_manifest.years_source_map)
    inferred_source_axis_mode = _infer_global_wfs_axis_mode(
        grouped_assets=grouped_assets,
        years_source_map=years_source_map,
        target_bbox=target_bbox,
        reference_year=config.calibration_reference_year,
    )
    logger.info(
        "Render: target_bbox=%s,%s,%s,%s target=%sx%s years=%s",
        target_bbox.min_x,
        target_bbox.min_y,
        target_bbox.max_x,
        target_bbox.max_y,
        target_width,
        target_height,
        len(source_manifest.years_included),
    )
    if inferred_source_axis_mode == "swapped":
        logger.warning(
            "Render: inferred WFS source_axis_mode=swapped from tile georeferencing; "
            "applying deterministic axis fix for WFS years."
        )

    calibration_error: str | None = None
    calibration = Calibration(
        enabled=False,
        reference_year=None,
        transform_type=config.calibration_transform,
        source_axis_mode=inferred_source_axis_mode,
        matrix=None,
        fit_error_px=None,
        report_path=None,
    )
    if config.wfs_global_calibration:
        try:
            calibration = _estimate_global_calibration(
                config=config,
                source_manifest=source_manifest,
                grouped_assets=grouped_assets,
                target_bbox=target_bbox,
                target_width=target_width,
                target_height=target_height,
            )
        except Exception as exc:
            calibration_error = str(exc)
            logger.error("Render: global calibration failed: %s", calibration_error)

    if calibration_error is not None and config.wfs_global_calibration:
        manifest = DatasetManifest(
            stage="render",
            years_requested=source_manifest.years_requested,
            years_available_wfs=source_manifest.years_available_wfs,
            years_included=source_manifest.years_included,
            years_excluded_with_reason=source_manifest.years_excluded_with_reason,
            common_tile_ids=source_manifest.common_tile_ids,
            tile_sources_by_year=source_manifest.tile_sources_by_year,
            assets=[],
            source_manifest=str(config.dataset_manifest),
            mode=config.mode,
            target_width=target_width,
            target_height=target_height,
            target_bbox=f"{target_bbox.min_x},{target_bbox.min_y},{target_bbox.max_x},{target_bbox.max_y}",
            target_srs=config.target_srs,
            profile=config.profile,
            px_per_meter=config.px_per_meter,
            years_source_map=source_manifest.years_source_map,
            resample_method=config.resample_method,
            render_backend="pyvips",
            pixel_profile="RGB_U8",
            wfs_global_calibration=config.wfs_global_calibration,
            calibration_reference_year=config.calibration_reference_year,
            calibration_transform=config.calibration_transform,
            calibration_source_axis_mode=calibration.source_axis_mode,
            calibration_max_error_px=config.calibration_max_error_px,
            passed=False,
            notes=f"Global calibration failed: {calibration_error}",
        )
        _write_json(config.output_json, manifest)
        return 1, config.output_json

    cache_signature = _render_cache_signature(
        config=config,
        target_bbox=target_bbox,
        target_width=target_width,
        target_height=target_height,
        calibration=calibration,
    )
    previous_manifest = _read_dataset_manifest(config.output_json) if config.output_json.exists() else None
    reusable_assets = {}
    if previous_manifest and previous_manifest.render_cache_signature == cache_signature:
        reusable_assets = _manifest_by_year(previous_manifest)

    rendered_assets: list[str] = []
    asset_stats: dict[str, dict[str, int | str | None]] = {}
    coverage_ratio_by_year: dict[int, float] = {}
    color_qc_by_year: dict[int, dict[str, float | list[float] | None]] = {}
    calibration_status_by_year: dict[int, str] = {}
    calibration_error_px_by_year: dict[int, float] = {}
    errors: list[str] = []

    for year in source_manifest.years_included:
        year_assets = grouped_assets.get(year, [])
        logger.info("Render: year=%s input_tiles=%s", year, len(year_assets))
        if not year_assets:
            errors.append(f"No assets for year {year} in download manifest.")
            continue

        out_path = config.render_root / f"year_{year}.tiff"
        source_type = years_source_map.get(year, "wfs")
        has_reuse_entry = year in reusable_assets and Path(reusable_assets[year]).name == out_path.name
        if has_reuse_entry and _can_reuse_render_output(
            out_path=out_path,
            target_bbox=target_bbox,
            target_width=target_width,
            target_height=target_height,
            target_srs=config.target_srs,
        ):
            rendered_assets.append(str(out_path))
            asset_stats[str(out_path)] = _collect_stats(out_path)
            cached_coverage = None
            if previous_manifest is not None:
                cached_coverage = previous_manifest.coverage_ratio_by_year.get(year)
            coverage_ratio_by_year[year] = float(cached_coverage or 1.0)
            if source_type == "wfs" and calibration.enabled:
                calibration_status_by_year[year] = "applied"
                calibration_error_px_by_year[year] = float(calibration.fit_error_px or 0.0)
            elif source_type in {"wms", "wms_fallback"}:
                calibration_status_by_year[year] = "skipped_wms"
            else:
                calibration_status_by_year[year] = "disabled"
            color_qc_by_year[year] = {
                "mean_rgb": _mean_rgb_from_file(out_path),
                "delta_to_wms_reference": None,
            }
            logger.info("Render: year=%s reusing existing output=%s", year, out_path)
            continue

        try:
            year_calibration = calibration.matrix if (source_type == "wfs" and calibration.enabled) else None
            coverage_ratio, mean_rgb = _render_year(
                year=year,
                assets=year_assets,
                out_path=out_path,
                target_bbox=target_bbox,
                target_width=target_width,
                target_height=target_height,
                target_srs=config.target_srs,
                resample_method=config.resample_method,
                tile_size=config.tile_size,
                compression=config.compression,
                force_srgb_from_ycbcr=(
                    config.experimental_force_srgb_from_ycbcr
                    or source_type in {"wms", "wms_fallback"}
                ),
                per_year_color_norm=(
                    config.experimental_per_year_color_norm
                    and not config.disable_color_norm
                    and source_type == "wfs"
                    and config.profile == "train"
                ),
                calibration_matrix=year_calibration,
                source_axis_mode=(calibration.source_axis_mode if source_type == "wfs" else "normal"),
            )
            rendered_assets.append(str(out_path))
            asset_stats[str(out_path)] = _collect_stats(out_path)
            coverage_ratio_by_year[year] = float(coverage_ratio)
            if source_type == "wfs" and calibration.enabled:
                calibration_status_by_year[year] = "applied"
                calibration_error_px_by_year[year] = float(calibration.fit_error_px or 0.0)
            elif source_type in {"wms", "wms_fallback"}:
                calibration_status_by_year[year] = "skipped_wms"
            else:
                calibration_status_by_year[year] = "disabled"
            color_qc_by_year[year] = {
                "mean_rgb": mean_rgb,
                "delta_to_wms_reference": None,
            }
            logger.info("Render: year=%s done output=%s", year, out_path)
        except Exception as exc:
            errors.append(f"Render failed for year {year}: {exc}")
            logger.exception("Render: year=%s failed: %r", year, exc)

    passed = source_manifest.passed and not errors and len(rendered_assets) == len(source_manifest.years_included)
    notes = f"Rendered years={len(rendered_assets)} errors={len(errors)} overviews={config.overview_levels}"
    if errors:
        notes += " | " + " ; ".join(errors[:3])

    diagnostics_report_path: str | None = None
    diagnostics_quicklook_dir: str | None = None
    try:
        rendered_assets_by_year: dict[int, Path] = {}
        for asset in rendered_assets:
            year = _parse_year_from_output_name(Path(asset))
            if year is not None:
                rendered_assets_by_year[year] = Path(asset)
        diag_path, quicklook_dir = diagnostics.build_mismatch_report(
            output_json_path=config.output_json.parent / "wfs_wms_mismatch_report.json",
            years=source_manifest.years_included,
            rendered_assets=rendered_assets_by_year,
            source_assets_by_year=grouped_assets,
            years_source_map=years_source_map,
            target_bbox=f"{target_bbox.min_x},{target_bbox.min_y},{target_bbox.max_x},{target_bbox.max_y}",
            target_width=target_width,
            target_height=target_height,
        )
        diagnostics_report_path = str(diag_path)
        diagnostics_quicklook_dir = str(quicklook_dir)
    except Exception as exc:
        notes += f" | diagnostics_error={exc}"

    manifest = DatasetManifest(
        stage="render",
        years_requested=source_manifest.years_requested,
        years_available_wfs=source_manifest.years_available_wfs,
        years_included=source_manifest.years_included,
        years_excluded_with_reason=source_manifest.years_excluded_with_reason,
        common_tile_ids=source_manifest.common_tile_ids,
        tile_sources_by_year=source_manifest.tile_sources_by_year,
        assets=rendered_assets,
        source_manifest=str(config.dataset_manifest),
        mode=config.mode,
        target_width=target_width,
        target_height=target_height,
        target_bbox=f"{target_bbox.min_x},{target_bbox.min_y},{target_bbox.max_x},{target_bbox.max_y}",
        target_srs=config.target_srs,
        profile=config.profile,
        px_per_meter=config.px_per_meter,
        years_source_map=years_source_map,
        coverage_ratio_by_year=coverage_ratio_by_year,
        color_qc_by_year=color_qc_by_year,
        resample_method=config.resample_method,
        render_backend="pyvips",
        asset_stats=asset_stats,
        pixel_profile="RGB_U8",
        wfs_global_calibration=config.wfs_global_calibration,
        calibration_reference_year=calibration.reference_year,
        calibration_transform=config.calibration_transform,
        calibration_source_axis_mode=calibration.source_axis_mode,
        calibration_max_error_px=config.calibration_max_error_px,
        calibration_fit_error_px=calibration.fit_error_px,
        calibration_matrix=list(calibration.matrix) if calibration.matrix else None,
        calibration_report_path=str(calibration.report_path) if calibration.report_path else None,
        calibration_status_by_year=calibration_status_by_year,
        calibration_error_px_by_year=calibration_error_px_by_year,
        render_cache_signature=cache_signature,
        diagnostics_report_path=diagnostics_report_path,
        diagnostics_quicklook_dir=diagnostics_quicklook_dir,
        passed=passed,
        notes=notes,
    )
    _write_json(config.output_json, manifest)
    logger.info(
        "Render: finished passed=%s rendered=%s errors=%s output=%s",
        manifest.passed,
        len(rendered_assets),
        len(errors),
        config.output_json,
    )
    return (0 if manifest.passed else 1), config.output_json
