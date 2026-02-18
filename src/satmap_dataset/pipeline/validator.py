from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import logging

import tifffile

from satmap_dataset.config import ValidateConfig
from satmap_dataset.models import DatasetManifest, ValidationReport

logger = logging.getLogger("satmap_dataset.validate")


@dataclass(frozen=True)
class YearPolicyOutcome:
    passed: bool
    years_included: list[int]
    missing_years: list[int]
    errors: list[str]
    warnings: list[str]


@dataclass(frozen=True)
class BBox:
    min_x: float
    min_y: float
    max_x: float
    max_y: float


def _parse_bbox(value: str) -> BBox:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must have 4 numeric values")
    min_x, min_y, max_x, max_y = parts
    if min_x >= max_x or min_y >= max_y:
        raise ValueError("bbox must satisfy min_x<max_x and min_y<max_y")
    return BBox(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y)


def evaluate_year_policy(
    requested_years: Sequence[int],
    available_years: Sequence[int],
    *,
    strict_years: bool,
    min_years: int,
) -> YearPolicyOutcome:
    if min_years < 1:
        raise ValueError("min_years must be >= 1")

    requested = sorted(set(requested_years))
    available = sorted(set(available_years))
    available_set = set(available)

    if requested:
        effective_available = [year for year in requested if year in available_set]
    else:
        effective_available = available

    missing = [year for year in requested if year not in available_set]

    errors: list[str] = []
    warnings: list[str] = []

    if len(effective_available) < min_years:
        errors.append(
            f"min_years policy failed: available={len(effective_available)} < min_years={min_years}"
        )

    if strict_years and missing:
        errors.append(
            f"strict_years=True policy failed: missing requested years {missing}"
        )
    elif not strict_years and missing:
        warnings.append(
            f"strict_years=False: missing requested years are tolerated {missing}"
        )

    return YearPolicyOutcome(
        passed=not errors,
        years_included=effective_available,
        missing_years=missing,
        errors=errors,
        warnings=warnings,
    )


def build_validation_report(
    *,
    requested_years: Sequence[int],
    years_included: Sequence[int],
    years_excluded_with_reason: dict[int, str],
    strict_years: bool,
    min_years: int,
) -> ValidationReport:
    policy = evaluate_year_policy(
        requested_years=requested_years,
        available_years=years_included,
        strict_years=strict_years,
        min_years=min_years,
    )

    return ValidationReport(
        requested_years=sorted(set(requested_years)),
        years_included=policy.years_included,
        years_excluded_with_reason=years_excluded_with_reason,
        missing_years=policy.missing_years,
        strict_years=strict_years,
        min_years=min_years,
        passed=policy.passed,
        errors=policy.errors,
        warnings=policy.warnings,
    )


def _read_dataset_manifest(path: Path) -> DatasetManifest:
    return DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: ValidationReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")


def _resolve_asset_path(asset: str, dataset_manifest_path: Path) -> Path:
    p = Path(asset)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    return (dataset_manifest_path.parent / p).resolve()


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
        if key_id == 3072 and count == 1:
            if tiff_tag_location == 0:
                return int(value_offset)
            geo_ascii = page.tags.get("GeoAsciiParamsTag")
            if geo_ascii is not None and tiff_tag_location == geo_ascii.code:
                return None
    return None


def _read_georef_bbox_and_epsg(asset_path: Path) -> tuple[BBox, int | None]:
    with tifffile.TiffFile(asset_path) as tif:
        page = tif.pages[0]
        scale_tag = page.tags.get("ModelPixelScaleTag")
        tie_tag = page.tags.get("ModelTiepointTag")
        if scale_tag is None or tie_tag is None:
            raise ValueError("Missing ModelPixelScaleTag or ModelTiepointTag")

        scale = scale_tag.value
        tie = tie_tag.value
        if len(scale) < 2 or len(tie) < 6:
            raise ValueError("Invalid ModelPixelScaleTag or ModelTiepointTag")

        pixel_size_x = float(scale[0])
        pixel_size_y = abs(float(scale[1]))
        origin_x = float(tie[3])
        origin_y = float(tie[4])

        width = int(page.imagewidth)
        height = int(page.imagelength)
        bbox = BBox(
            min_x=origin_x,
            min_y=origin_y - height * pixel_size_y,
            max_x=origin_x + width * pixel_size_x,
            max_y=origin_y,
        )
        epsg = _extract_epsg_from_geokey(page)
    return bbox, epsg


def _bbox_within_tolerance(actual: BBox, expected: BBox, tol_x: float, tol_y: float) -> bool:
    return (
        abs(actual.min_x - expected.min_x) <= tol_x
        and abs(actual.max_x - expected.max_x) <= tol_x
        and abs(actual.min_y - expected.min_y) <= tol_y
        and abs(actual.max_y - expected.max_y) <= tol_y
    )


def _validate_sidecars(asset_path: Path, target_srs: str) -> list[str]:
    errors: list[str] = []
    tfw = asset_path.with_suffix(".tfw")
    prj = asset_path.with_suffix(".prj")

    if not tfw.exists():
        errors.append(f"Missing world file for {asset_path.name}: {tfw.name}")
    else:
        lines = [line.strip() for line in tfw.read_text(encoding="ascii").splitlines() if line.strip()]
        if len(lines) != 6:
            errors.append(f"Invalid world file line count for {asset_path.name}: expected 6")
        else:
            try:
                [float(v) for v in lines]
            except ValueError:
                errors.append(f"World file contains non-numeric values for {asset_path.name}")

    if target_srs.upper() == "EPSG:2180":
        if not prj.exists():
            errors.append(f"Missing projection file for {asset_path.name}: {prj.name}")
        else:
            prj_text = prj.read_text(encoding="ascii", errors="ignore")
            if "EPSG\",\"2180" not in prj_text and "CS92" not in prj_text:
                errors.append(f"Projection file does not indicate EPSG:2180 for {asset_path.name}")
    return errors


def run(config: ValidateConfig) -> tuple[int, Path]:
    logger.info("Validate: start dataset_manifest=%s", config.dataset_manifest)
    dataset_manifest = _read_dataset_manifest(config.dataset_manifest)
    requested_years = config.requested_years or dataset_manifest.years_requested
    years_included = dataset_manifest.years_included

    report = build_validation_report(
        requested_years=requested_years,
        years_included=years_included,
        years_excluded_with_reason=dataset_manifest.years_excluded_with_reason,
        strict_years=config.strict_years,
        min_years=config.min_years,
    )
    report = report.model_copy(update={"run_parameters": config.model_dump(mode="json")})

    missing_assets = []
    for asset in dataset_manifest.assets:
        asset_path = _resolve_asset_path(asset, config.dataset_manifest)
        if not asset_path.exists():
            missing_assets.append(asset)
    if missing_assets:
        report.errors.append(f"Missing asset files: {len(missing_assets)}")
        report.passed = False

    if dataset_manifest.pixel_profile != "RGB_U8":
        report.errors.append(f"Invalid pixel profile: {dataset_manifest.pixel_profile}")
        report.passed = False

    expected_count = len(dataset_manifest.years_included)
    if len(dataset_manifest.assets) != expected_count:
        report.errors.append(
            f"Rendered asset count mismatch: assets={len(dataset_manifest.assets)} years_included={expected_count}"
        )
        report.passed = False

    width_ref = dataset_manifest.target_width
    height_ref = dataset_manifest.target_height
    target_bbox = _parse_bbox(dataset_manifest.target_bbox) if dataset_manifest.target_bbox else None
    target_srs = (dataset_manifest.target_srs or "EPSG:2180").upper()

    for asset in dataset_manifest.assets:
        asset_path = _resolve_asset_path(asset, config.dataset_manifest)
        if not asset_path.exists():
            continue
        try:
            with tifffile.TiffFile(asset_path) as tif:
                page = tif.pages[0]
                width = int(page.imagewidth)
                height = int(page.imagelength)
                samples = int(getattr(page, "samplesperpixel", 1))
                dtype = str(page.dtype)

            if width_ref is not None and height_ref is not None:
                if width != width_ref or height != height_ref:
                    report.errors.append(
                        f"Invalid asset size for {asset}: {width}x{height} expected {width_ref}x{height_ref}"
                    )
                    report.passed = False
            if samples != 3:
                report.errors.append(f"Invalid band count for {asset}: {samples} expected 3")
                report.passed = False
            if dtype != "uint8":
                report.errors.append(f"Invalid dtype for {asset}: {dtype} expected uint8")
                report.passed = False

            georef_bbox, epsg = _read_georef_bbox_and_epsg(asset_path)
            if target_srs == "EPSG:2180" and epsg != 2180:
                report.errors.append(f"Invalid GeoTIFF EPSG for {asset}: {epsg} expected 2180")
                report.passed = False

            if target_bbox is not None and width_ref is not None and height_ref is not None:
                tol_x = max(1e-9, (target_bbox.max_x - target_bbox.min_x) / width_ref)
                tol_y = max(1e-9, (target_bbox.max_y - target_bbox.min_y) / height_ref)
                if not _bbox_within_tolerance(georef_bbox, target_bbox, tol_x, tol_y):
                    report.errors.append(
                        f"Geo bbox mismatch for {asset}: actual={georef_bbox} expected={target_bbox}"
                    )
                    report.passed = False

            for msg in _validate_sidecars(asset_path, target_srs):
                report.errors.append(msg)
                report.passed = False

        except Exception as exc:
            report.errors.append(f"Failed to inspect rendered asset {asset}: {exc}")
            report.passed = False

    _write_json(config.output_json, report)
    logger.info(
        "Validate: finished passed=%s errors=%s warnings=%s output=%s",
        report.passed,
        len(report.errors),
        len(report.warnings),
        config.output_json,
    )

    return (0 if report.passed else 1), config.output_json
