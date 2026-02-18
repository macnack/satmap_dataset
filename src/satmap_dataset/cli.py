from __future__ import annotations

import json
from pathlib import Path
import subprocess
import math
import re
import unicodedata

import typer
from pydantic import ValidationError
from rich.console import Console
import httpx

from satmap_dataset.models import IndexManifest
from satmap_dataset.logging_utils import configure_logging
from satmap_dataset.config import (
    DownloadConfig,
    IndexConfig,
    RenderConfig,
    RunConfig,
    ValidateConfig,
)
from satmap_dataset.pipeline import downloader, index_builder, render, run_all, validator

app = typer.Typer(help="satmap_dataset CLI (WFS-first pipeline)", no_args_is_help=True)
console = Console(stderr=True)
DEFAULT_CENTER_SQUARE_KM = 4.0


def _print_validation_error(error: ValidationError) -> None:
    console.print("[red]Invalid configuration:[/red]")
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"])
        console.print(f"- {location}: {item['msg']}")


def _finish(exit_code: int, artifact_path: Path) -> None:
    typer.echo(str(artifact_path))
    raise typer.Exit(code=exit_code)


def _lonlat_to_epsg2180(lon: float, lat: float) -> tuple[float, float]:
    # Return strict EPSG:2180 project axis order: (x, y).
    # Example Poznan center 52.4012627,16.9517999 -> x~360700, y~505900.
    try:
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:2180", always_xy=True)
        x, y = transformer.transform(lon, lat)
        return float(x), float(y)
    except Exception:
        proj_cmd = [
            "proj",
            "+proj=tmerc",
            "+lat_0=0",
            "+lon_0=19",
            "+k=0.9993",
            "+x_0=500000",
            "+y_0=-5300000",
            "+ellps=GRS80",
            "+units=m",
            "+no_defs",
        ]
        try:
            completed = subprocess.run(
                proj_cmd,
                input=f"{lon} {lat}\n",
                text=True,
                capture_output=True,
                check=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "Center-based bbox input requires pyproj or the PROJ 'proj' CLI in PATH."
            ) from exc
        values = completed.stdout.strip().split()
        if len(values) < 2:
            raise RuntimeError("Failed to parse PROJ output for center-based bbox conversion.")
        try:
            x = float(values[0])
            y = float(values[1])
        except ValueError as exc:
            raise RuntimeError("Failed to parse numeric PROJ output for center-based bbox conversion.") from exc
        return x, y


def _bbox_from_center_latlon(center_lat: float, center_lon: float, square_km: float) -> str:
    if square_km <= 0:
        raise ValueError("square_km must be > 0")
    center_x, center_y = _lonlat_to_epsg2180(center_lon, center_lat)
    side_km = math.sqrt(square_km)
    half_size_m = (side_km * 1000.0) / 2.0
    return (
        f"{center_x - half_size_m:.3f},"
        f"{center_y - half_size_m:.3f},"
        f"{center_x + half_size_m:.3f},"
        f"{center_y + half_size_m:.3f}"
    )


def _resolve_bbox_input(
    *,
    bbox: str | None,
    center_lat: float | None,
    center_lon: float | None,
    square_km: float | None,
    srs: str,
    required: bool,
) -> str | None:
    center_mode_supplied = any(value is not None for value in (center_lat, center_lon, square_km))
    if bbox is not None and center_mode_supplied:
        raise typer.BadParameter(
            "Provide either --bbox or center mode (--center-lat/--center-lon/--square-km), not both."
        )

    if center_mode_supplied:
        if center_lat is None or center_lon is None:
            raise typer.BadParameter("Center mode requires both --center-lat and --center-lon.")
        if srs.upper() != "EPSG:2180":
            raise typer.BadParameter("Center mode currently supports only --srs EPSG:2180.")
        effective_square_km = square_km if square_km is not None else DEFAULT_CENTER_SQUARE_KM
        try:
            return _bbox_from_center_latlon(center_lat, center_lon, effective_square_km)
        except RuntimeError as error:
            raise typer.BadParameter(str(error)) from error
        except ValueError as error:
            raise typer.BadParameter(str(error)) from error

    if required and bbox is None:
        raise typer.BadParameter(
            "bbox is required. Use --bbox xmin,ymin,xmax,ymax or center mode options."
        )
    return bbox


def _load_params_json_dict(params_json: Path) -> dict[str, object]:
    try:
        payload = json.loads(params_json.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        console.print(f"[red]Missing params JSON:[/red] {params_json}")
        raise typer.Exit(code=2) from error
    except json.JSONDecodeError as error:
        console.print(f"[red]Invalid JSON:[/red] {params_json} ({error})")
        raise typer.Exit(code=2) from error
    if not isinstance(payload, dict):
        console.print("[red]Invalid params JSON:[/red] top-level object must be a JSON object.")
        raise typer.Exit(code=2)
    return payload


def _as_optional_float(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as error:
            raise typer.BadParameter(f"{field_name} must be numeric") from error
    raise typer.BadParameter(f"{field_name} must be numeric")


def _resolve_json_center_bbox(payload: dict[str, object], *, required: bool) -> dict[str, object]:
    normalized = dict(payload)
    center_lat = _as_optional_float(normalized.get("center_lat"), "center_lat")
    center_lon = _as_optional_float(normalized.get("center_lon"), "center_lon")
    square_km = _as_optional_float(normalized.get("square_km"), "square_km")
    area_km2 = _as_optional_float(normalized.get("area_km2"), "area_km2")
    if square_km is not None and area_km2 is not None:
        raise typer.BadParameter("Use only one of square_km or area_km2 in JSON params.")
    effective_square_km = square_km if square_km is not None else area_km2
    bbox_value = normalized.get("bbox")
    bbox = str(bbox_value) if bbox_value is not None else None
    srs = str(normalized.get("srs", "EPSG:2180"))
    resolved_bbox = _resolve_bbox_input(
        bbox=bbox,
        center_lat=center_lat,
        center_lon=center_lon,
        square_km=effective_square_km,
        srs=srs,
        required=required,
    )
    normalized["bbox"] = resolved_bbox
    normalized["srs"] = srs
    normalized.pop("center_lat", None)
    normalized.pop("center_lon", None)
    normalized.pop("square_km", None)
    normalized.pop("area_km2", None)
    return normalized


def _slugify_location_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    lowered = ascii_only.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    slug = re.sub(r"_+", "_", slug)
    if not slug:
        raise typer.BadParameter(f"Cannot build slug from location_name={value!r}")
    return slug


def _apply_location_paths_policy(payload: dict[str, object], repo_root: Path) -> dict[str, object]:
    normalized = dict(payload)
    location_name = normalized.get("location_name")
    if location_name is None:
        return normalized
    slug = _slugify_location_name(str(location_name))
    normalized.setdefault("download_root", str(repo_root / f"downloads_{slug}"))
    normalized.setdefault("render_root", str(repo_root / f"rendered_{slug}"))
    normalized.setdefault("artifacts_dir", str(repo_root / f"artifacts_{slug}"))
    return normalized


def _build_run_config_from_base_and_location(*, base_json: Path, location_json: Path) -> RunConfig:
    base_payload = _load_params_json_dict(base_json)
    location_payload = _load_params_json_dict(location_json)
    merged: dict[str, object] = dict(base_payload)
    merged.update(location_payload)
    repo_root = base_json.resolve().parents[2] if len(base_json.resolve().parents) >= 3 else Path.cwd().resolve()
    merged = _apply_location_paths_policy(merged, repo_root)
    merged = _resolve_json_center_bbox(merged, required=True)
    return RunConfig.model_validate(merged)


def _build_index_config_from_base_and_location(*, base_json: Path, location_json: Path) -> IndexConfig:
    base_payload = _load_params_json_dict(base_json)
    location_payload = _load_params_json_dict(location_json)
    merged: dict[str, object] = dict(base_payload)
    merged.update(location_payload)
    repo_root = base_json.resolve().parents[2] if len(base_json.resolve().parents) >= 3 else Path.cwd().resolve()
    merged = _apply_location_paths_policy(merged, repo_root)
    merged = _resolve_json_center_bbox(merged, required=True)
    artifacts_dir = Path(str(merged.get("artifacts_dir")))
    merged.setdefault("output_json", str(artifacts_dir / "index_manifest.json"))
    merged.setdefault("year_availability_output_json", str(artifacts_dir / "year_availability_report.json"))
    return IndexConfig.model_validate(merged)


def _build_download_config_from_base_and_location(*, base_json: Path, location_json: Path) -> DownloadConfig:
    base_payload = _load_params_json_dict(base_json)
    location_payload = _load_params_json_dict(location_json)
    merged: dict[str, object] = dict(base_payload)
    merged.update(location_payload)
    repo_root = base_json.resolve().parents[2] if len(base_json.resolve().parents) >= 3 else Path.cwd().resolve()
    merged = _apply_location_paths_policy(merged, repo_root)
    merged = _resolve_json_center_bbox(merged, required=False)
    artifacts_dir = Path(str(merged.get("artifacts_dir")))
    merged.setdefault("index_manifest", str(artifacts_dir / "index_manifest.json"))
    merged.setdefault("output_json", str(artifacts_dir / "dataset_manifest_download.json"))
    return DownloadConfig.model_validate(merged)


def _build_validate_config_from_base_and_location(*, base_json: Path, location_json: Path) -> ValidateConfig:
    base_payload = _load_params_json_dict(base_json)
    location_payload = _load_params_json_dict(location_json)
    merged: dict[str, object] = dict(base_payload)
    merged.update(location_payload)
    repo_root = base_json.resolve().parents[2] if len(base_json.resolve().parents) >= 3 else Path.cwd().resolve()
    merged = _apply_location_paths_policy(merged, repo_root)
    artifacts_dir = Path(str(merged.get("artifacts_dir")))

    requested_years: list[int] = []
    if "requested_years" in merged:
        raw = merged.get("requested_years")
        if isinstance(raw, list):
            requested_years = [int(value) for value in raw]
    elif "year_start" in merged and "year_end" in merged:
        year_start = int(merged["year_start"])
        year_end = int(merged["year_end"])
        requested_years = list(range(year_start, year_end + 1))

    payload = {
        "dataset_manifest": str(merged.get("dataset_manifest", artifacts_dir / "dataset_manifest_render.json")),
        "requested_years": requested_years,
        "strict_years": bool(merged.get("strict_years", False)),
        "min_years": int(merged.get("min_years", 1)),
        "output_json": str(merged.get("validation_output_json", artifacts_dir / "validation_report.json")),
    }
    return ValidateConfig.model_validate(payload)


def _location_files_or_exit(locations_dir: Path) -> list[Path]:
    files = sorted(locations_dir.glob("*.json"))
    if not files:
        console.print(f"[red]No location JSON files found in:[/red] {locations_dir}")
        raise typer.Exit(code=2)
    return files


@app.command("index")
def index_command(
    year_start: int = typer.Option(..., help="First year (inclusive)."),
    year_end: int = typer.Option(..., help="Last year (inclusive)."),
    bbox: str | None = typer.Option(
        None, help="Bounding box: xmin,ymin,xmax,ymax in the provided SRS."
    ),
    center_lat: float | None = typer.Option(None, help="Center latitude (WGS84) for center-based bbox mode."),
    center_lon: float | None = typer.Option(None, help="Center longitude (WGS84) for center-based bbox mode."),
    square_km: float | None = typer.Option(
        None,
        min=0.0001,
        help="Square AOI area in km^2 for center-based bbox mode (default: 4.0 => 2km x 2km).",
    ),
    srs: str = typer.Option("EPSG:2180", help="Spatial reference system."),
    strict_years: bool = typer.Option(
        False, "--strict-years/--no-strict-years", help="Require all requested years."
    ),
    experimental_wfs_swap_bbox_axes: bool = typer.Option(
        False,
        "--experimental-wfs-swap-bbox-axes/--no-experimental-wfs-swap-bbox-axes",
        help="Deprecated legacy option. Swaps X/Y axis for WFS BBOX.",
    ),
    min_years: int = typer.Option(1, min=1, help="Minimum required available years."),
    output_json: Path = typer.Option(
        Path("artifacts/index_manifest.json"), help="Output manifest JSON path."
    ),
    year_availability_output_json: Path = typer.Option(
        Path("artifacts/year_availability_report.json"),
        help="Output year availability report JSON path.",
    ),
) -> None:
    try:
        resolved_bbox = _resolve_bbox_input(
            bbox=bbox,
            center_lat=center_lat,
            center_lon=center_lon,
            square_km=square_km,
            srs=srs,
            required=True,
        )
    except typer.BadParameter as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=2) from error

    try:
        config = IndexConfig(
            year_start=year_start,
            year_end=year_end,
            bbox=resolved_bbox or "",
            srs=srs,
            strict_years=strict_years,
            experimental_wfs_swap_bbox_axes=experimental_wfs_swap_bbox_axes,
            min_years=min_years,
            output_json=output_json,
            year_availability_output_json=year_availability_output_json,
        )
    except ValidationError as error:
        _print_validation_error(error)
        raise typer.Exit(code=2) from error

    try:
        exit_code, artifact_path = index_builder.run(config)
    except httpx.HTTPError as error:
        console.print(
            "[red]Index request failed due to transient server/network error.[/red] "
            "Retry command after a short delay."
        )
        console.print(f"[yellow]{error}[/yellow]")
        raise typer.Exit(code=1) from error
    if artifact_path.exists():
        try:
            manifest = IndexManifest.model_validate_json(artifact_path.read_text(encoding="utf-8"))
            if manifest.aoi_preview_html:
                console.print(f"[cyan]AOI preview HTML:[/cyan] {manifest.aoi_preview_html}")
        except Exception:
            pass
    _finish(exit_code, artifact_path)


@app.command("download")
def download_command(
    index_manifest: Path = typer.Option(
        Path("artifacts/index_manifest.json"), help="Path to index manifest JSON."
    ),
    download_root: Path = typer.Option(Path("downloads"), help="Directory for downloaded TIFF files."),
    mode: str = typer.Option(
        "hybrid",
        help="Pipeline mode: wms_tiled, wfs_render, hybrid.",
    ),
    profile: str = typer.Option("train", help="Pipeline profile: train or reference."),
    bbox: str | None = typer.Option(None, help="BBox xmin,ymin,xmax,ymax in the provided SRS. Required for reference profile."),
    center_lat: float | None = typer.Option(None, help="Center latitude (WGS84) for center-based bbox mode."),
    center_lon: float | None = typer.Option(None, help="Center longitude (WGS84) for center-based bbox mode."),
    square_km: float | None = typer.Option(
        None,
        min=0.0001,
        help="Square AOI area in km^2 for center-based bbox mode (default: 4.0 => 2km x 2km).",
    ),
    srs: str = typer.Option("EPSG:2180", help="Spatial reference system."),
    px_per_meter: float = typer.Option(15.0, min=0.0001, help="Pixels per meter for WMS fallback in reference mode."),
    wms_fallback_missing_years: bool = typer.Option(
        True,
        "--wms-fallback-missing-years/--no-wms-fallback-missing-years",
        help="Download WMS fallback images for years missing in WFS.",
    ),
    force_wms_year: list[int] | None = typer.Option(
        None,
        "--force-wms-year",
        help="Force selected year to use WMS source (repeat option).",
    ),
    concurrency: int = typer.Option(6, min=1, help="Number of parallel download workers."),
    retries: int = typer.Option(3, min=0, help="Retries per file."),
    retry_delay: float = typer.Option(1.0, min=0.01, help="Base retry delay in seconds."),
    timeout: float = typer.Option(120.0, min=1.0, help="HTTP timeout in seconds."),
    sleep_min: float = typer.Option(
        0.6, min=0.0, help="Random pre-request sleep minimum in seconds."
    ),
    sleep_max: float = typer.Option(
        2.2, min=0.0, help="Random pre-request sleep maximum in seconds."
    ),
    overwrite: bool = typer.Option(False, "--overwrite/--no-overwrite", help="Overwrite existing files."),
    output_json: Path = typer.Option(
        Path("artifacts/dataset_manifest_download.json"), help="Output dataset manifest JSON."
    ),
) -> None:
    try:
        resolved_bbox = _resolve_bbox_input(
            bbox=bbox,
            center_lat=center_lat,
            center_lon=center_lon,
            square_km=square_km,
            srs=srs,
            required=False,
        )
    except typer.BadParameter as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=2) from error

    try:
        config = DownloadConfig(
            index_manifest=index_manifest,
            download_root=download_root,
            mode=mode,
            profile=profile,
            bbox=resolved_bbox,
            srs=srs,
            px_per_meter=px_per_meter,
            wms_fallback_missing_years=wms_fallback_missing_years,
            force_wms_years=force_wms_year or [],
            concurrency=concurrency,
            retries=retries,
            retry_delay=retry_delay,
            timeout=timeout,
            sleep_min=sleep_min,
            sleep_max=sleep_max,
            overwrite=overwrite,
            output_json=output_json,
        )
    except ValidationError as error:
        _print_validation_error(error)
        raise typer.Exit(code=2) from error

    exit_code, artifact_path = downloader.run(config)
    _finish(exit_code, artifact_path)


@app.command("render")
def render_command(
    dataset_manifest: Path = typer.Option(
        Path("artifacts/dataset_manifest_download.json"), help="Path to dataset manifest JSON."
    ),
    render_root: Path = typer.Option(Path("rendered"), help="Directory for rendered yearly TIFF outputs."),
    mode: str = typer.Option(
        "hybrid",
        help="Pipeline mode: wms_tiled, wfs_render, hybrid.",
    ),
    profile: str = typer.Option("train", help="Render profile: train or reference."),
    px_per_meter: float = typer.Option(15.0, min=0.0001, help="Pixels per meter used in reference profile."),
    target_width: int | None = typer.Option(None, min=1, help="Target mosaic width (override)."),
    target_height: int | None = typer.Option(None, min=1, help="Target mosaic height (override)."),
    auto_size_from_bbox: bool = typer.Option(
        True,
        "--auto-size-from-bbox/--no-auto-size-from-bbox",
        help="Compute target size from bbox and px_per_meter when width/height override is not set.",
    ),
    target_bbox: str | None = typer.Option(
        None, help="Target bbox xmin,ymin,xmax,ymax. Defaults to index bbox."
    ),
    target_srs: str = typer.Option("EPSG:2180", help="Target CRS."),
    resample_method: str = typer.Option("bilinear", help="Resampling method: bilinear or nearest."),
    tile_size: int = typer.Option(512, min=64, help="TIFF internal tile size."),
    compression: str = typer.Option("deflate", help="Compression method."),
    overview_level: list[int] | None = typer.Option(
        None, "--overview-level", help="Overview decimation level (repeat option)."
    ),
    wms_fallback_missing_years: bool = typer.Option(
        True,
        "--wms-fallback-missing-years/--no-wms-fallback-missing-years",
        help="Treat WMS fallback years as valid render inputs in reference profile.",
    ),
    disable_color_norm: bool = typer.Option(
        False,
        "--disable-color-norm/--no-disable-color-norm",
        help="Disable per-year color normalization.",
    ),
    experimental_force_srgb_from_ycbcr: bool = typer.Option(
        False,
        "--experimental-force-srgb-from-ycbcr/--no-experimental-force-srgb-from-ycbcr",
        help="Experimental: force pyvips color conversion to sRGB before rendering.",
    ),
    experimental_per_year_color_norm: bool = typer.Option(
        False,
        "--experimental-per-year-color-norm/--no-experimental-per-year-color-norm",
        help="Experimental: apply per-year gray-world color normalization.",
    ),
    output_json: Path = typer.Option(
        Path("artifacts/dataset_manifest_render.json"), help="Output dataset manifest JSON."
    ),
) -> None:
    overview_levels = overview_level or [2, 4, 8, 16]
    try:
        config = RenderConfig(
            dataset_manifest=dataset_manifest,
            render_root=render_root,
            mode=mode,
            profile=profile,
            px_per_meter=px_per_meter,
            target_width=target_width,
            target_height=target_height,
            auto_size_from_bbox=auto_size_from_bbox,
            target_bbox=target_bbox,
            target_srs=target_srs,
            resample_method=resample_method,
            tile_size=tile_size,
            compression=compression,
            overview_levels=overview_levels,
            wms_fallback_missing_years=wms_fallback_missing_years,
            disable_color_norm=disable_color_norm,
            experimental_force_srgb_from_ycbcr=experimental_force_srgb_from_ycbcr,
            experimental_per_year_color_norm=experimental_per_year_color_norm,
            output_json=output_json,
        )
    except ValidationError as error:
        _print_validation_error(error)
        raise typer.Exit(code=2) from error

    exit_code, artifact_path = render.run(config)
    _finish(exit_code, artifact_path)


@app.command("mosaic")
def mosaic_alias_command(
    dataset_manifest: Path = typer.Option(
        Path("artifacts/dataset_manifest_download.json"), help="Path to dataset manifest JSON."
    ),
    output_json: Path = typer.Option(
        Path("artifacts/dataset_manifest_render.json"), help="Output dataset manifest JSON."
    ),
) -> None:
    """Backward compatible alias: maps old 'mosaic' command to new render stage."""
    try:
        config = RenderConfig(dataset_manifest=dataset_manifest, output_json=output_json)
    except ValidationError as error:
        _print_validation_error(error)
        raise typer.Exit(code=2) from error
    exit_code, artifact_path = render.run(config)
    _finish(exit_code, artifact_path)


@app.command("validate")
def validate_command(
    dataset_manifest: Path = typer.Option(
        Path("artifacts/dataset_manifest_render.json"), help="Path to dataset manifest JSON."
    ),
    year: list[int] | None = typer.Option(
        None, "--year", help="Requested year (repeat option for multiple years)."
    ),
    strict_years: bool = typer.Option(
        False, "--strict-years/--no-strict-years", help="Require all requested years."
    ),
    min_years: int = typer.Option(1, min=1, help="Minimum required available years."),
    output_json: Path = typer.Option(
        Path("artifacts/validation_report.json"), help="Output validation report JSON."
    ),
) -> None:
    requested_years = year or []
    try:
        config = ValidateConfig(
            dataset_manifest=dataset_manifest,
            requested_years=requested_years,
            strict_years=strict_years,
            min_years=min_years,
            output_json=output_json,
        )
    except ValidationError as error:
        _print_validation_error(error)
        raise typer.Exit(code=2) from error

    exit_code, artifact_path = validator.run(config)
    _finish(exit_code, artifact_path)


@app.command("run")
def run_command(
    year_start: int = typer.Option(..., help="First year (inclusive)."),
    year_end: int = typer.Option(..., help="Last year (inclusive)."),
    bbox: str | None = typer.Option(
        None, help="Bounding box: xmin,ymin,xmax,ymax in the provided SRS."
    ),
    center_lat: float | None = typer.Option(None, help="Center latitude (WGS84) for center-based bbox mode."),
    center_lon: float | None = typer.Option(None, help="Center longitude (WGS84) for center-based bbox mode."),
    square_km: float | None = typer.Option(
        None,
        min=0.0001,
        help="Square AOI area in km^2 for center-based bbox mode (default: 4.0 => 2km x 2km).",
    ),
    srs: str = typer.Option("EPSG:2180", help="Spatial reference system."),
    mode: str = typer.Option(
        "hybrid",
        help="Pipeline mode: wms_tiled, wfs_render, hybrid.",
    ),
    strict_years: bool = typer.Option(
        False, "--strict-years/--no-strict-years", help="Require all requested years."
    ),
    profile: str = typer.Option("train", help="Pipeline profile: train or reference."),
    px_per_meter: float = typer.Option(15.0, min=0.0001, help="Pixels per meter for reference profile."),
    wms_fallback_missing_years: bool = typer.Option(
        True,
        "--wms-fallback-missing-years/--no-wms-fallback-missing-years",
        help="Enable WMS fallback for years missing in WFS (reference profile).",
    ),
    force_wms_year: list[int] | None = typer.Option(
        None,
        "--force-wms-year",
        help="Force selected year to use WMS source (repeat option).",
    ),
    disable_color_norm: bool = typer.Option(
        False,
        "--disable-color-norm/--no-disable-color-norm",
        help="Disable per-year color normalization.",
    ),
    experimental_wfs_swap_bbox_axes: bool = typer.Option(
        False,
        "--experimental-wfs-swap-bbox-axes/--no-experimental-wfs-swap-bbox-axes",
        help="Deprecated legacy option. Swaps X/Y axis for WFS BBOX.",
    ),
    min_years: int = typer.Option(1, min=1, help="Minimum required available years."),
    target_width: int | None = typer.Option(None, min=1, help="Target mosaic width (override)."),
    target_height: int | None = typer.Option(None, min=1, help="Target mosaic height (override)."),
    auto_size_from_bbox: bool = typer.Option(
        True,
        "--auto-size-from-bbox/--no-auto-size-from-bbox",
        help="Compute target size from bbox and px_per_meter when width/height override is not set.",
    ),
    pixel_profile: str = typer.Option("RGB_U8", help="Pixel profile identifier."),
    render_root: Path = typer.Option(Path("rendered"), help="Directory for rendered yearly TIFF outputs."),
    target_bbox: str | None = typer.Option(
        None, help="Target bbox xmin,ymin,xmax,ymax. Defaults to index bbox."
    ),
    target_srs: str = typer.Option("EPSG:2180", help="Target CRS."),
    resample_method: str = typer.Option("bilinear", help="Resampling method: bilinear or nearest."),
    tile_size: int = typer.Option(512, min=64, help="TIFF internal tile size."),
    compression: str = typer.Option("deflate", help="Compression method."),
    overview_level: list[int] | None = typer.Option(
        None, "--overview-level", help="Overview decimation level (repeat option)."
    ),
    experimental_force_srgb_from_ycbcr: bool = typer.Option(
        False,
        "--experimental-force-srgb-from-ycbcr/--no-experimental-force-srgb-from-ycbcr",
        help="Experimental: force pyvips color conversion to sRGB before rendering.",
    ),
    experimental_per_year_color_norm: bool = typer.Option(
        False,
        "--experimental-per-year-color-norm/--no-experimental-per-year-color-norm",
        help="Experimental: apply per-year gray-world color normalization.",
    ),
    download_root: Path = typer.Option(Path("downloads"), help="Directory for downloaded TIFF files."),
    concurrency: int = typer.Option(6, min=1, help="Number of parallel download workers."),
    retries: int = typer.Option(3, min=0, help="Retries per file."),
    retry_delay: float = typer.Option(1.0, min=0.01, help="Base retry delay in seconds."),
    timeout: float = typer.Option(120.0, min=1.0, help="HTTP timeout in seconds."),
    sleep_min: float = typer.Option(0.6, min=0.0, help="Random pre-request sleep minimum."),
    sleep_max: float = typer.Option(2.2, min=0.0, help="Random pre-request sleep maximum."),
    overwrite: bool = typer.Option(False, "--overwrite/--no-overwrite", help="Overwrite existing files."),
    artifacts_dir: Path = typer.Option(Path("artifacts"), help="Directory for pipeline artifacts."),
) -> None:
    try:
        resolved_bbox = _resolve_bbox_input(
            bbox=bbox,
            center_lat=center_lat,
            center_lon=center_lon,
            square_km=square_km,
            srs=srs,
            required=True,
        )
    except typer.BadParameter as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=2) from error

    overview_levels = overview_level or [2, 4, 8, 16]
    try:
        config = RunConfig(
            year_start=year_start,
            year_end=year_end,
            bbox=resolved_bbox or "",
            srs=srs,
            mode=mode,
            strict_years=strict_years,
            profile=profile,
            px_per_meter=px_per_meter,
            wms_fallback_missing_years=wms_fallback_missing_years,
            force_wms_years=force_wms_year or [],
            disable_color_norm=disable_color_norm,
            experimental_wfs_swap_bbox_axes=experimental_wfs_swap_bbox_axes,
            min_years=min_years,
            target_width=target_width,
            target_height=target_height,
            auto_size_from_bbox=auto_size_from_bbox,
            pixel_profile=pixel_profile,
            render_root=render_root,
            target_bbox=target_bbox,
            target_srs=target_srs,
            resample_method=resample_method,
            tile_size=tile_size,
            compression=compression,
            overview_levels=overview_levels,
            experimental_force_srgb_from_ycbcr=experimental_force_srgb_from_ycbcr,
            experimental_per_year_color_norm=experimental_per_year_color_norm,
            download_root=download_root,
            concurrency=concurrency,
            retries=retries,
            retry_delay=retry_delay,
            timeout=timeout,
            sleep_min=sleep_min,
            sleep_max=sleep_max,
            overwrite=overwrite,
            artifacts_dir=artifacts_dir,
        )
    except ValidationError as error:
        _print_validation_error(error)
        raise typer.Exit(code=2) from error

    exit_code, artifact_path = run_all.run(config)
    _finish(exit_code, artifact_path)


@app.command("index-json")
def index_json_command(
    params_json: Path = typer.Argument(
        ...,
        help="Path to JSON file with IndexConfig fields. Supports center_lat/center_lon + square_km|area_km2.",
    ),
) -> None:
    try:
        payload = _load_params_json_dict(params_json)
        payload = _resolve_json_center_bbox(payload, required=True)
        config = IndexConfig.model_validate(payload)
    except typer.BadParameter as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=2) from error
    except ValidationError as error:
        _print_validation_error(error)
        raise typer.Exit(code=2) from error

    exit_code, artifact_path = index_builder.run(config)
    _finish(exit_code, artifact_path)


@app.command("download-json")
def download_json_command(
    params_json: Path = typer.Argument(
        ...,
        help="Path to JSON file with DownloadConfig fields. Supports center_lat/center_lon + square_km|area_km2.",
    ),
) -> None:
    try:
        payload = _load_params_json_dict(params_json)
        payload = _resolve_json_center_bbox(payload, required=False)
        config = DownloadConfig.model_validate(payload)
    except typer.BadParameter as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=2) from error
    except ValidationError as error:
        _print_validation_error(error)
        raise typer.Exit(code=2) from error

    exit_code, artifact_path = downloader.run(config)
    _finish(exit_code, artifact_path)


@app.command("render-json")
def render_json_command(
    params_json: Path = typer.Argument(..., help="Path to JSON file with RenderConfig fields."),
) -> None:
    try:
        config = RenderConfig.model_validate_json(params_json.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        console.print(f"[red]Missing params JSON:[/red] {params_json}")
        raise typer.Exit(code=2) from error
    except ValidationError as error:
        _print_validation_error(error)
        raise typer.Exit(code=2) from error

    exit_code, artifact_path = render.run(config)
    _finish(exit_code, artifact_path)


@app.command("validate-json")
def validate_json_command(
    params_json: Path = typer.Argument(..., help="Path to JSON file with ValidateConfig fields."),
) -> None:
    try:
        config = ValidateConfig.model_validate_json(params_json.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        console.print(f"[red]Missing params JSON:[/red] {params_json}")
        raise typer.Exit(code=2) from error
    except ValidationError as error:
        _print_validation_error(error)
        raise typer.Exit(code=2) from error

    exit_code, artifact_path = validator.run(config)
    _finish(exit_code, artifact_path)


@app.command("run-json")
def run_json_command(
    params_json: Path = typer.Argument(
        ...,
        help="Path to JSON file with RunConfig fields. Supports center_lat/center_lon + square_km|area_km2.",
    ),
) -> None:
    try:
        payload = _load_params_json_dict(params_json)
        payload = _resolve_json_center_bbox(payload, required=True)
        config = RunConfig.model_validate(payload)
    except typer.BadParameter as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=2) from error
    except ValidationError as error:
        _print_validation_error(error)
        raise typer.Exit(code=2) from error

    exit_code, artifact_path = run_all.run(config)
    _finish(exit_code, artifact_path)


@app.command("run-location-json")
def run_location_json_command(
    location_json: Path = typer.Argument(..., help="Path to location JSON (location_name, center_lat, center_lon)."),
    base_json: Path = typer.Option(
        Path("configs/run/base.json"),
        "--base-json",
        help="Path to base JSON with shared run parameters.",
    ),
) -> None:
    try:
        config = _build_run_config_from_base_and_location(base_json=base_json, location_json=location_json)
    except typer.BadParameter as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=2) from error
    except ValidationError as error:
        _print_validation_error(error)
        raise typer.Exit(code=2) from error

    exit_code, artifact_path = run_all.run(config)
    _finish(exit_code, artifact_path)


@app.command("run-all-location-json")
def run_all_location_json_command(
    locations_dir: Path = typer.Option(
        Path("configs/run/locations"),
        "--locations-dir",
        help="Directory with location JSON files.",
    ),
    base_json: Path = typer.Option(
        Path("configs/run/base.json"),
        "--base-json",
        help="Path to base JSON with shared run parameters.",
    ),
    continue_on_error: bool = typer.Option(
        False,
        "--continue-on-error/--no-continue-on-error",
        help="Continue with remaining locations when one location fails.",
    ),
) -> None:
    location_files = _location_files_or_exit(locations_dir)

    failures: list[str] = []
    for location_json in location_files:
        console.print(f"[cyan]run-all-location-json:[/cyan] {location_json}")
        try:
            config = _build_run_config_from_base_and_location(
                base_json=base_json,
                location_json=location_json,
            )
        except typer.BadParameter as error:
            console.print(f"[red]{error}[/red]")
            failures.append(f"{location_json}: {error}")
            if not continue_on_error:
                raise typer.Exit(code=2) from error
            continue
        except ValidationError as error:
            _print_validation_error(error)
            failures.append(f"{location_json}: validation_error")
            if not continue_on_error:
                raise typer.Exit(code=2) from error
            continue

        exit_code, artifact_path = run_all.run(config)
        console.print(str(artifact_path))
        if exit_code != 0:
            failures.append(f"{location_json}: exit={exit_code}")
            if not continue_on_error:
                raise typer.Exit(code=exit_code)

    if failures:
        console.print("[yellow]run-all-location-json finished with failures:[/yellow]")
        for entry in failures:
            console.print(f"- {entry}")
        raise typer.Exit(code=1)

    raise typer.Exit(code=0)


@app.command("index-all-location-json")
def index_all_location_json_command(
    locations_dir: Path = typer.Option(
        Path("configs/run/locations"),
        "--locations-dir",
        help="Directory with location JSON files.",
    ),
    base_json: Path = typer.Option(
        Path("configs/run/base.json"),
        "--base-json",
        help="Path to base JSON with shared parameters.",
    ),
    continue_on_error: bool = typer.Option(
        False,
        "--continue-on-error/--no-continue-on-error",
        help="Continue with remaining locations when one location fails.",
    ),
) -> None:
    location_files = _location_files_or_exit(locations_dir)
    failures: list[str] = []

    for location_json in location_files:
        console.print(f"[cyan]index-all-location-json:[/cyan] {location_json}")
        try:
            config = _build_index_config_from_base_and_location(
                base_json=base_json,
                location_json=location_json,
            )
        except (typer.BadParameter, ValidationError) as error:
            if isinstance(error, ValidationError):
                _print_validation_error(error)
                message = "validation_error"
            else:
                console.print(f"[red]{error}[/red]")
                message = str(error)
            failures.append(f"{location_json}: {message}")
            if not continue_on_error:
                raise typer.Exit(code=2) from error
            continue

        exit_code, artifact_path = index_builder.run(config)
        console.print(str(artifact_path))
        if exit_code != 0:
            failures.append(f"{location_json}: exit={exit_code}")
            if not continue_on_error:
                raise typer.Exit(code=exit_code)

    if failures:
        console.print("[yellow]index-all-location-json finished with failures:[/yellow]")
        for entry in failures:
            console.print(f"- {entry}")
        raise typer.Exit(code=1)
    raise typer.Exit(code=0)


@app.command("download-all-location-json")
def download_all_location_json_command(
    locations_dir: Path = typer.Option(
        Path("configs/run/locations"),
        "--locations-dir",
        help="Directory with location JSON files.",
    ),
    base_json: Path = typer.Option(
        Path("configs/run/base.json"),
        "--base-json",
        help="Path to base JSON with shared parameters.",
    ),
    continue_on_error: bool = typer.Option(
        False,
        "--continue-on-error/--no-continue-on-error",
        help="Continue with remaining locations when one location fails.",
    ),
) -> None:
    location_files = _location_files_or_exit(locations_dir)
    failures: list[str] = []

    for location_json in location_files:
        console.print(f"[cyan]download-all-location-json:[/cyan] {location_json}")
        try:
            config = _build_download_config_from_base_and_location(
                base_json=base_json,
                location_json=location_json,
            )
        except (typer.BadParameter, ValidationError) as error:
            if isinstance(error, ValidationError):
                _print_validation_error(error)
                message = "validation_error"
            else:
                console.print(f"[red]{error}[/red]")
                message = str(error)
            failures.append(f"{location_json}: {message}")
            if not continue_on_error:
                raise typer.Exit(code=2) from error
            continue

        exit_code, artifact_path = downloader.run(config)
        console.print(str(artifact_path))
        if exit_code != 0:
            failures.append(f"{location_json}: exit={exit_code}")
            if not continue_on_error:
                raise typer.Exit(code=exit_code)

    if failures:
        console.print("[yellow]download-all-location-json finished with failures:[/yellow]")
        for entry in failures:
            console.print(f"- {entry}")
        raise typer.Exit(code=1)
    raise typer.Exit(code=0)


@app.command("validate-all-location-json")
def validate_all_location_json_command(
    locations_dir: Path = typer.Option(
        Path("configs/run/locations"),
        "--locations-dir",
        help="Directory with location JSON files.",
    ),
    base_json: Path = typer.Option(
        Path("configs/run/base.json"),
        "--base-json",
        help="Path to base JSON with shared parameters.",
    ),
    continue_on_error: bool = typer.Option(
        False,
        "--continue-on-error/--no-continue-on-error",
        help="Continue with remaining locations when one location fails.",
    ),
) -> None:
    location_files = _location_files_or_exit(locations_dir)
    failures: list[str] = []

    for location_json in location_files:
        console.print(f"[cyan]validate-all-location-json:[/cyan] {location_json}")
        try:
            config = _build_validate_config_from_base_and_location(
                base_json=base_json,
                location_json=location_json,
            )
        except (typer.BadParameter, ValidationError) as error:
            if isinstance(error, ValidationError):
                _print_validation_error(error)
                message = "validation_error"
            else:
                console.print(f"[red]{error}[/red]")
                message = str(error)
            failures.append(f"{location_json}: {message}")
            if not continue_on_error:
                raise typer.Exit(code=2) from error
            continue

        exit_code, artifact_path = validator.run(config)
        console.print(str(artifact_path))
        if exit_code != 0:
            failures.append(f"{location_json}: exit={exit_code}")
            if not continue_on_error:
                raise typer.Exit(code=exit_code)

    if failures:
        console.print("[yellow]validate-all-location-json finished with failures:[/yellow]")
        for entry in failures:
            console.print(f"- {entry}")
        raise typer.Exit(code=1)
    raise typer.Exit(code=0)


def main() -> None:
    configure_logging("INFO")
    app()


if __name__ == "__main__":
    main()
