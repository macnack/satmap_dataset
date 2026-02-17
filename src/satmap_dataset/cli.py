from __future__ import annotations

from pathlib import Path

import typer
from pydantic import ValidationError
from rich.console import Console
import httpx

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


def _print_validation_error(error: ValidationError) -> None:
    console.print("[red]Invalid configuration:[/red]")
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"])
        console.print(f"- {location}: {item['msg']}")


def _finish(exit_code: int, artifact_path: Path) -> None:
    typer.echo(str(artifact_path))
    raise typer.Exit(code=exit_code)


@app.command("index")
def index_command(
    year_start: int = typer.Option(..., help="First year (inclusive)."),
    year_end: int = typer.Option(..., help="Last year (inclusive)."),
    bbox: str = typer.Option(..., help="Bounding box: xmin,ymin,xmax,ymax in the provided SRS."),
    srs: str = typer.Option("EPSG:2180", help="Spatial reference system."),
    strict_years: bool = typer.Option(
        False, "--strict-years/--no-strict-years", help="Require all requested years."
    ),
    experimental_wfs_swap_bbox_axes: bool = typer.Option(
        False,
        "--experimental-wfs-swap-bbox-axes/--no-experimental-wfs-swap-bbox-axes",
        help="Deprecated legacy option: swaps X/Y axis for WFS BBOX.",
    ),
    min_years: int = typer.Option(2, min=1, help="Minimum required available years."),
    output_json: Path = typer.Option(
        Path("artifacts/index_manifest.json"), help="Output manifest JSON path."
    ),
    year_availability_output_json: Path = typer.Option(
        Path("artifacts/year_availability_report.json"),
        help="Output year availability report JSON path.",
    ),
) -> None:
    try:
        config = IndexConfig(
            year_start=year_start,
            year_end=year_end,
            bbox=bbox,
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
    srs: str = typer.Option("EPSG:2180", help="Spatial reference system."),
    px_per_meter: float = typer.Option(15.0, min=0.0001, help="Pixels per meter for WMS fallback in reference mode."),
    wms_fallback_missing_years: bool = typer.Option(
        True,
        "--wms-fallback-missing-years/--no-wms-fallback-missing-years",
        help="Download WMS fallback images for years missing in WFS.",
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
        config = DownloadConfig(
            index_manifest=index_manifest,
            download_root=download_root,
            mode=mode,
            profile=profile,
            bbox=bbox,
            srs=srs,
            px_per_meter=px_per_meter,
            wms_fallback_missing_years=wms_fallback_missing_years,
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
    wfs_global_calibration: bool = typer.Option(
        False,
        "--wfs-global-calibration/--no-wfs-global-calibration",
        help="Apply one global WFS->WMS calibration transform for all WFS years (reference profile).",
    ),
    skip_calibration: bool = typer.Option(
        False,
        "--skip-calibration/--no-skip-calibration",
        help="Shortcut alias: disable global WFS calibration for this render run.",
    ),
    calibration_reference_year: int | None = typer.Option(
        None,
        help="Optional reference year used to estimate global WFS->WMS calibration.",
    ),
    calibration_transform: str = typer.Option(
        "affine",
        help="Calibration transform model: affine or homography.",
    ),
    calibration_max_error_px: float = typer.Option(
        2.0,
        min=0.0001,
        help="Maximum accepted global calibration residual error in pixels.",
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
    effective_wfs_global_calibration = wfs_global_calibration and (not skip_calibration)
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
            wfs_global_calibration=effective_wfs_global_calibration,
            calibration_reference_year=calibration_reference_year,
            calibration_transform=calibration_transform,
            calibration_max_error_px=calibration_max_error_px,
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
    min_years: int = typer.Option(2, min=1, help="Minimum required available years."),
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
    bbox: str = typer.Option(..., help="Bounding box: xmin,ymin,xmax,ymax in the provided SRS."),
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
    wfs_global_calibration: bool = typer.Option(
        False,
        "--wfs-global-calibration/--no-wfs-global-calibration",
        help="Apply one global WFS->WMS calibration transform for all WFS years (reference profile).",
    ),
    skip_calibration: bool = typer.Option(
        False,
        "--skip-calibration/--no-skip-calibration",
        help="Shortcut alias: disable global WFS calibration for this run.",
    ),
    calibration_reference_year: int | None = typer.Option(
        None,
        help="Optional reference year used to estimate global WFS->WMS calibration.",
    ),
    calibration_transform: str = typer.Option(
        "affine",
        help="Calibration transform model: affine or homography.",
    ),
    calibration_max_error_px: float = typer.Option(
        2.0,
        min=0.0001,
        help="Maximum accepted global calibration residual error in pixels.",
    ),
    disable_color_norm: bool = typer.Option(
        False,
        "--disable-color-norm/--no-disable-color-norm",
        help="Disable per-year color normalization.",
    ),
    experimental_wfs_swap_bbox_axes: bool = typer.Option(
        False,
        "--experimental-wfs-swap-bbox-axes/--no-experimental-wfs-swap-bbox-axes",
        help="Deprecated legacy option: swaps X/Y axis for WFS BBOX.",
    ),
    min_years: int = typer.Option(2, min=1, help="Minimum required available years."),
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
    overview_levels = overview_level or [2, 4, 8, 16]
    effective_wfs_global_calibration = wfs_global_calibration and (not skip_calibration)
    try:
        config = RunConfig(
            year_start=year_start,
            year_end=year_end,
            bbox=bbox,
            srs=srs,
            mode=mode,
            strict_years=strict_years,
            profile=profile,
            px_per_meter=px_per_meter,
            wms_fallback_missing_years=wms_fallback_missing_years,
            wfs_global_calibration=effective_wfs_global_calibration,
            calibration_reference_year=calibration_reference_year,
            calibration_transform=calibration_transform,
            calibration_max_error_px=calibration_max_error_px,
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


def main() -> None:
    configure_logging("INFO")
    app()


if __name__ == "__main__":
    main()
