from __future__ import annotations

from pathlib import Path

import httpx
import typer
from pydantic import ValidationError

from satmap_dataset.cli.app import app
from satmap_dataset.cli.common import (
    _finish,
    _has_successful_validation_artifact,
    _location_files_or_exit,
    _print_validation_error,
    _resolve_bbox_input,
    console,
)
from satmap_dataset.cli.config_builders import (
    _build_dem_config_from_base_and_location,
    _build_download_config_from_base_and_location,
    _build_index_config_from_base_and_location,
    _build_osm_config_from_base_and_location,
    _build_render_config_from_base_and_location,
    _build_run_config_from_base_and_location,
    _build_validate_config_from_base_and_location,
)
from satmap_dataset.cli.registry import StageCommandSpec, register_stage_commands
from satmap_dataset.config import DownloadConfig, IndexConfig, RenderConfig, RunConfig, ValidateConfig
from satmap_dataset.models import IndexManifest
from satmap_dataset.pipeline import location_run, render, run_all, validator
from satmap_dataset.providers import get_provider


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
    provider: str = typer.Option(
        "geoportal",
        "--provider",
        help="Data provider: geoportal (Polish PZGiK) or lantmateriet (Sweden STAC).",
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
            provider=provider,
        )
    except ValidationError as error:
        _print_validation_error(error)
        raise typer.Exit(code=2) from error

    try:
        exit_code, artifact_path = get_provider(config.provider).index(config)
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
            if getattr(manifest, "aoi_preview_png", None):
                console.print(f"[cyan]AOI preview PNG:[/cyan] {manifest.aoi_preview_png}")
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
    provider: str = typer.Option(
        "geoportal",
        "--provider",
        help="Data provider: geoportal (Polish PZGiK) or lantmateriet (Sweden STAC).",
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
            provider=provider,
        )
    except ValidationError as error:
        _print_validation_error(error)
        raise typer.Exit(code=2) from error

    exit_code, artifact_path = get_provider(config.provider).download(config)
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
    provider: str = typer.Option(
        "geoportal",
        "--provider",
        help="Data provider: geoportal (Polish PZGiK) or lantmateriet (Sweden STAC).",
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
            provider=provider,
        )
    except ValidationError as error:
        _print_validation_error(error)
        raise typer.Exit(code=2) from error

    exit_code, artifact_path = run_all.run(config)
    _finish(exit_code, artifact_path)


@app.command("location-run-json")
def location_run_json_command(
    location_json: Path = typer.Argument(..., help="Path to location JSON (location_name, center_lat, center_lon)."),
    base_json: Path = typer.Option(
        Path("configs/run/base.json"),
        "--base-json",
        help="Path to base JSON with shared run parameters.",
    ),
    run_dem: bool = typer.Option(
        True,
        "--dem/--no-dem",
        show_default=False,
        help="Produce the DEM layer aligned to the RGB grid (default: on).",
    ),
    run_osm: bool = typer.Option(
        True,
        "--osm/--no-osm",
        show_default=False,
        help="Produce the OSM label layer aligned to the RGB grid (default: on).",
    ),
    validate: bool = typer.Option(
        True,
        "--validate/--no-validate",
        show_default=False,
        help="Run the validator on the RGB layer manifest (default: on).",
    ),
) -> None:
    """Produce the aligned RGB + DEM + OSM stack for one location in one pass.

    Unlike ``run-location-json`` (RGB only), this drives the layer orchestrator:
    the RGB layer defines the shared ReferenceGrid and the DEM/OSM layers align
    to it in memory (no re-reading a render manifest from disk).
    """
    try:
        rgb_config = _build_run_config_from_base_and_location(
            base_json=base_json, location_json=location_json
        )
        dem_config = (
            _build_dem_config_from_base_and_location(
                base_json=base_json, location_json=location_json
            )
            if run_dem
            else None
        )
        osm_config = (
            _build_osm_config_from_base_and_location(
                base_json=base_json, location_json=location_json
            )
            if run_osm
            else None
        )
    except typer.BadParameter as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=2) from error
    except ValidationError as error:
        _print_validation_error(error)
        raise typer.Exit(code=2) from error

    exit_code, artifact_path = location_run.run_location(
        rgb_config=rgb_config,
        dem_config=dem_config,
        osm_config=osm_config,
        artifacts_dir=rgb_config.artifacts_dir,
        run_dem=run_dem,
        run_osm=run_osm,
        validate=validate,
    )
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

        if _has_successful_validation_artifact(config.artifacts_dir):
            console.print(f"[green]skip existing artifact:[/green] {config.artifacts_dir / 'validation_report.json'}")
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



def register(app: typer.Typer) -> None:
    """Register registry-driven JSON/location flavors for RGB pipeline stages."""
    specs = [
        StageCommandSpec(
            name="index",
            config_cls=IndexConfig,
            runner=lambda c: get_provider(c.provider).index(c),
            json_help="Path to JSON file with IndexConfig fields. Supports center_lat/center_lon + square_km|area_km2.",
            resolve_bbox="required",
            build_location=_build_index_config_from_base_and_location,
            flavors=frozenset({"json", "all-location-json"}),
            error_style="index",
        ),
        StageCommandSpec(
            name="download",
            config_cls=DownloadConfig,
            runner=lambda c: get_provider(c.provider).download(c),
            json_help="Path to JSON file with DownloadConfig fields. Supports center_lat/center_lon + square_km|area_km2.",
            resolve_bbox="optional",
            build_location=_build_download_config_from_base_and_location,
            flavors=frozenset({"json", "all-location-json"}),
            error_style="index",
        ),
        StageCommandSpec(
            name="render",
            config_cls=RenderConfig,
            runner=lambda c: render.run(c),
            json_help="Path to JSON file with RenderConfig fields.",
            resolve_bbox="none",
            raw_json_validate=True,
            build_location=_build_render_config_from_base_and_location,
            flavors=frozenset({"json", "location-json"}),
        ),
        StageCommandSpec(
            name="validate",
            config_cls=ValidateConfig,
            runner=lambda c: validator.run(c),
            json_help="Path to JSON file with ValidateConfig fields.",
            resolve_bbox="none",
            raw_json_validate=True,
            build_location=_build_validate_config_from_base_and_location,
            flavors=frozenset({"json", "all-location-json"}),
            error_style="index",
        ),
        StageCommandSpec(
            name="run",
            config_cls=RunConfig,
            runner=lambda c: run_all.run(c),
            json_help="Path to JSON file with RunConfig fields. Supports center_lat/center_lon + square_km|area_km2.",
            resolve_bbox="required",
            build_location=_build_run_config_from_base_and_location,
            flavors=frozenset({"json", "location-json"}),
        ),
    ]
    register_stage_commands(app, specs)
