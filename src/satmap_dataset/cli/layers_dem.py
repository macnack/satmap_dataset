from __future__ import annotations

from pathlib import Path

import typer
from pydantic import ValidationError

from satmap_dataset.cli.app import app
from satmap_dataset.cli.common import (
    _finish,
    _print_availability_table,
    _print_validation_error,
    _resolve_json_center_bbox,
    console,
)
from satmap_dataset.cli.config_builders import (
    _build_dem_availability_config_from_base_and_location,
    _build_dem_config_from_base_and_location,
)
from satmap_dataset.cli.registry import StageCommandSpec, register_stage_commands
from satmap_dataset.config import DemAvailabilityConfig, DemConfig
from satmap_dataset.models import DemAvailabilityReport
from satmap_dataset.pipeline import dem, dem_availability


@app.command("dem")
def dem_command(
    bbox: str = typer.Option(None, "--bbox", help="xmin,ymin,xmax,ymax in --srs."),
    srs: str = typer.Option("EPSG:2180", "--srs"),
    transport: str = typer.Option("wcs", "--transport", help="wcs (current composite) or skorowidz (historical per-year)."),
    year_start: int = typer.Option(None, "--year-start", help="First year (skorowidz transport)."),
    year_end: int = typer.Option(None, "--year-end", help="Last year (skorowidz transport)."),
    center_lat: float = typer.Option(None, "--center-lat"),
    center_lon: float = typer.Option(None, "--center-lon"),
    square_km: float = typer.Option(None, "--square-km"),
    products: str = typer.Option("nmt,nmpt", "--products", help="Comma-separated subset of nmt,nmpt."),
    vertical_datum: str = typer.Option("evrf2007", "--vertical-datum", help="evrf2007 or kron86."),
    dem_root: Path = typer.Option(Path("dem"), "--dem-root"),
    align_to_render: bool = typer.Option(True, "--align/--no-align"),
    render_manifest: Path = typer.Option(None, "--render-manifest"),
    max_request_px: int = typer.Option(2048, "--max-request-px"),
    overwrite: bool = typer.Option(False, "--overwrite"),
    output_json: Path = typer.Option(None, "--output-json"),
) -> None:
    try:
        payload: dict[str, object] = {
            "bbox": bbox,
            "srs": srs,
            "transport": transport,
            "year_start": year_start,
            "year_end": year_end,
            "center_lat": center_lat,
            "center_lon": center_lon,
            "square_km": square_km,
            "products": [p.strip() for p in products.split(",") if p.strip()],
            "vertical_datum": vertical_datum,
            "dem_root": str(dem_root),
            "align_to_render": align_to_render,
            "max_request_px": max_request_px,
            "overwrite": overwrite,
        }
        if render_manifest is not None:
            payload["render_manifest"] = str(render_manifest)
        payload["output_json"] = str(output_json) if output_json is not None else str(dem_root / "dem_manifest.json")
        payload = _resolve_json_center_bbox(payload, required=True)
        config = DemConfig.model_validate(payload)
    except typer.BadParameter as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=2) from error
    except ValidationError as error:
        _print_validation_error(error)
        raise typer.Exit(code=2) from error

    exit_code, artifact_path = dem.run(config)
    _finish(exit_code, artifact_path)



def _dem_availability_after_run(exit_code: int, artifact_path: Path) -> None:
    _print_availability_table(
        DemAvailabilityReport.model_validate_json(artifact_path.read_text(encoding="utf-8"))
    )


def register(app: typer.Typer) -> None:
    register_stage_commands(
        app,
        [
            StageCommandSpec(
                name="dem",
                config_cls=DemConfig,
                runner=lambda c: dem.run(c),
                json_help="Path to JSON file with DemConfig fields. Supports center_lat/center_lon + square_km|area_km2.",
                resolve_bbox="required",
                build_location=_build_dem_config_from_base_and_location,
                error_style="separate",
                all_label="dem-all-location-json",
            ),
            StageCommandSpec(
                name="dem-availability",
                config_cls=DemAvailabilityConfig,
                runner=lambda c: dem_availability.run(c),
                json_help="JSON with DemAvailabilityConfig fields (center_lat/lon + square_km|area_km2 supported).",
                resolve_bbox="required",
                build_location=_build_dem_availability_config_from_base_and_location,
                location_help="Location JSON (location_name, center_lat, center_lon).",
                after_run=_dem_availability_after_run,
                flavors=frozenset({"json", "location-json", "all-location-json"}),
                all_label="dem-availability",
                all_failure_header="dem-availability-all finished with failures:",
                error_style="invalid",
            ),
        ],
    )
