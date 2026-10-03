from __future__ import annotations

from pathlib import Path

import typer
from pydantic import ValidationError

from satmap_dataset.cli.app import app
from satmap_dataset.cli.common import (
    _finish,
    _print_validation_error,
    _resolve_json_center_bbox,
    console,
)
from satmap_dataset.cli.config_builders import _build_osm_config_from_base_and_location
from satmap_dataset.cli.registry import StageCommandSpec, register_stage_commands
from satmap_dataset.config import OsmConfig
from satmap_dataset.pipeline import osm as osm_pipeline


@app.command("osm")
def osm_command(
    bbox: str = typer.Option(None, "--bbox", help="xmin,ymin,xmax,ymax in --srs."),
    srs: str = typer.Option("EPSG:2180", "--srs"),
    center_lat: float = typer.Option(None, "--center-lat"),
    center_lon: float = typer.Option(None, "--center-lon"),
    square_km: float = typer.Option(None, "--square-km"),
    categories: str = typer.Option(
        "buildings,highways,landuse,water", "--categories",
        help="Comma-separated subset of buildings,highways,landuse,water.",
    ),
    render_manifest: Path = typer.Option(None, "--render-manifest"),
    osm_root: Path = typer.Option(Path("osm"), "--osm-root"),
    overwrite: bool = typer.Option(False, "--overwrite"),
    output_json: Path = typer.Option(None, "--output-json"),
) -> None:
    try:
        payload: dict[str, object] = {
            "bbox": bbox,
            "srs": srs,
            "center_lat": center_lat,
            "center_lon": center_lon,
            "square_km": square_km,
            "categories": [c.strip() for c in categories.split(",") if c.strip()],
            "osm_root": str(osm_root),
            "overwrite": overwrite,
        }
        if render_manifest is not None:
            payload["render_manifest"] = str(render_manifest)
        payload["output_json"] = (
            str(output_json) if output_json is not None else str(osm_root / "osm_manifest.json")
        )
        payload = _resolve_json_center_bbox(payload, required=True)
        config = OsmConfig.model_validate(payload)
    except typer.BadParameter as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=2) from error
    except ValidationError as error:
        _print_validation_error(error)
        raise typer.Exit(code=2) from error

    exit_code, artifact_path = osm_pipeline.run(config)
    _finish(exit_code, artifact_path)



def register(app: typer.Typer) -> None:
    register_stage_commands(
        app,
        [
            StageCommandSpec(
                name="osm",
                config_cls=OsmConfig,
                runner=lambda c: osm_pipeline.run(c),
                json_help="Path to JSON file with OsmConfig fields.",
                resolve_bbox="required",
                build_location=_build_osm_config_from_base_and_location,
                location_help="Path to location JSON.",
                error_style="combined",
                all_label="osm-all-location-json",
            ),
        ],
    )
