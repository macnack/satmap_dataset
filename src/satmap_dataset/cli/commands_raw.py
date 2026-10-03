from __future__ import annotations

from pathlib import Path

import typer
from pydantic import ValidationError

from satmap_dataset.cli.app import app
from satmap_dataset.cli.common import (
    _finish,
    _load_params_json_dict,
    _print_validation_error,
    console,
)
from satmap_dataset.cli.config_builders import _build_raw_export_config_from_base_and_location
from satmap_dataset.cli.registry import StageCommandSpec, register_stage_commands
from satmap_dataset.config import RawExportConfig, _default_raw_root
from satmap_dataset.pipeline import raw_export
from satmap_dataset.raw_tiles.split_manifest import build_test_manifest



@app.command("raw-export")
def raw_export_command(
    provider: str = typer.Option("geoportal", help="geoportal|lantmateriet|nls (sentinel2 rejected)."),
    area: str = typer.Option(..., help="Area slug (output namespace under <raw_root>/<provider>/)."),
    download_root: Path = typer.Option(..., help="downloads_<slug> root with <year>/*.tif."),
    raw_root: Path | None = typer.Option(None, help="Shared sat_data_raw root (default: $SATMAP_RAW_ROOT or ~/sat_data_raw)."),
    download_manifest: Path | None = typer.Option(None, help="Optional download manifest for provenance."),
    min_coverage: float | None = typer.Option(None, help="Override per-provider coverage gate (0,1]."),
    link_mode: str = typer.Option("symlink", help="symlink|copy for exported native tiles."),
    cell_mode: str = typer.Option("footprint", help="footprint (verbatim) | world_window (co-register mixed-GSD years to one equal-dim stack)."),
    equalize_gsd: bool = typer.Option(True, "--equalize-gsd/--raw-gsd", help="world_window: resample years to coarsest GSD (equal-dim) or keep native GSD (raw, lossless, mixed dims)."),
    cell_size_m: float | None = typer.Option(None, help="Override cell size in metres."),
    artifacts_dir: Path = typer.Option(Path("artifacts"), help="Where raw_export_manifest.json is written."),
    output_json: Path | None = typer.Option(None, help="Stage artifact path."),
) -> None:
    payload: dict[str, object] = {
        "provider": provider, "area": area, "download_root": str(download_root),
        "link_mode": link_mode, "cell_mode": cell_mode, "equalize_gsd": equalize_gsd,
        "artifacts_dir": str(artifacts_dir),
        "output_json": str(output_json) if output_json else str(artifacts_dir / "raw_export_manifest.json"),
    }
    if raw_root is not None:
        payload["raw_root"] = str(raw_root)
    if download_manifest is not None:
        payload["download_manifest"] = str(download_manifest)
    if min_coverage is not None:
        payload["min_coverage"] = min_coverage
    if cell_size_m is not None:
        payload["cell_size_m"] = cell_size_m
    try:
        config = RawExportConfig.model_validate(payload)
    except ValidationError as error:
        _print_validation_error(error)
        raise typer.Exit(code=2) from error
    exit_code, artifact_path = raw_export.run(config)
    _finish(exit_code, artifact_path)

@app.command("raw-export-all-location-json")
def raw_export_all_location_json_command(
    locations_dir: Path = typer.Option(Path("configs/run/locations"), "--locations-dir"),
    base_json: Path = typer.Option(Path("configs/run/base.json"), "--base-json"),
    continue_on_error: bool = typer.Option(False, "--continue-on-error"),
) -> None:
    location_files = sorted(locations_dir.glob("*.json"))
    if not location_files:
        console.print(f"[red]No location JSONs under {locations_dir}[/red]")
        raise typer.Exit(code=2)
    last_path: Path | None = None
    failures = 0
    for loc in location_files:
        try:
            config = _build_raw_export_config_from_base_and_location(base_json=base_json, location_json=loc)
            exit_code, last_path = raw_export.run(config)
            if exit_code != 0:
                failures += 1
                if not continue_on_error:
                    raise typer.Exit(code=1)
        except (typer.BadParameter, ValidationError) as error:
            failures += 1
            console.print(f"[red]{loc.name}: {error}[/red]")
            if not continue_on_error:
                raise typer.Exit(code=2) from error
    if last_path is not None:
        typer.echo(str(last_path))
    raise typer.Exit(code=1 if failures else 0)

@app.command("raw-test-manifest")
def raw_test_manifest_command(
    raw_root: Path | None = typer.Option(None, help="Shared sat_data_raw root (default: $SATMAP_RAW_ROOT or ~/sat_data_raw)."),
    out: Path | None = typer.Option(None, help="Output split manifest path (default: <raw_root>/test_manifest.yaml)."),
    min_years: int = typer.Option(2, help="Minimum seasons per kept cell."),
) -> None:
    root = Path(raw_root) if raw_root is not None else _default_raw_root()
    out_path = Path(out) if out is not None else root / "test_manifest.yaml"
    import sys
    cli_mod = sys.modules.get("satmap_dataset.cli")
    builder = (
        getattr(cli_mod, "build_test_manifest", build_test_manifest)
        if cli_mod is not None
        else build_test_manifest
    )
    builder(root, out_path, min_years=min_years)
    typer.echo(str(out_path.resolve()))



def register(app: typer.Typer) -> None:
    register_stage_commands(
        app,
        [
            StageCommandSpec(
                name="raw-export",
                config_cls=RawExportConfig,
                runner=lambda c: raw_export.run(c),
                json_help="JSON file with RawExportConfig fields.",
                resolve_bbox="none",
                build_location=_build_raw_export_config_from_base_and_location,
                location_help="Location JSON (location_name, center_lat, center_lon).",
                flavors=frozenset({"json", "location-json"}),
            ),
        ],
    )
