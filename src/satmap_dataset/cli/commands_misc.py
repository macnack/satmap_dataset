from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from satmap_dataset.cli.app import app
from satmap_dataset.cli.common import (
    _finish,
    _format_compact_float,
    _format_years_list,
    _load_available_years_from_artifacts,
    _load_params_json_dict,
    _location_files_or_exit,
    _manifest_checkpoint,
    _print_validation_error,
    _slugify_location_name,
    _years_label,
    console,
)
from satmap_dataset.config import DownloadConfig, IndexConfig
from satmap_dataset.logging_utils import configure_logging
from satmap_dataset.providers import get_provider


def _nls_force_provider(payload: dict) -> dict:
    """Pin provider='nls' on the payload so the NLS-specific config validators run.

    Without this, a config that omits `provider` (or sets `provider='geoportal'`)
    would skip the EPSG:3067 srs guard even though we're about to run NlsProvider.
    """
    payload = dict(payload)
    payload["provider"] = "nls"
    return payload


def _nls_build_download_config(payload: dict, index_manifest_path: Path) -> DownloadConfig:
    """Construct a DownloadConfig from an NLS run/index JSON payload.

    Treats payload['output_json'] as the *index* manifest path (matching the
    SIVL config naming convention) and writes the download manifest as a
    sibling. Without this, DownloadConfig.output_json would shadow
    DownloadConfig.index_manifest and the download stage would read
    artifacts/index_manifest.json instead of the file the index step wrote.
    """
    download_payload = {k: v for k, v in payload.items() if k in DownloadConfig.model_fields}
    download_payload["index_manifest"] = str(index_manifest_path)
    download_payload["output_json"] = str(
        index_manifest_path.parent / "dataset_manifest_download.json"
    )
    download_payload.setdefault("provider", "nls")
    download_payload.setdefault("provider_options", payload.get("provider_options", {}))
    download_payload.setdefault("bbox", payload.get("bbox"))
    download_payload.setdefault("srs", payload.get("srs", "EPSG:3067"))
    return DownloadConfig(**download_payload)




@app.command("summary-locations")
def summary_locations_command(
    locations_dir: Path = typer.Argument(..., help="Directory with location JSON files."),
    base_json: Path = typer.Option(
        Path("configs/run/base.json"),
        "--base-json",
        help="Optional base JSON merged with every location for shared defaults.",
    ),
) -> None:
    base_payload: dict[str, object] = {}
    if base_json.exists():
        base_payload = _load_params_json_dict(base_json)
    location_files = _location_files_or_exit(locations_dir)
    repo_root = base_json.resolve().parents[2] if len(base_json.resolve().parents) >= 3 else Path.cwd().resolve()

    # Wide, colorless console so year ranges like "2014-2016 (3)" stay intact and
    # tests/CI can match plain text. FORCE_COLOR overrides no_color alone — pin
    # color_system=None as well.
    output_console = Console(
        width=220, force_terminal=True, no_color=True, color_system=None
    )
    table = Table(show_header=True, header_style="bold")
    table.add_column("File", overflow="fold")
    table.add_column("Location", overflow="fold")
    table.add_column("Requested", overflow="fold", no_wrap=True)
    table.add_column("Available", overflow="fold", no_wrap=True)
    table.add_column("Area km2", overflow="fold")
    table.add_column("Px/m", overflow="fold")
    table.add_column("Downloaded", overflow="fold")
    table.add_column("Rendered", overflow="fold")

    for location_json in location_files:
        location_payload = _load_params_json_dict(location_json)
        merged: dict[str, object] = dict(base_payload)
        merged.update(location_payload)

        location_name = str(merged.get("location_name") or location_json.stem)
        requested_years = _years_label(merged)
        area = _format_compact_float(merged.get("area_km2") or merged.get("square_km"))

        artifacts_value = merged.get("artifacts_dir")
        if artifacts_value is None:
            try:
                slug = _slugify_location_name(location_name)
                artifacts = str(repo_root / f"artifacts_{slug}")
            except typer.BadParameter:
                artifacts = "-"
        else:
            artifacts = str(artifacts_value)

        available_years = _format_years_list(_load_available_years_from_artifacts(Path(artifacts)))
        px_per_meter = _format_compact_float(merged.get("px_per_meter"))
        downloaded = _manifest_checkpoint(Path(artifacts), "dataset_manifest_download.json")
        rendered = _manifest_checkpoint(Path(artifacts), "dataset_manifest_render.json")
        table.add_row(
            location_json.name,
            location_name,
            requested_years,
            available_years,
            area,
            px_per_meter,
            downloaded,
            rendered,
        )

    output_console.print(f"[cyan]Locations summary:[/cyan] {len(location_files)} files")
    output_console.print(table)
    raise typer.Exit(code=0)

@app.command("nls-index-json")
def nls_index_json(config_json: Path = typer.Argument(..., exists=True)) -> None:
    payload = _nls_force_provider(json.loads(config_json.read_text(encoding="utf-8")))
    try:
        cfg = IndexConfig(**payload)
    except ValidationError as error:
        _print_validation_error(error)
        raise typer.Exit(code=2) from error
    exit_code, artifact = get_provider("nls").index(cfg)
    _finish(exit_code, artifact)

@app.command("nls-download-json")
def nls_download_json(config_json: Path = typer.Argument(..., exists=True)) -> None:
    payload = _nls_force_provider(json.loads(config_json.read_text(encoding="utf-8")))
    try:
        # Parse as IndexConfig to learn where the index manifest lives
        # (the SIVL configs use a single output_json field for the index path).
        index_cfg = IndexConfig(
            **{k: v for k, v in payload.items() if k in IndexConfig.model_fields}
        )
        cfg = _nls_build_download_config(payload, index_cfg.output_json)
    except ValidationError as error:
        _print_validation_error(error)
        raise typer.Exit(code=2) from error
    exit_code, artifact = get_provider("nls").download(cfg)
    _finish(exit_code, artifact)

@app.command("nls-run-json")
def nls_run_json(config_json: Path = typer.Argument(..., exists=True)) -> None:
    """Single-shot NLS index + download from one JSON config."""
    payload = _nls_force_provider(json.loads(config_json.read_text(encoding="utf-8")))
    try:
        index_cfg = IndexConfig(**{k: v for k, v in payload.items() if k in IndexConfig.model_fields})
    except ValidationError as error:
        _print_validation_error(error)
        raise typer.Exit(code=2) from error
    provider = get_provider("nls")
    exit_code, index_artifact = provider.index(index_cfg)
    if exit_code != 0:
        _finish(exit_code, index_artifact)
    try:
        download_cfg = _nls_build_download_config(payload, index_artifact)
    except ValidationError as error:
        _print_validation_error(error)
        raise typer.Exit(code=2) from error
    exit_code, artifact = provider.download(download_cfg)
    _finish(exit_code, artifact)

@app.command("trajectory")
def trajectory_cmd(
    track: Path = typer.Option(..., "--track", help="Track file (.csv/.igc) or a directory with one .igc."),
    out: Path = typer.Option(..., "--out", help="Output directory for the manifest, preview, and downloads."),
    cell_km: float = typer.Option(1.0, "--cell-km", min=0.0001, help="Grid cell size in km."),
    year_start: int = typer.Option(2020, "--year-start"),
    year_end: int = typer.Option(2025, "--year-end"),
    download: bool = typer.Option(False, "--download/--no-download", help="Download source orthophoto for each window."),
    preview: bool = typer.Option(True, "--preview/--no-preview", help="Write a GeoJSON preview."),
) -> None:
    from satmap_dataset.config import TrajectoryConfig
    from satmap_dataset.pipeline import trajectory as trajectory_stage

    try:
        config = TrajectoryConfig(
            track_path=track,
            output_dir=out,
            cell_km=cell_km,
            year_start=year_start,
            year_end=year_end,
            download=download,
            preview=preview,
        )
        code, path = trajectory_stage.run(config)
    except (ValueError, RuntimeError, OSError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)
    _finish(code, path)

@app.command("trajectory-json")
def trajectory_json_cmd(
    config_json: Path = typer.Argument(..., help="JSON file mapped 1:1 onto TrajectoryConfig."),
) -> None:
    from satmap_dataset.config import TrajectoryConfig
    from satmap_dataset.pipeline import trajectory as trajectory_stage

    payload = _load_params_json_dict(config_json)
    try:
        config = TrajectoryConfig(**payload)
        code, path = trajectory_stage.run(config)
    except (ValueError, RuntimeError, OSError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)
    _finish(code, path)

@app.command("studio")
def studio_command(
    port: int = typer.Option(8501, help="Streamlit server port."),
    host: str = typer.Option("localhost", help="Streamlit server host."),
) -> None:
    """Launch the satmap-studio Streamlit UI."""
    import subprocess
    import sys

    app_path = Path(__file__).resolve().parents[1] / "studio" / "app.py"
    try:
        import streamlit  # noqa: F401
    except ImportError:
        console.print(
            "[red]studio extras not installed.[/red] Run: "
            "python -m pip install -e '.[studio]'"
        )
        raise typer.Exit(code=2)
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--server.address",
        host,
    ]
    raise typer.Exit(code=subprocess.call(cmd))



def main() -> None:
    configure_logging("INFO")
    app()



def summary_locations_main() -> None:
    configure_logging("INFO")
    app(args=["summary-locations", *sys.argv[1:]], prog_name="summary-locations")

