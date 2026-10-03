from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import typer
from pydantic import ValidationError

from satmap_dataset.cli.common import (
    _finish,
    _load_params_json_dict,
    _location_files_or_exit,
    _print_validation_error,
    _resolve_json_center_bbox,
    console,
)

ResolveBbox = Literal["required", "optional", "none"]
ErrorStyle = Literal["separate", "combined", "index", "invalid"]

Runner = Callable[[Any], tuple[int, Path]]
Builder = Callable[..., Any]
AfterRun = Callable[[int, Path], None]


@dataclass(frozen=True)
class StageCommandSpec:
    """Declarative registration for repeated JSON / location-json / all-location-json flavors."""

    name: str
    config_cls: type
    runner: Runner
    json_help: str
    resolve_bbox: ResolveBbox = "required"
    build_location: Builder | None = None
    location_help: str = "Path to location JSON (location_name, center_lat, center_lon)."
    flavors: frozenset[str] = frozenset({"json", "location-json", "all-location-json"})
    after_run: AfterRun | None = None
    all_label: str | None = None
    all_failure_header: str | None = None
    error_style: ErrorStyle = "index"
    # When True, *-json uses model_validate_json(text) instead of load+bbox+validate.
    raw_json_validate: bool = False


def _handle_config_error(error: Exception, *, style: ErrorStyle) -> str:
    if isinstance(error, ValidationError):
        if style == "combined":
            console.print(f"[red]{error}[/red]")
            return str(error)
        _print_validation_error(error)
        if style == "invalid":
            return "invalid"
        return "validation_error"
    console.print(f"[red]{error}[/red]")
    if style == "invalid":
        return "invalid"
    return str(error)


def register_stage_commands(app: typer.Typer, specs: list[StageCommandSpec]) -> None:
    for spec in specs:
        _register_one(app, spec)


def _register_one(app: typer.Typer, spec: StageCommandSpec) -> None:
    if "json" in spec.flavors:
        _register_json(app, spec)
    if "location-json" in spec.flavors and spec.build_location is not None:
        _register_location(app, spec)
    if "all-location-json" in spec.flavors and spec.build_location is not None:
        _register_all(app, spec)


def _run_and_finish(spec: StageCommandSpec, config: Any) -> None:
    exit_code, artifact_path = spec.runner(config)
    if spec.after_run is not None:
        spec.after_run(exit_code, artifact_path)
    _finish(exit_code, artifact_path)


def _register_json(app: typer.Typer, spec: StageCommandSpec) -> None:
    command_name = f"{spec.name}-json"

    @app.command(command_name)
    def _json_command(
        params_json: Path = typer.Argument(..., help=spec.json_help),
    ) -> None:
        try:
            if spec.raw_json_validate:
                config = spec.config_cls.model_validate_json(
                    params_json.read_text(encoding="utf-8")
                )
            else:
                payload = _load_params_json_dict(params_json)
                if spec.resolve_bbox == "required":
                    payload = _resolve_json_center_bbox(payload, required=True)
                elif spec.resolve_bbox == "optional":
                    payload = _resolve_json_center_bbox(payload, required=False)
                config = spec.config_cls.model_validate(payload)
        except FileNotFoundError as error:
            console.print(f"[red]Missing params JSON:[/red] {params_json}")
            raise typer.Exit(code=2) from error
        except typer.BadParameter as error:
            console.print(f"[red]{error}[/red]")
            raise typer.Exit(code=2) from error
        except ValidationError as error:
            _print_validation_error(error)
            raise typer.Exit(code=2) from error
        _run_and_finish(spec, config)

    _json_command.__name__ = f"{spec.name.replace('-', '_')}_json_command"
    _json_command.__qualname__ = _json_command.__name__


def _register_location(app: typer.Typer, spec: StageCommandSpec) -> None:
    command_name = f"{spec.name}-location-json"
    build = spec.build_location
    assert build is not None

    @app.command(command_name)
    def _location_command(
        location_json: Path = typer.Argument(..., help=spec.location_help),
        base_json: Path = typer.Option(
            Path("configs/run/base.json"),
            "--base-json",
            help="Path to base JSON with shared parameters.",
        ),
    ) -> None:
        try:
            config = build(base_json=base_json, location_json=location_json)
        except typer.BadParameter as error:
            console.print(f"[red]{error}[/red]")
            raise typer.Exit(code=2) from error
        except ValidationError as error:
            _print_validation_error(error)
            raise typer.Exit(code=2) from error
        _run_and_finish(spec, config)

    _location_command.__name__ = f"{spec.name.replace('-', '_')}_location_json_command"
    _location_command.__qualname__ = _location_command.__name__


def _register_all(app: typer.Typer, spec: StageCommandSpec) -> None:
    command_name = f"{spec.name}-all-location-json"
    build = spec.build_location
    assert build is not None
    label = spec.all_label or command_name
    failure_header = spec.all_failure_header or f"{command_name} finished with failures:"

    @app.command(command_name)
    def _all_command(
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
            help="Continue with remaining locations when one fails.",
        ),
    ) -> None:
        location_files = _location_files_or_exit(locations_dir)
        failures: list[str] = []
        for location_json in location_files:
            console.print(f"[cyan]{label}:[/cyan] {location_json}")
            try:
                config = build(base_json=base_json, location_json=location_json)
            except (typer.BadParameter, ValidationError) as error:
                message = _handle_config_error(error, style=spec.error_style)
                failures.append(f"{location_json}: {message}")
                if not continue_on_error:
                    raise typer.Exit(code=2) from error
                continue

            exit_code, artifact_path = spec.runner(config)
            console.print(str(artifact_path))
            if exit_code != 0:
                failures.append(f"{location_json}: exit={exit_code}")
                if not continue_on_error:
                    raise typer.Exit(code=exit_code)

        if failures:
            console.print(f"[yellow]{failure_header}[/yellow]")
            for entry in failures:
                console.print(f"- {entry}")
            raise typer.Exit(code=1)
        raise typer.Exit(code=0)

    _all_command.__name__ = f"{spec.name.replace('-', '_')}_all_location_json_command"
    _all_command.__qualname__ = _all_command.__name__
