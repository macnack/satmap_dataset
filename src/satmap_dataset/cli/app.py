from __future__ import annotations

import typer

app = typer.Typer(help="satmap_dataset CLI (WFS-first pipeline)", no_args_is_help=True)


def register_all() -> None:
    """Import command modules (side-effect @app.command) and register registry flavors."""
    from . import commands_misc, commands_raw, layers_dem, layers_osm, layers_rgb

    # Flag / special commands register via @app.command on import.
    _ = commands_misc
    layers_rgb.register(app)
    layers_dem.register(app)
    layers_osm.register(app)
    commands_raw.register(app)
