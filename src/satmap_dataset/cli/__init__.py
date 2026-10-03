"""CLI package — thin public surface for ``python -m satmap_dataset.cli``.

Command implementations live in sibling modules; repeated JSON / location-json /
all-location-json flavors are generated from :mod:`satmap_dataset.cli.registry`.
"""

from __future__ import annotations

from satmap_dataset.cli.app import app, register_all
from satmap_dataset.cli.common import (
    DEFAULT_CENTER_SQUARE_KM,
    _apply_location_paths_policy,
    _as_optional_float,
    _bbox_from_center_latlon,
    _bbox_from_center_rect,
    _center_mode_srs_supported,
    _finish,
    _format_compact_float,
    _format_years_list,
    _has_successful_validation_artifact,
    _load_available_years_from_artifacts,
    _load_params_json_dict,
    _location_files_or_exit,
    _lonlat_to_target_srs,
    _manifest_checkpoint,
    _print_availability_table,
    _print_validation_error,
    _requested_years_from_payload,
    _resolve_bbox_input,
    _resolve_json_center_bbox,
    _slugify_location_name,
    _years_label,
    console,
)
from satmap_dataset.cli.config_builders import (
    _build_dem_availability_config_from_base_and_location,
    _build_dem_config_from_base_and_location,
    _build_download_config_from_base_and_location,
    _build_index_config_from_base_and_location,
    _build_osm_config_from_base_and_location,
    _build_raw_export_config_from_base_and_location,
    _build_render_config_from_base_and_location,
    _build_run_config_from_base_and_location,
    _build_validate_config_from_base_and_location,
)
from satmap_dataset.pipeline import (
    dem,
    dem_availability,
    downloader,
    index_builder,
    location_run,
    raw_export,
    render,
    run_all,
    validator,
)
from satmap_dataset.pipeline import osm as osm_pipeline
from satmap_dataset.raw_tiles.split_manifest import build_test_manifest

register_all()

from satmap_dataset.cli.commands_misc import main, summary_locations_main

__all__ = [
    "DEFAULT_CENTER_SQUARE_KM",
    "app",
    "build_test_manifest",
    "console",
    "dem",
    "dem_availability",
    "downloader",
    "index_builder",
    "location_run",
    "main",
    "osm_pipeline",
    "raw_export",
    "render",
    "run_all",
    "summary_locations_main",
    "validator",
    "_apply_location_paths_policy",
    "_as_optional_float",
    "_bbox_from_center_latlon",
    "_bbox_from_center_rect",
    "_build_dem_availability_config_from_base_and_location",
    "_build_dem_config_from_base_and_location",
    "_build_download_config_from_base_and_location",
    "_build_index_config_from_base_and_location",
    "_build_osm_config_from_base_and_location",
    "_build_raw_export_config_from_base_and_location",
    "_build_render_config_from_base_and_location",
    "_build_run_config_from_base_and_location",
    "_build_validate_config_from_base_and_location",
    "_center_mode_srs_supported",
    "_finish",
    "_format_compact_float",
    "_format_years_list",
    "_has_successful_validation_artifact",
    "_load_available_years_from_artifacts",
    "_load_params_json_dict",
    "_location_files_or_exit",
    "_lonlat_to_target_srs",
    "_manifest_checkpoint",
    "_print_availability_table",
    "_print_validation_error",
    "_requested_years_from_payload",
    "_resolve_bbox_input",
    "_resolve_json_center_bbox",
    "_slugify_location_name",
    "_years_label",
]
