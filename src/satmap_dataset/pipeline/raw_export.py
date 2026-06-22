"""Raw-tile export + ingest stage.

Lays native download tiles into <raw_root>/<provider>/<area>/<year>/*.tif, then
ingests co-located season-cell stacks (+ .tfw/.prj) and the per-area
manifest.yaml via the ported `raw_tiles` core. Writes raw_export_manifest.json.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from satmap_dataset.config import RawExportConfig
from satmap_dataset.models import RawExportManifest
from satmap_dataset.raw_tiles import core as rt
from satmap_dataset.raw_tiles.world_window import ingest_area_world_window


def _export_native_tiles(config: RawExportConfig) -> dict[int, int]:
    """Lay download_root/<year>/*.tif into <raw_root>/<provider>/<area>/<year>/.

    Returns a per-year exported-tile count. Symlinks by default; copies when
    link_mode == 'copy'.
    """
    src_root = Path(config.download_root)
    out_area = Path(config.raw_root) / config.provider / config.area
    counts: dict[int, int] = {}
    for year_dir in sorted(src_root.glob("*")):
        if not (year_dir.is_dir() and rt._YEAR_DIR_RE.match(year_dir.name)):
            continue
        year = int(year_dir.name)
        tiles = sorted(year_dir.glob("*.tif"))
        if not tiles:
            continue
        dest_dir = out_area / year_dir.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for tile in tiles:
            dest = dest_dir / tile.name
            if dest.is_symlink() or dest.exists():
                dest.unlink()
            if config.link_mode == "copy":
                shutil.copy2(tile, dest)
            else:
                dest.symlink_to(tile.resolve())
        counts[year] = len(tiles)
    return counts


def _can_reuse_raw_export(config: RawExportConfig, prior: RawExportManifest) -> bool:
    if not prior.passed or prior.stage != "raw_export":
        return False
    if (prior.provider, prior.area) != (config.provider, config.area):
        return False
    if str(prior.raw_root) != str(Path(config.raw_root)):
        return False
    if prior.link_mode != config.link_mode:
        return False
    if prior.cell_mode != config.cell_mode:
        return False
    if prior.run_parameters.get("equalize_gsd") != config.equalize_gsd:
        return False
    if prior.min_coverage != config.min_coverage:
        return False
    out_area = Path(config.raw_root) / config.provider / config.area
    if not (out_area / "manifest.yaml").exists():
        return False
    return all((out_area / cell).exists() for cell in prior.cell_dirs)


def run(config: RawExportConfig) -> tuple[int, Path]:
    output_json = Path(config.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    # Idempotent reuse: if a prior artifact still matches and all outputs exist, skip.
    if output_json.exists():
        try:
            prior = RawExportManifest.model_validate_json(output_json.read_text())
        except Exception:
            prior = None
        if prior is not None and _can_reuse_raw_export(config, prior):
            print(str(output_json))
            return 0, output_json

    warnings: list[str] = []
    errors: list[str] = []

    # 1. Export native tiles into the canonical layout.
    exported = _export_native_tiles(config)
    if not exported:
        errors.append(f"No <year>/*.tif tiles found under {config.download_root}")
        manifest = RawExportManifest(
            provider=config.provider, area=config.area, raw_root=str(Path(config.raw_root)),
            link_mode=config.link_mode, min_coverage=config.min_coverage,
            passed=False, errors=errors,
            run_parameters={"download_root": str(config.download_root)},
        )
        output_json.write_text(manifest.model_dump_json(indent=2))
        print(str(output_json))
        return 1, output_json

    # 2. Ingest cells. 'footprint' uses the verbatim ported core; 'world_window'
    #    co-registers mixed-GSD years to one equal-dimension stack.
    registry = rt.load_provider_registry()
    src_area = Path(config.raw_root) / config.provider / config.area
    if config.cell_mode == "world_window":
        area_manifest = ingest_area_world_window(
            src_area, Path(config.raw_root), registry,
            cell_size_m=config.cell_size_m, min_coverage=config.min_coverage,
            equalize_gsd=config.equalize_gsd,
        )
    else:
        area_manifest = rt.ingest_area(
            src_area, Path(config.raw_root), registry,
            cell_size_m=config.cell_size_m, min_coverage=config.min_coverage,
        )

    # EPSG cross-check (warning only).
    epsg = area_manifest.get("epsg")
    mismatch = False
    if epsg is not None:
        detected = rt.provider_for_epsg(epsg, registry)
        if detected != config.provider:
            mismatch = True
            warnings.append(
                f"EPSG:{epsg} maps to provider '{detected}' but configured provider "
                f"is '{config.provider}'"
            )

    # 3. Write the per-area manifest.yaml (handoff contract).
    out_provider_area = Path(config.raw_root) / area_manifest["provider"] / area_manifest["area"]
    out_provider_area.mkdir(parents=True, exist_ok=True)
    per_area_path = out_provider_area / "manifest.yaml"
    per_area_path.write_text(yaml.safe_dump(area_manifest, sort_keys=False))

    # 4. Tally + write the stage artifact.
    locations = area_manifest.get("locations", {})
    seasons_kept = sum(
        sum(1 for s in loc["seasons"] if not s.get("dropped")) for loc in locations.values()
    )
    seasons_dropped = sum(
        sum(1 for s in loc["seasons"] if s.get("dropped")) for loc in locations.values()
    )
    manifest = RawExportManifest(
        provider=config.provider,
        area=config.area,
        raw_root=str(Path(config.raw_root)),
        epsg=epsg,
        epsg_provider_mismatch=mismatch,
        link_mode=config.link_mode,
        cell_mode=config.cell_mode,
        min_coverage=config.min_coverage,
        cell_size_m=area_manifest.get("cell_size_m"),
        exported_tile_counts_by_year=exported,
        cells_produced=len(locations),
        seasons_kept=seasons_kept,
        seasons_dropped=seasons_dropped,
        per_area_manifest_path=str(per_area_path),
        cell_dirs=sorted(locations.keys()),
        source_download_manifest=str(config.download_manifest) if config.download_manifest else None,
        passed=True,
        warnings=warnings,
        errors=errors,
        run_parameters={"download_root": str(config.download_root),
                        "equalize_gsd": config.equalize_gsd},
    )
    output_json.write_text(manifest.model_dump_json(indent=2))
    print(str(output_json))
    return 0, output_json
