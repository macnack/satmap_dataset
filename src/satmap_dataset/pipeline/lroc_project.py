"""Optional LROC NAC projection stage (ISIS cam2map).

Reads a download LayerManifest, runs lronac2isis → spiceinit → cam2map per
frame when USGS ISIS is on PATH, and writes ``project_manifest.json``.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from satmap_dataset.config import LrocProjectConfig
from satmap_dataset.models import LayerManifest, LrocProjectFailure, LrocProjectManifest
from satmap_dataset.providers.lroc_nac import isis

logger = logging.getLogger("satmap_dataset.lroc_project")

RunCmd = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _default_run_cmd(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        check=False,
        capture_output=True,
        text=True,
    )


def _year_from_asset(path: Path) -> str:
    parent = path.parent.name
    if parent.isdigit() and len(parent) == 4:
        return parent
    return "unknown"


def _tools_as_str(tools: dict[str, Path | None]) -> dict[str, str | None]:
    return {name: (str(path) if path is not None else None) for name, path in tools.items()}


def _project_one(
    source: Path,
    *,
    project_root: Path,
    map_file: Path | None,
    overwrite: bool,
    keep_work_cubes: bool,
    tools: dict[str, Path],
    run_cmd: RunCmd,
) -> tuple[str | None, str | None, LrocProjectFailure | None]:
    """Return (projected_path | None, skipped_source | None, failure | None)."""
    year = _year_from_asset(source)
    out_dir = project_root / year
    work_cub = out_dir / f"{source.stem}.cub"
    map_cub = out_dir / f"{source.stem}_map.cub"

    if map_cub.exists() and map_cub.stat().st_size > 0 and not overwrite:
        # Reuse counts as projected; source listed under assets_skipped for provenance.
        return str(map_cub), str(source), None

    if not source.exists() or source.stat().st_size == 0:
        return None, None, LrocProjectFailure(
            source=str(source), step="source", message="missing or empty source asset"
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    steps: list[tuple[str, list[str], Path | None]] = [
        (
            "lronac2isis",
            isis.build_lronac2isis_cmd(source, work_cub, binary=tools["lronac2isis"]),
            work_cub,
        ),
        (
            "spiceinit",
            isis.build_spiceinit_cmd(work_cub, binary=tools["spiceinit"]),
            None,
        ),
        (
            "cam2map",
            isis.build_cam2map_cmd(
                work_cub, map_cub, map_file=map_file, binary=tools["cam2map"]
            ),
            map_cub,
        ),
    ]

    for step_name, argv, expected in steps:
        logger.info("LROC project %s: %s", step_name, " ".join(argv))
        result = run_cmd(argv)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            message = f"exit {result.returncode}"
            if detail:
                message = f"{message}: {detail[:500]}"
            return None, None, LrocProjectFailure(
                source=str(source), step=step_name, message=message
            )
        if expected is not None and (not expected.exists() or expected.stat().st_size == 0):
            return None, None, LrocProjectFailure(
                source=str(source),
                step=step_name,
                message=f"expected output missing or empty: {expected}",
            )

    if not keep_work_cubes and work_cub.exists() and work_cub != map_cub:
        try:
            work_cub.unlink()
        except OSError as exc:
            logger.warning("Could not remove work cube %s: %s", work_cub, exc)

    return str(map_cub), None, None


def run(
    config: LrocProjectConfig,
    *,
    which: Callable[[str], str | None] | None = None,
    run_cmd: RunCmd | None = None,
) -> tuple[int, Path]:
    output_json = Path(config.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    project_root = Path(config.project_root)
    project_root.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    warnings: list[str] = []
    assets_projected: list[str] = []
    assets_skipped: list[str] = []
    assets_failed: list[LrocProjectFailure] = []

    tools = isis.resolve_isis_tools(which=which)
    tools_str = _tools_as_str(tools)
    run_parameters: dict[str, Any] = config.model_dump(mode="json")

    def _write(passed: bool) -> tuple[int, Path]:
        manifest = LrocProjectManifest(
            srs=config.srs,
            project_root=str(project_root),
            source_download_manifest=str(config.download_manifest),
            assets_projected=sorted(assets_projected),
            assets_skipped=sorted(assets_skipped),
            assets_failed=assets_failed,
            isis_tools=tools_str,
            passed=passed,
            warnings=warnings,
            errors=errors,
            run_parameters=run_parameters,
        )
        output_json.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return (0 if passed else 1), output_json

    missing = isis.missing_isis_tools(tools)
    if missing:
        errors.append(
            "ISIS required but not on PATH (missing: "
            + ", ".join(missing)
            + "). Install USGS ISIS and ensure lronac2isis, spiceinit, and "
            "cam2map are available."
        )
        return _write(False)

    # All required tools present (narrow type for binary paths).
    tool_bins = {name: path for name, path in tools.items() if path is not None}

    manifest_path = Path(config.download_manifest)
    if not manifest_path.exists():
        errors.append(f"download_manifest not found: {manifest_path}")
        return _write(False)

    try:
        download = LayerManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        errors.append(f"invalid download_manifest: {exc}")
        return _write(False)

    if download.provider != "lroc_nac":
        errors.append(
            f"lroc-project requires download_manifest.provider='lroc_nac'; "
            f"got {download.provider!r}"
        )
        return _write(False)

    if not download.assets:
        errors.append("download_manifest.assets is empty")
        return _write(False)

    cmd_runner = run_cmd or _default_run_cmd
    for asset in download.assets:
        source = Path(asset)
        projected, skipped, failure = _project_one(
            source,
            project_root=project_root,
            map_file=Path(config.map_file) if config.map_file else None,
            overwrite=config.overwrite,
            keep_work_cubes=config.keep_work_cubes,
            tools=tool_bins,
            run_cmd=cmd_runner,
        )
        if failure is not None:
            assets_failed.append(failure)
            continue
        if projected is not None:
            assets_projected.append(projected)
        if skipped is not None:
            assets_skipped.append(skipped)

    passed = bool(assets_projected) and not assets_failed and not errors
    if not assets_projected and not errors:
        errors.append("no frames projected")
        passed = False
    return _write(passed)
