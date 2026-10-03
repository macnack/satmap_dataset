from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any

import typer
from pydantic import ValidationError
from rich.console import Console

console = Console(stderr=True)
DEFAULT_CENTER_SQUARE_KM = 4.0


def _patchable(name: str, fallback):
    """Resolve a helper via satmap_dataset.cli when loaded (test monkeypatch surface)."""
    import sys

    cli_mod = sys.modules.get("satmap_dataset.cli")
    if cli_mod is not None and hasattr(cli_mod, name):
        return getattr(cli_mod, name)
    return fallback

_CENTER_MODE_SUPPORTED_SRS = {"EPSG:2180", "EPSG:3006", "EPSG:3067"}


def _print_validation_error(error: ValidationError) -> None:
    console.print("[red]Invalid configuration:[/red]")
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"])
        console.print(f"- {location}: {item['msg']}")


def _finish(exit_code: int, artifact_path: Path) -> None:
    typer.echo(str(artifact_path))
    raise typer.Exit(code=exit_code)


def _print_availability_table(report) -> None:
    rows = sorted(
        [e for e in report.entries if e.tile_count > 0],
        key=lambda e: (e.product, e.datum, e.year),
    )
    console.print(f"[cyan]DEM availability[/cyan] AOI={report.aoi_bbox} ({report.srs})")
    console.print("  product datum     year  tiles  coverage      formats")
    for e in rows:
        cov = e.coverage if e.coverage != "partial" else f"partial({e.coverage_pct:g}%)"
        console.print(
            f"  {e.product:<7} {e.datum:<9} {e.year}   {e.tile_count:<5} {cov:<13} {','.join(e.formats)}"
        )
    combos = sorted({(e.product, e.datum) for e in report.entries})
    for product, datum in combos:
        missing = sorted(
            e.year for e in report.entries
            if e.product == product and e.datum == datum and e.tile_count == 0
        )
        if missing:
            console.print(f"  [yellow]no data:[/yellow] {product}/{datum} {missing}")
    for combo, msg in report.errors.items():
        console.print(f"  [red]error:[/red] {combo}: {msg}")


def _center_mode_srs_supported(srs: str) -> bool:
    normalized = srs.upper()
    if normalized in _CENTER_MODE_SUPPORTED_SRS:
        return True
    if normalized.startswith("EPSG:326") or normalized.startswith("EPSG:327"):
        return True
    return False


def _lonlat_to_target_srs(lon: float, lat: float, target_srs: str) -> tuple[float, float]:
    from satmap_dataset.providers.lantmateriet.crs import transform_point

    try:
        return transform_point("EPSG:4326", target_srs, lon, lat)
    except Exception as exc:
        raise RuntimeError(
            "Center-based bbox input requires pyproj or the PROJ 'proj' CLI in PATH."
        ) from exc


def _bbox_from_center_latlon(
    center_lat: float, center_lon: float, square_km: float, *, target_srs: str = "EPSG:2180"
) -> str:
    if square_km <= 0:
        raise ValueError("square_km must be > 0")
    side_m = math.sqrt(square_km) * 1000.0
    return _bbox_from_center_rect(
        center_lat,
        center_lon,
        width_meters=side_m,
        height_meters=side_m,
        target_srs=target_srs,
    )


def _bbox_from_center_rect(
    center_lat: float,
    center_lon: float,
    *,
    width_meters: float,
    height_meters: float,
    target_srs: str = "EPSG:2180",
) -> str:
    if width_meters <= 0 or height_meters <= 0:
        raise ValueError("width_meters and height_meters must be > 0")
    center_x, center_y = _patchable("_lonlat_to_target_srs", _lonlat_to_target_srs)(
        center_lon, center_lat, target_srs
    )
    half_w = width_meters / 2.0
    half_h = height_meters / 2.0
    return (
        f"{center_x - half_w:.3f},"
        f"{center_y - half_h:.3f},"
        f"{center_x + half_w:.3f},"
        f"{center_y + half_h:.3f}"
    )


def _resolve_bbox_input(
    *,
    bbox: str | None,
    center_lat: float | None,
    center_lon: float | None,
    square_km: float | None,
    srs: str,
    required: bool,
    width_meters: float | None = None,
    height_meters: float | None = None,
) -> str | None:
    rect_supplied = width_meters is not None or height_meters is not None
    center_mode_supplied = any(
        value is not None for value in (center_lat, center_lon, square_km)
    ) or rect_supplied
    if bbox is not None and center_mode_supplied:
        raise typer.BadParameter(
            "Provide either --bbox or center mode (--center-lat/--center-lon/--square-km/"
            "--width-meters+--height-meters), not both."
        )
    if rect_supplied and square_km is not None:
        raise typer.BadParameter(
            "Use either square_km/area_km2 or width_meters+height_meters, not both."
        )
    if rect_supplied and (width_meters is None or height_meters is None):
        raise typer.BadParameter(
            "Rectangular center mode requires both width_meters and height_meters."
        )

    if center_mode_supplied:
        if center_lat is None or center_lon is None:
            raise typer.BadParameter("Center mode requires both --center-lat and --center-lon.")
        normalized_srs = srs.upper()
        if not _center_mode_srs_supported(normalized_srs):
            raise typer.BadParameter(
                "Center mode currently supports EPSG:2180, EPSG:3006, EPSG:3067, "
                f"and WGS84 UTM zones (EPSG:326NN/327NN), got --srs {srs}."
            )
        try:
            if rect_supplied:
                return _patchable("_bbox_from_center_rect", _bbox_from_center_rect)(
                    center_lat,
                    center_lon,
                    width_meters=float(width_meters),
                    height_meters=float(height_meters),
                    target_srs=normalized_srs,
                )
            effective_square_km = square_km if square_km is not None else DEFAULT_CENTER_SQUARE_KM
            return _patchable("_bbox_from_center_latlon", _bbox_from_center_latlon)(
                center_lat, center_lon, effective_square_km, target_srs=normalized_srs
            )
        except RuntimeError as error:
            raise typer.BadParameter(str(error)) from error
        except ValueError as error:
            raise typer.BadParameter(str(error)) from error

    if required and bbox is None:
        raise typer.BadParameter(
            "bbox is required. Use --bbox xmin,ymin,xmax,ymax or center mode options."
        )
    return bbox


def _load_params_json_dict(params_json: Path) -> dict[str, object]:
    try:
        payload = json.loads(params_json.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        console.print(f"[red]Missing params JSON:[/red] {params_json}")
        raise typer.Exit(code=2) from error
    except json.JSONDecodeError as error:
        console.print(f"[red]Invalid JSON:[/red] {params_json} ({error})")
        raise typer.Exit(code=2) from error
    if not isinstance(payload, dict):
        console.print("[red]Invalid params JSON:[/red] top-level object must be a JSON object.")
        raise typer.Exit(code=2)
    return payload


def _as_optional_float(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as error:
            raise typer.BadParameter(f"{field_name} must be numeric") from error
    raise typer.BadParameter(f"{field_name} must be numeric")


def _resolve_json_center_bbox(payload: dict[str, object], *, required: bool) -> dict[str, object]:
    normalized = dict(payload)
    center_lat = _as_optional_float(normalized.get("center_lat"), "center_lat")
    center_lon = _as_optional_float(normalized.get("center_lon"), "center_lon")
    square_km = _as_optional_float(normalized.get("square_km"), "square_km")
    area_km2 = _as_optional_float(normalized.get("area_km2"), "area_km2")
    width_meters = _as_optional_float(normalized.get("width_meters"), "width_meters")
    height_meters = _as_optional_float(normalized.get("height_meters"), "height_meters")
    if square_km is not None and area_km2 is not None:
        raise typer.BadParameter("Use only one of square_km or area_km2 in JSON params.")
    effective_square_km = square_km if square_km is not None else area_km2
    # Rectangular spec (width+height) overrides any square spec inherited from
    # a base config — caller intent is clearly to use the rectangle.
    if width_meters is not None and height_meters is not None:
        effective_square_km = None
    bbox_value = normalized.get("bbox")
    bbox = str(bbox_value) if bbox_value is not None else None
    srs = str(normalized.get("srs", "EPSG:2180"))
    resolved_bbox = _resolve_bbox_input(
        bbox=bbox,
        center_lat=center_lat,
        center_lon=center_lon,
        square_km=effective_square_km,
        srs=srs,
        required=required,
        width_meters=width_meters,
        height_meters=height_meters,
    )
    normalized["bbox"] = resolved_bbox
    normalized["srs"] = srs
    normalized.pop("center_lat", None)
    normalized.pop("center_lon", None)
    normalized.pop("square_km", None)
    normalized.pop("area_km2", None)
    normalized.pop("width_meters", None)
    normalized.pop("height_meters", None)
    return normalized


def _slugify_location_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    lowered = ascii_only.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    slug = re.sub(r"_+", "_", slug)
    if not slug:
        raise typer.BadParameter(f"Cannot build slug from location_name={value!r}")
    return slug


def _apply_location_paths_policy(payload: dict[str, object], repo_root: Path) -> dict[str, object]:
    normalized = dict(payload)
    location_name = normalized.get("location_name")
    if location_name is None:
        return normalized
    slug = _slugify_location_name(str(location_name))
    normalized.setdefault("download_root", str(repo_root / f"downloads_{slug}"))
    normalized.setdefault("render_root", str(repo_root / f"rendered_{slug}"))
    normalized.setdefault("artifacts_dir", str(repo_root / f"artifacts_{slug}"))
    normalized.setdefault("dem_root", str(repo_root / f"dem_{slug}"))
    normalized.setdefault("osm_root", str(repo_root / f"osm_{slug}"))
    return normalized


def _location_files_or_exit(locations_dir: Path) -> list[Path]:
    files = sorted(locations_dir.glob("*.json"))
    if not files:
        console.print(f"[red]No location JSON files found in:[/red] {locations_dir}")
        raise typer.Exit(code=2)
    return files


def _format_compact_float(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{numeric:.6f}".rstrip("0").rstrip(".")


def _years_label(payload: dict[str, object]) -> str:
    years = _requested_years_from_payload(payload)
    if years:
        return _format_years_list(years)
    return "-"


def _requested_years_from_payload(payload: dict[str, object]) -> list[int]:
    if isinstance(payload.get("requested_years"), list):
        return sorted({int(year) for year in payload["requested_years"]})
    year_start = payload.get("year_start")
    year_end = payload.get("year_end")
    if year_start is not None and year_end is not None:
        try:
            start = int(year_start)
            end = int(year_end)
        except (TypeError, ValueError):
            return []
        if end >= start:
            return list(range(start, end + 1))
    return []


def _format_years_list(years: list[int]) -> str:
    if not years:
        return "-"
    ordered = sorted({int(year) for year in years})
    count = len(ordered)
    if count == 1:
        return f"{ordered[0]} (1)"
    contiguous = all((ordered[idx + 1] - ordered[idx]) == 1 for idx in range(count - 1))
    if contiguous:
        return f"{ordered[0]}-{ordered[-1]} ({count})"
    if count <= 6:
        return f"{','.join(str(year) for year in ordered)} ({count})"
    return f"{ordered[0]}..{ordered[-1]} ({count})"


def _load_available_years_from_artifacts(artifacts_dir: Path) -> list[int]:
    candidates = [
        artifacts_dir / "year_availability_report.json",
        artifacts_dir / "index_manifest.json",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        values = payload.get("years_available_wfs")
        if isinstance(values, list):
            try:
                return sorted({int(year) for year in values})
            except (TypeError, ValueError):
                continue
    return []


def _manifest_checkpoint(artifacts_dir: Path, filename: str) -> str:
    manifest_path = artifacts_dir / filename
    if not manifest_path.exists():
        return "-"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "file"
    passed = payload.get("passed")
    if passed is True:
        return "yes"
    if passed is False:
        return "fail"
    return "file"


def _has_successful_validation_artifact(artifacts_dir: Path) -> bool:
    return _manifest_checkpoint(artifacts_dir, "validation_report.json") == "yes"

