from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from satmap_dataset.config import DemAvailabilityConfig
from satmap_dataset.geo.bbox import parse as parse_project_bbox, wfs_query_bbox_str
from satmap_dataset.geoportal import dem_skorowidz_client
from satmap_dataset.geoportal.http import RetryPolicy
from satmap_dataset.models import DemAvailabilityEntry, DemAvailabilityReport
from satmap_dataset.progress_report import report_log, report_progress

logger = logging.getLogger("satmap_dataset.dem_availability")

_FULL_THRESHOLD = 99.9


def _parse_bbox(value: str) -> tuple[float, float, float, float]:
    return parse_project_bbox(value).as_tuple()


def _coverage_pct(
    aoi: tuple[float, float, float, float],
    tile_bboxes: list[tuple[float, float, float, float]],
    *,
    grid: int = 200,
) -> float:
    """Percent of the AOI rectangle covered by the union of tile rectangles.

    Both the AOI and the tile bboxes must be in the SAME coordinate convention
    (the report uses the swapped WFS query space for both, so the ratio is
    orientation-invariant). Computed by sampling a ``grid`` x ``grid`` lattice of
    cell centres over the AOI — no geometry dependency.
    """
    import numpy as np

    a0, b0, a1, b1 = aoi
    if a1 <= a0 or b1 <= b0:
        return 0.0
    if not tile_bboxes:
        return 0.0
    ax = a0 + (np.arange(grid) + 0.5) * (a1 - a0) / grid
    by = b0 + (np.arange(grid) + 0.5) * (b1 - b0) / grid
    gx, gy = np.meshgrid(ax, by)
    covered = np.zeros((grid, grid), dtype=bool)
    for t0, u0, t1, u1 in tile_bboxes:
        lo_a, hi_a = (t0, t1) if t0 <= t1 else (t1, t0)
        lo_b, hi_b = (u0, u1) if u0 <= u1 else (u1, u0)
        covered |= (gx >= lo_a) & (gx <= hi_a) & (gy >= lo_b) & (gy <= hi_b)
    return float(round(covered.mean() * 100.0, 1))


def _classify(pct: float) -> str:
    if pct >= _FULL_THRESHOLD:
        return "full"
    if pct > 0.0:
        return "partial"
    return "none"


def _formats_from_urls(urls: list[str]) -> list[str]:
    found: set[str] = set()
    for url in urls:
        name = Path(url).name.lower()
        if name.endswith(".xyz.zip"):
            found.add("xyz.zip")
        elif name.endswith(".zip"):
            found.add("zip")
        elif name.endswith(".xyz"):
            found.add("xyz")
        elif name.endswith(".asc"):
            found.add("asc")
        elif name.endswith((".tif", ".tiff")):
            found.add("tif")
    return sorted(found)



async def _run_async(config: DemAvailabilityConfig) -> tuple[int, Path]:
    retry_policy = RetryPolicy(max_attempts=config.retries, backoff_seconds=config.retry_delay)
    options = dict(config.provider_options)
    if options.get("wfs_swap_bbox_axes") is False:
        query_bbox = config.bbox
    else:
        query_bbox = wfs_query_bbox_str(config.bbox, config.srs)
    cov_aoi = _parse_bbox(query_bbox)  # coverage computed in the same (query) space
    year_filter = config.requested_years  # None = all advertised

    entries: list[DemAvailabilityEntry] = []
    errors: dict[str, str] = {}

    # Plan work units for progress reporting.
    plans: list[tuple[str, str, dict[int, str], list[int]]] = []
    for product in config.products:
        for datum in config.datums:
            combo = f"{product}|{datum}"
            try:
                year_to_typename = await dem_skorowidz_client.year_typenames(
                    product, datum, options, timeout=config.timeout, retry_policy=retry_policy
                )
            except Exception as exc:  # noqa: BLE001 - record and continue
                errors[combo] = str(exc)
                report_log(f"DEM catalog error {combo}: {exc}")
                continue
            years = sorted(year_to_typename)
            if year_filter is not None:
                years = [y for y in years if y in set(year_filter)]
            plans.append((product, datum, year_to_typename, years))

    total_steps = sum(len(years) for _, _, _, years in plans)
    total_steps = max(total_steps, 1)
    step = 0
    report_progress(0, total_steps, "Starting DEM skorowidz availability probe…")

    for product, datum, year_to_typename, years in plans:
        combo = f"{product}|{datum}"
        for year in years:
            label = f"DEM {product}/{datum} year {year}"
            report_progress(step, total_steps, label)
            try:
                _status, tiles, tile_bboxes, tile_acq = await dem_skorowidz_client.tiles_for_year(
                    product, datum, year, query_bbox, config.srs,
                    year_to_typename=year_to_typename, options=options,
                    timeout=config.timeout, retry_policy=retry_policy,
                )
            except Exception as exc:  # noqa: BLE001
                errors[f"{combo}|{year}"] = str(exc)
                report_log(f"{label}: error {exc}")
                step += 1
                continue
            pct = _coverage_pct(cov_aoi, [tuple(v) for v in tile_bboxes.values()])
            dates = sorted({
                str(meta.get("acquisition_date"))
                for meta in tile_acq.values()
                if meta.get("acquisition_date")
            })
            entries.append(
                DemAvailabilityEntry(
                    product=product, datum=datum, year=year,
                    godla=sorted(tiles.keys()), tile_count=len(tiles),
                    formats=_formats_from_urls(list(tiles.values())),
                    coverage=_classify(pct), coverage_pct=pct,
                    acquisition_dates=dates,
                )
            )
            report_log(f"{label}: {len(tiles)} tiles, coverage {_classify(pct)} ({pct:g}%)")
            step += 1

    report_progress(total_steps, total_steps, "Writing DEM availability report…")

    full_options = [
        {"product": e.product, "datum": e.datum, "year": e.year}
        for e in entries if e.coverage == "full"
    ]
    report = DemAvailabilityReport(
        provider="geoportal", aoi_bbox=config.bbox, srs=config.srs,
        entries=entries, errors=errors, full_coverage_options=full_options,
        run_parameters=config.model_dump(mode="json"),
    )
    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    config.output_json.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    logger.info(
        "DEM availability: entries=%s full=%s errors=%s",
        len(entries), len(full_options), len(errors),
    )
    return 0, config.output_json


def run(config: DemAvailabilityConfig) -> tuple[int, Path]:
    return asyncio.run(_run_async(config))
