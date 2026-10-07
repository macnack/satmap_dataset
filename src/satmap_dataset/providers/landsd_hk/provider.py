"""Hong Kong Lands Department Imagery XYZ provider (experimental).

Stitches anonymous LandsD Imagery Map API PNG tiles into an EPSG:3857 GeoTIFF.
The public XYZ service is a **current mosaic** (not year-aware). Index reports a
single synthetic capture year (``imagery_year``, default = ``year_end``) when it
falls inside the requested range — enough for trajectory eval at ~0.28 m (z19).

For true multi-year ~30 cm stacks over HK, use Esri Wayback via ArcGIS export or
an external tool such as GEHistoricalImagery (licence-restricted); see
``docs/DATA_LICENSING.md`` and ``docs/providers/landsd_hk.md``.

``provider_options``:

- ``zoom`` explicit zoom (else from ``gsd_m`` / ``px_per_meter``)
- ``gsd_m`` target ground GSD in metres (default 0.3 → typically z19)
- ``min_zoom`` / ``max_zoom`` (default 15 / 19)
- ``max_tiles`` hard cap (default 1024)
- ``imagery_year`` synthetic year label for the current mosaic (default year_end)
- ``xyz_base`` / ``sr`` endpoint overrides (default WGS84 Imagery API)
- ``user_agent`` override
"""

from __future__ import annotations

import asyncio
import logging
import random
from pathlib import Path
from typing import Any

import httpx

from satmap_dataset.config import DownloadConfig, IndexConfig
from satmap_dataset.models import (
    IndexManifest,
    LayerManifest,
    TileAcquisitionMetadata,
    YearAvailabilityReport,
    YearStatus,
)
from satmap_dataset.pipeline.validator import evaluate_year_policy
from satmap_dataset.providers.base import Provider
from satmap_dataset.providers.landsd_hk import tiles, xyz

PROVIDER_NAME = "landsd_hk"
DEFAULT_TARGET_GSD_M = 0.30
DEFAULT_MAX_TILES = 1024

logger = logging.getLogger("satmap_dataset.landsd_hk")


def _parse_bbox(value: str) -> tuple[float, float, float, float]:
    parts = [float(p.strip()) for p in value.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must have format xmin,ymin,xmax,ymax")
    xmin, ymin, xmax, ymax = parts
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("bbox must satisfy xmin<xmax and ymin<ymax")
    return xmin, ymin, xmax, ymax


def _opt(options: dict[str, Any], key: str, default: Any) -> Any:
    value = options.get(key)
    return default if value in (None, "") else value


def _resolve_zoom(
    options: dict[str, Any],
    *,
    px_per_meter: float | None,
    aoi: tiles.MercatorBBox,
) -> tuple[int, list[str]]:
    notes: list[str] = []
    min_z = int(_opt(options, "min_zoom", xyz.DEFAULT_MIN_ZOOM))
    max_z = int(_opt(options, "max_zoom", xyz.DEFAULT_MAX_ZOOM))
    if min_z > max_z:
        raise ValueError(f"min_zoom ({min_z}) > max_zoom ({max_z})")
    lat = tiles.mercator_to_lonlat(*aoi.center)[1]
    if "zoom" in options and options["zoom"] not in (None, ""):
        zoom = int(options["zoom"])
        notes.append(f"zoom={zoom} from provider_options")
    else:
        target = float(_opt(options, "gsd_m", DEFAULT_TARGET_GSD_M))
        if px_per_meter and px_per_meter > 0 and "gsd_m" not in options:
            target = 1.0 / float(px_per_meter)
            notes.append(f"gsd_m={target:.4f} from px_per_meter")
        else:
            notes.append(f"gsd_m={target:.4f}")
        zoom = tiles.zoom_for_gsd(target, lat, min_zoom=min_z, max_zoom=max_z)
        notes.append(f"selected zoom={zoom} (ground_gsd≈{tiles.ground_gsd(zoom, lat):.3f} m)")
    zoom = max(min_z, min(max_z, zoom))
    return zoom, notes


class LandsdHkProvider(Provider):
    name = PROVIDER_NAME
    default_target_srs = "EPSG:3857"

    def index(self, config: IndexConfig) -> tuple[int, Path]:
        options = dict(config.provider_options)
        imagery_year = int(_opt(options, "imagery_year", config.year_end))
        errors: list[str] = []
        warnings: list[str] = [
            "LandsD Imagery XYZ is a current mosaic (not multi-year). "
            f"Reporting synthetic imagery_year={imagery_year}."
        ]
        year_statuses: list[YearStatus] = []
        years_included: list[int] = []
        years_excluded: dict[int, str] = {}
        tile_sources_by_year: dict[int, dict[str, str]] = {}
        tile_bboxes_by_year: dict[int, dict[str, list[float]]] = {}
        tile_acquisition_by_year: dict[int, dict[str, TileAcquisitionMetadata]] = {}

        for year in config.requested_years:
            if year == imagery_year:
                years_included.append(year)
                year_statuses.append(
                    YearStatus(
                        year=year,
                        typename_exists=True,
                        feature_count=1,
                        status="has_features",
                        reason="current LandsD Imagery XYZ mosaic",
                    )
                )
                tile_id = f"landsd_xyz_{year}"
                base = str(_opt(options, "xyz_base", xyz.DEFAULT_XYZ_BASE))
                sr = str(_opt(options, "sr", xyz.DEFAULT_SR))
                tile_sources_by_year[year] = {
                    tile_id: xyz.tile_url(base, sr=sr, z=0, x=0, y=0).rsplit("/", 3)[0]
                    + "/{z}/{x}/{y}.png"
                }
                # Placeholder bbox filled at download from AOI; keep empty ok for index.
                tile_bboxes_by_year[year] = {}
                tile_acquisition_by_year[year] = {
                    tile_id: TileAcquisitionMetadata(
                        acquisition_year=year,
                        gsd=float(_opt(options, "gsd_m", DEFAULT_TARGET_GSD_M)),
                    )
                }
            else:
                reason = (
                    f"landsd_hk XYZ has no per-year archive; only imagery_year={imagery_year}"
                )
                years_excluded[year] = reason
                year_statuses.append(
                    YearStatus(
                        year=year,
                        typename_exists=False,
                        feature_count=0,
                        status="no_typename",
                        reason=reason,
                    )
                )

        if not years_included:
            errors.append(
                f"imagery_year={imagery_year} not in requested "
                f"[{config.year_start}, {config.year_end}]; set provider_options.imagery_year "
                "or widen the year range."
            )

        policy = evaluate_year_policy(
            requested_years=config.requested_years,
            available_years=years_included,
            strict_years=config.strict_years,
            min_years=config.min_years,
        )
        combined_errors = errors + list(policy.errors)
        manifest = IndexManifest(
            provider=PROVIDER_NAME,
            year_start=config.year_start,
            year_end=config.year_end,
            bbox=config.bbox,
            srs=config.srs,
            strict_years=config.strict_years,
            min_years=config.min_years,
            wfs_bbox_axes_swapped=False,
            years_requested=config.requested_years,
            year_statuses=year_statuses,
            years_available_wfs=[imagery_year] if years_included else [],
            years_included=years_included,
            years_excluded_with_reason=years_excluded,
            common_tile_ids=[],
            tile_sources_by_year=tile_sources_by_year,
            tile_bboxes_by_year=tile_bboxes_by_year,
            tile_acquisition_by_year=tile_acquisition_by_year,
            passed=policy.passed and bool(years_included) and not errors,
            errors=combined_errors,
            warnings=warnings + list(policy.warnings),
            run_parameters=config.model_dump(mode="json"),
            provider_metadata={
                "experimental": True,
                "attribution": xyz.ATTRIBUTION,
                "license_notice": xyz.LICENSE_NOTICE,
                "xyz_base": str(_opt(options, "xyz_base", xyz.DEFAULT_XYZ_BASE)),
                "sr": str(_opt(options, "sr", xyz.DEFAULT_SR)),
                "imagery_year": imagery_year,
                "product": "LandsD Imagery Map API (current mosaic)",
            },
        )
        availability = YearAvailabilityReport(
            year_start=config.year_start,
            year_end=config.year_end,
            bbox=config.bbox,
            srs=config.srs,
            wfs_bbox_axes_swapped=False,
            years_requested=manifest.years_requested,
            year_statuses=manifest.year_statuses,
            years_available_wfs=manifest.years_available_wfs,
            years_included=manifest.years_included,
            years_excluded_with_reason=manifest.years_excluded_with_reason,
            strict_years=manifest.strict_years,
            min_years=manifest.min_years,
            passed=manifest.passed,
            errors=manifest.errors,
            warnings=manifest.warnings,
            run_parameters=manifest.run_parameters,
        )
        config.year_availability_output_json.parent.mkdir(parents=True, exist_ok=True)
        config.output_json.parent.mkdir(parents=True, exist_ok=True)
        config.year_availability_output_json.write_text(
            availability.model_dump_json(indent=2), encoding="utf-8"
        )
        config.output_json.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        logger.info(
            "landsd_hk index: imagery_year=%s included=%s passed=%s",
            imagery_year,
            years_included,
            manifest.passed,
        )
        return (0 if manifest.passed else 1), config.output_json

    def download(self, config: DownloadConfig) -> tuple[int, Path]:
        return asyncio.run(self._download_async(config))

    async def _download_async(self, config: DownloadConfig) -> tuple[int, Path]:
        index_manifest = IndexManifest.model_validate_json(
            config.index_manifest.read_text(encoding="utf-8")
        )
        options = dict(config.provider_options)
        if config.bbox is None:
            raise ValueError("bbox is required for landsd_hk download")
        aoi = tiles.aoi_to_mercator_bbox(_parse_bbox(config.bbox), config.srs)
        errors: list[str] = []
        notes: list[str] = []
        try:
            zoom, zoom_notes = _resolve_zoom(
                options, px_per_meter=config.px_per_meter, aoi=aoi
            )
            notes.extend(zoom_notes)
        except ValueError as exc:
            return self._write_download_manifest(
                config, index_manifest, assets=[], years_source_map={}, errors=[str(exc)], details={}, zoom=None
            )

        pad = 2 * tiles.resolution(zoom)
        crop = tiles.pad_bbox(aoi, pad)
        tile_range = tiles.tile_range_for_bbox(crop, zoom)
        if tile_range.count > int(_opt(options, "max_tiles", DEFAULT_MAX_TILES)):
            return self._write_download_manifest(
                config,
                index_manifest,
                assets=[],
                years_source_map={},
                errors=[
                    f"{tile_range.count} tiles at z{zoom} exceed max_tiles="
                    f"{int(_opt(options, 'max_tiles', DEFAULT_MAX_TILES))}"
                ],
                details={},
                zoom=zoom,
            )

        base = str(_opt(options, "xyz_base", xyz.DEFAULT_XYZ_BASE))
        sr = str(_opt(options, "sr", xyz.DEFAULT_SR))
        user_agent = str(_opt(options, "user_agent", xyz.DEFAULT_USER_AGENT))
        assets: list[str] = []
        years_source_map: dict[int, str] = {}
        details: dict[str, Any] = {}

        timeout = httpx.Timeout(timeout=config.timeout, connect=min(config.timeout, 20.0))
        limits = httpx.Limits(
            max_connections=config.concurrency, max_keepalive_connections=config.concurrency
        )
        headers = {"User-Agent": user_agent}
        semaphore = asyncio.Semaphore(max(1, config.concurrency))

        async with httpx.AsyncClient(
            follow_redirects=True, timeout=timeout, limits=limits, headers=headers
        ) as client:
            for year in index_manifest.years_included:
                output_path = config.download_root / str(year) / f"landsd_xyz_z{zoom}.tif"
                if output_path.exists() and output_path.stat().st_size > 0 and not config.overwrite:
                    assets.append(str(output_path))
                    years_source_map[year] = "xyz"
                    details[str(year)] = {"path": str(output_path), "reused": True, "zoom": zoom}
                    continue
                try:
                    detail = await self._fetch_and_stitch(
                        client,
                        semaphore,
                        base=base,
                        sr=sr,
                        tile_range=tile_range,
                        crop=crop,
                        output_path=output_path,
                        sleep_min=config.sleep_min,
                        sleep_max=config.sleep_max,
                        retries=config.retries,
                        retry_delay=config.retry_delay,
                        max_missing_frac=float(_opt(options, "max_missing_tile_fraction", 0.05)),
                    )
                except (httpx.HTTPError, ValueError, OSError) as exc:
                    errors.append(f"year {year}: {exc}")
                    continue
                details[str(year)] = detail
                assets.append(str(output_path))
                years_source_map[year] = "xyz"

        lat = tiles.mercator_to_lonlat(*aoi.center)[1]
        details["_summary"] = {
            "zoom": zoom,
            "tiles": tile_range.count,
            "ground_gsd_m": round(tiles.ground_gsd(zoom, lat), 4),
            "mercator_pixel_size_m": tiles.resolution(zoom),
            "notes": notes,
            "attribution": xyz.ATTRIBUTION,
        }
        return self._write_download_manifest(
            config,
            index_manifest,
            assets=assets,
            years_source_map=years_source_map,
            errors=errors,
            details=details,
            zoom=zoom,
        )

    async def _fetch_and_stitch(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        *,
        base: str,
        sr: str,
        tile_range: tiles.TileRange,
        crop: tiles.MercatorBBox,
        output_path: Path,
        sleep_min: float,
        sleep_max: float,
        retries: int,
        retry_delay: float,
        max_missing_frac: float,
    ) -> dict[str, Any]:
        z = tile_range.z

        async def fetch(xy: tuple[int, int]):
            x, y = xy
            url = xyz.tile_url(base, sr=sr, z=z, x=x, y=y)
            async with semaphore:
                if sleep_max > 0:
                    await asyncio.sleep(random.uniform(sleep_min, sleep_max))
                last_exc: Exception | None = None
                for attempt in range(retries + 1):
                    try:
                        response = await client.get(url)
                        if response.status_code in {408, 429} or response.status_code >= 500:
                            raise httpx.HTTPStatusError(
                                f"retryable {response.status_code}",
                                request=response.request,
                                response=response,
                            )
                        response.raise_for_status()
                        return xy, tiles.decode_tile(response.content)
                    except (httpx.HTTPError, ValueError) as exc:
                        last_exc = exc
                        if attempt < retries:
                            await asyncio.sleep(retry_delay * (2**attempt))
                logger.warning("landsd_hk tile failed z=%s x=%s y=%s: %s", z, x, y, last_exc)
                return xy, None

        results = await asyncio.gather(*(fetch(xy) for xy in tile_range.tiles()))
        tile_arrays = dict(results)
        missing = sum(1 for arr in tile_arrays.values() if arr is None)
        if missing / max(1, tile_range.count) > max_missing_frac:
            raise ValueError(
                f"{missing}/{tile_range.count} tiles missing at z{z} "
                f"(max_missing_tile_fraction={max_missing_frac})"
            )
        result = tiles.stitch(tile_arrays, tile_range, crop)
        tiles.write_geotiff_3857(output_path, result)
        return {
            "path": str(output_path),
            "zoom": z,
            "tiles": tile_range.count,
            "missing_tiles": missing,
            "width": int(result.array.shape[1]),
            "height": int(result.array.shape[0]),
            "bbox_3857": result.bbox.as_list(),
            "srs": "EPSG:3857",
        }

    def _write_download_manifest(
        self,
        config: DownloadConfig,
        index_manifest: IndexManifest,
        *,
        assets: list[str],
        years_source_map: dict[int, str],
        errors: list[str],
        details: dict[str, Any],
        zoom: int | None,
    ) -> tuple[int, Path]:
        years_included = sorted(years_source_map)
        passed = (
            bool(assets)
            and not errors
            and len(years_included) == len(index_manifest.years_included)
        )
        manifest = LayerManifest(
            layer=f"{PROVIDER_NAME}_rgb",
            role="rgb",
            stage="download",
            provider=PROVIDER_NAME,
            years_requested=index_manifest.years_requested,
            years_available_wfs=index_manifest.years_available_wfs,
            years_included=years_included,
            years_excluded_with_reason=index_manifest.years_excluded_with_reason,
            common_tile_ids=[],
            tile_sources_by_year=index_manifest.tile_sources_by_year,
            tile_bboxes_by_year=index_manifest.tile_bboxes_by_year,
            tile_acquisition_by_year=index_manifest.tile_acquisition_by_year,
            assets=sorted(set(assets)),
            source_manifest=str(config.index_manifest),
            mode="wms_tiled",
            target_bbox=config.bbox,
            target_srs=config.srs,
            profile=config.profile,
            px_per_meter=config.px_per_meter,
            years_source_map=years_source_map,
            forced_wms_years=[],
            passed=passed,
            notes=(
                f"provider={PROVIDER_NAME} zoom={zoom} downloaded={len(assets)} errors={len(errors)}"
                + (" | " + " ; ".join(errors[:3]) if errors else "")
            ),
            run_parameters=config.model_dump(mode="json"),
            provider_metadata={
                **(index_manifest.provider_metadata or {}),
                "download_zoom": zoom,
                "downloads_by_year": details,
                "download_errors": errors,
            },
        )
        config.output_json.parent.mkdir(parents=True, exist_ok=True)
        config.output_json.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        logger.info(
            "landsd_hk download: assets=%s errors=%s passed=%s",
            len(assets),
            len(errors),
            passed,
        )
        return (0 if passed else 1), config.output_json
