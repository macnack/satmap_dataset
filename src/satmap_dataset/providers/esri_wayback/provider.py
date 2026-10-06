"""Esri World Imagery Wayback provider (experimental).

**Licensing:** Esri World Imagery / Wayback is governed by the Esri Master
Agreement. Its terms prohibit scraping/downloading/storing basemap data outside
Esri Content Packages and using it to train AI/ML systems outside Esri
software. See ``docs/DATA_LICENSING.md`` before running this provider.

Index: parse every release from WMTS GetCapabilities, collapse releases into
distinct imagery versions for the AOI (``tilemap`` local changes, or tile
content hash), read the capture date/source/resolution from each version's
metadata layer, bucket by **capture year** and pick one version per year.

Download: stitch ``EPSG:3857`` JPEG tiles of the selected release per year into
a GeoTIFF tagged ``EPSG:3857`` cropped to the AOI. Render reprojects to
``target_srs`` through the existing gdalwarp path.

``provider_options``:

- ``zoom`` explicit zoom (else derived from ``gsd_m`` or ``px_per_meter``)
- ``gsd_m`` requested ground GSD in metres
- ``min_zoom`` / ``max_zoom`` clamp for derived zoom (default 12 / 18)
- ``max_tiles`` per-version tile cap (default 1024); derived zoom steps down to fit
- ``max_total_tiles`` cap across all years in one download (default 12000)
- ``probe_zoom`` zoom used for local-change probes (default ``zoom`` or 17)
- ``probe`` ``grid`` (center + 4 corners, default) or ``center``
- ``dedupe_mode`` ``tilemap`` (default; falls back to ``content_hash``),
  ``content_hash`` or ``none``
- ``collapse_same_capture`` merge versions with identical AOI capture metadata (default true)
- ``release_date_start`` / ``release_date_end`` / ``release_numbers`` /
  ``exclude_release_numbers`` release filter
- ``max_versions`` cap on versions whose metadata is queried (default 80)
- ``index_sleep_min`` / ``index_sleep_max`` jitter for index requests (default 0.2 / 0.6 s)
- ``user_agent`` override the descriptive User-Agent
- ``capabilities_url`` / ``config_url`` / ``tilemap_url_template`` endpoint overrides
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import httpx

from satmap_dataset.config import DownloadConfig, IndexConfig
from satmap_dataset.geoportal.http import RetryPolicy
from satmap_dataset.models import (
    IndexManifest,
    LayerManifest,
    TileAcquisitionMetadata,
    YearAvailabilityReport,
    YearStatus,
)
from satmap_dataset.pipeline.validator import evaluate_year_policy
from satmap_dataset.providers.base import Provider
from satmap_dataset.providers.esri_wayback import catalog, tiles, versions
from satmap_dataset.providers.esri_wayback.client import (
    DEFAULT_USER_AGENT,
    PoliteClient,
    WaybackHTTPError,
    metadata_layer_id_for_zoom,
)

PROVIDER_NAME = "esri_wayback"
DEFAULT_PROBE_ZOOM = 17
DEFAULT_MIN_ZOOM = 12
DEFAULT_MAX_ZOOM = 18
DEFAULT_MAX_TILES = 1024
DEFAULT_MAX_TOTAL_TILES = 12000
DEFAULT_MAX_VERSIONS = 80
DEFAULT_MAX_TILEMAP_REQUESTS_PER_PROBE = 250
DEDUPE_MODES = {"tilemap", "content_hash", "none"}
ATTRIBUTION = (
    "Esri, Vantor, Earthstar Geographics, and the GIS User Community "
    "(Esri World Imagery Wayback)"
)
LICENSE_NOTICE = (
    "Esri World Imagery Wayback is licensed under the Esri Master Agreement. "
    "Its terms prohibit scraping/downloading/storing basemap tiles outside Esri "
    "Content Packages and using the data to train AI/ML systems outside Esri "
    "software. Experimental; see docs/DATA_LICENSING.md."
)

logger = logging.getLogger("satmap_dataset.esri_wayback")


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


def _dedupe_mode(options: dict[str, Any]) -> str:
    mode = str(_opt(options, "dedupe_mode", "tilemap")).strip().lower()
    if mode not in DEDUPE_MODES:
        raise ValueError(f"dedupe_mode must be one of {sorted(DEDUPE_MODES)}, got {mode!r}")
    return mode


def _bool_opt(options: dict[str, Any], key: str, default: bool) -> bool:
    value = options.get(key)
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _http_client(options: dict[str, Any], *, timeout: float, concurrency: int = 2) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(timeout=timeout, connect=min(timeout, 20.0)),
        limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency),
        headers={"User-Agent": str(_opt(options, "user_agent", DEFAULT_USER_AGENT))},
    )


def resolve_download_zoom(
    options: dict[str, Any],
    *,
    px_per_meter: float,
    aoi: tiles.MercatorBBox,
) -> tuple[int, list[str]]:
    """Pick the download zoom: explicit ``zoom`` > ``gsd_m`` > ``1/px_per_meter``, capped by ``max_tiles``."""
    notes: list[str] = []
    max_tiles = int(_opt(options, "max_tiles", DEFAULT_MAX_TILES))
    _, lat = tiles.mercator_to_lonlat(*aoi.center)
    if options.get("zoom") not in (None, ""):
        z = int(options["zoom"])
        if not 0 <= z <= 23:
            raise ValueError("zoom must be within 0..23")
        count = tiles.tile_range_for_bbox(aoi, z).count
        if count > max_tiles:
            raise ValueError(
                f"zoom={z} needs {count} tiles per version for this AOI (max_tiles={max_tiles}); "
                "lower zoom, shrink the AOI, or raise max_tiles"
            )
        return z, notes
    min_zoom = int(_opt(options, "min_zoom", DEFAULT_MIN_ZOOM))
    max_zoom = int(_opt(options, "max_zoom", DEFAULT_MAX_ZOOM))
    if min_zoom > max_zoom:
        raise ValueError("min_zoom must be <= max_zoom")
    gsd = float(options["gsd_m"]) if options.get("gsd_m") not in (None, "") else 1.0 / float(px_per_meter)
    z = tiles.zoom_for_gsd(gsd, lat, min_zoom=min_zoom, max_zoom=max_zoom)
    while z > min_zoom and tiles.tile_range_for_bbox(aoi, z).count > max_tiles:
        notes.append(f"zoom {z} exceeds max_tiles={max_tiles}; stepping down")
        z -= 1
    if tiles.tile_range_for_bbox(aoi, z).count > max_tiles:
        raise ValueError(f"AOI needs more than max_tiles={max_tiles} even at min_zoom={min_zoom}")
    return z, notes


class EsriWaybackProvider(Provider):
    name = PROVIDER_NAME
    default_target_srs = "EPSG:3857"

    # ------------------------------------------------------------------ index
    def index(self, config: IndexConfig) -> tuple[int, Path]:
        return asyncio.run(self._index_async(config))

    async def _index_async(self, config: IndexConfig) -> tuple[int, Path]:
        options = dict(config.provider_options)
        warnings: list[str] = [LICENSE_NOTICE]
        errors: list[str] = []
        caps_url = str(_opt(options, "capabilities_url", os.environ.get("SATMAP_WAYBACK_CAPABILITIES_URL") or catalog.DEFAULT_CAPABILITIES_URL))
        config_url = str(_opt(options, "config_url", os.environ.get("SATMAP_WAYBACK_CONFIG_URL") or catalog.DEFAULT_CONFIG_URL))
        tilemap_template = str(_opt(options, "tilemap_url_template", catalog.DEFAULT_TILEMAP_URL_TEMPLATE))
        dedupe_mode = _dedupe_mode(options)
        probe_mode = str(_opt(options, "probe", "grid"))
        release_filter = catalog.ReleaseFilter.from_options(options)
        max_versions = int(_opt(options, "max_versions", DEFAULT_MAX_VERSIONS))
        max_probe_requests = int(_opt(options, "max_tilemap_requests", DEFAULT_MAX_TILEMAP_REQUESTS_PER_PROBE))
        probe_zoom = int(_opt(options, "probe_zoom", _opt(options, "zoom", DEFAULT_PROBE_ZOOM)))
        aoi = tiles.aoi_to_mercator_bbox(_parse_bbox(config.bbox), config.srs)
        probes = tiles.probe_tiles(aoi, probe_zoom, probe_mode)
        metadata_layer = metadata_layer_id_for_zoom(probe_zoom)

        all_releases: list[catalog.WaybackRelease] = []
        filtered: list[catalog.WaybackRelease] = []
        version_list: list[versions.ImageryVersion] = []
        effective_mode = dedupe_mode
        request_counts: dict[str, int] = {}

        timeout = float(_opt(options, "timeout", 60.0))
        async with _http_client(options, timeout=timeout) as http:
            client = PoliteClient(
                http,
                sleep_min=float(_opt(options, "index_sleep_min", 0.2)),
                sleep_max=float(_opt(options, "index_sleep_max", 0.6)),
                retry_policy=RetryPolicy(max_attempts=int(_opt(options, "retry_max_attempts", 4))),
            )
            try:
                caps_resp = await client.get(caps_url)
                if caps_resp is None:
                    raise WaybackHTTPError(f"GetCapabilities returned 4xx: {caps_url}")
                all_releases = catalog.parse_capabilities(caps_resp.content)
                try:
                    all_releases = catalog.apply_config(all_releases, await client.get_json(config_url))
                except (WaybackHTTPError, ValueError) as exc:
                    warnings.append(f"waybackconfig.json unavailable ({exc}); metadata URLs derived from layer identifiers")
            except (WaybackHTTPError, catalog.CatalogParseError) as exc:
                errors.append(f"Wayback catalog failed: {exc}")

            filtered = [r for r in all_releases if release_filter.accepts(r)]
            by_num = {r.release_num: r for r in all_releases}
            if all_releases and not filtered:
                errors.append("Release filter excluded every Wayback release.")

            version_nums: list[int] = []
            if filtered:
                if dedupe_mode == "tilemap":
                    try:
                        for probe in probes:
                            origins, n = await versions.walk_local_changes(
                                filtered,
                                probe,
                                lambda rel, t: client.tilemap(rel, t, template=tilemap_template),
                                max_requests=max_probe_requests,
                            )
                            request_counts[f"tilemap_{probe[1]}_{probe[2]}"] = n
                            version_nums.extend(o for o in origins if o not in version_nums)
                    except WaybackHTTPError as exc:
                        warnings.append(f"tilemap unavailable ({exc}); falling back to tile content hash")
                        effective_mode = "content_hash"
                        version_nums = []
                if effective_mode == "content_hash":
                    observations: dict[int, tuple[str, int | None]] = {}
                    center = probes[0]
                    for release in filtered[: max_probe_requests]:
                        try:
                            obs = await client.tile_digest(release, center)
                        except WaybackHTTPError as exc:
                            warnings.append(f"content hash failed for release {release.release_num}: {exc}")
                            continue
                        if obs is not None:
                            observations[release.release_num] = obs
                    request_counts["content_hash"] = len(observations)
                    version_nums = versions.group_by_content_hash(filtered, observations)
                elif effective_mode == "none":
                    version_nums = [r.release_num for r in filtered]

            version_list = versions.assign_represented_releases(version_nums, filtered, by_num)
            if len(version_list) > max_versions:
                warnings.append(
                    f"{len(version_list)} distinct versions exceed max_versions={max_versions}; keeping the newest"
                )
                version_list = version_list[:max_versions]

            cx, cy = aoi.center
            envelope = f"{aoi.min_x},{aoi.min_y},{aoi.max_x},{aoi.max_y}"
            metadata_failures = 0
            for version in version_list:
                try:
                    point = await client.metadata_query(
                        version.release,
                        layer_id=metadata_layer,
                        geometry=f"{cx},{cy}",
                        geometry_type="esriGeometryPoint",
                    )
                    env = await client.metadata_query(
                        version.release,
                        layer_id=metadata_layer,
                        geometry=envelope,
                        geometry_type="esriGeometryEnvelope",
                    )
                    version.capture = versions.capture_from_features(point, env)
                except WaybackHTTPError as exc:
                    metadata_failures += 1
                    logger.warning("metadata failed for release %s: %s", version.release.release_num, exc)
            if metadata_failures:
                warnings.append(
                    f"capture metadata unavailable for {metadata_failures} version(s); "
                    "their capture year falls back to the release date"
                )
            request_counts["total"] = client.requests_made

        distinct_before_collapse = len(version_list)
        if _bool_opt(options, "collapse_same_capture", True):
            version_list = versions.collapse_same_capture(version_list)

        selections = versions.select_per_capture_year(version_list, config.requested_years)
        selected_nums = {s.selected.release.release_num for s in selections.values()}

        year_statuses: list[YearStatus] = []
        tile_sources_by_year: dict[int, dict[str, str]] = {}
        tile_bboxes_by_year: dict[int, dict[str, list[float]]] = {}
        tile_acquisition_by_year: dict[int, dict[str, TileAcquisitionMetadata]] = {}
        years_excluded: dict[int, str] = {}
        capture_years_all = sorted({v.capture_year for v in version_list})
        for year in config.requested_years:
            selection = selections.get(year)
            if selection is None:
                reason = "no_distinct_wayback_version_captured_in_year"
                years_excluded[year] = reason
                year_statuses.append(
                    YearStatus(year=year, typename_exists=False, feature_count=0, status="no_typename", reason=reason)
                )
                continue
            chosen = selection.selected
            tile_id = f"wayback_{chosen.release.release_num}"
            tile_sources_by_year[year] = {tile_id: chosen.release.tile_url_template.replace("{release}", str(chosen.release.release_num))}
            tile_bboxes_by_year[year] = {tile_id: list(_parse_bbox(config.bbox))}
            tile_acquisition_by_year[year] = {
                tile_id: TileAcquisitionMetadata(
                    acquisition_date=chosen.effective_capture_date,
                    publication_date=chosen.release.release_date,
                    acquisition_year=year,
                    gsd=chosen.capture.resolution_m,
                )
            }
            year_statuses.append(
                YearStatus(
                    year=year,
                    typename_exists=True,
                    feature_count=1 + len(selection.alternatives),
                    status="has_features",
                    reason=f"capture_date={chosen.effective_capture_date} ({chosen.capture_date_source})",
                )
            )

        years_included = sorted(tile_sources_by_year)
        policy = evaluate_year_policy(
            requested_years=config.requested_years,
            available_years=years_included,
            strict_years=config.strict_years,
            min_years=config.min_years,
        )
        combined_errors = errors + list(policy.errors)
        if not years_included and not combined_errors:
            combined_errors.append("No distinct Wayback imagery version has a capture year in the requested range.")

        version_dicts = []
        for v in version_list:
            row = v.as_dict()
            row["selected"] = v.release.release_num in selected_nums
            version_dicts.append(row)

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
            years_available_wfs=capture_years_all,
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
                "attribution": ATTRIBUTION,
                "license_notice": LICENSE_NOTICE,
                "capabilities_url": caps_url,
                "config_url": config_url,
                "releases_total": len(all_releases),
                "releases_considered": len(filtered),
                "release_filter": release_filter.as_dict(),
                "dedupe_mode": dedupe_mode,
                "dedupe_mode_effective": effective_mode,
                "collapse_same_capture": _bool_opt(options, "collapse_same_capture", True),
                "probe": probe_mode,
                "probe_zoom": probe_zoom,
                "probe_tiles": [list(p) for p in probes],
                "metadata_layer_id": metadata_layer,
                "aoi_bbox_3857": aoi.as_list(),
                "distinct_versions_tilemap": distinct_before_collapse,
                "distinct_versions": len(version_list),
                "capture_years": capture_years_all,
                "versions": version_dicts,
                "selection_by_year": {str(y): s.as_dict() for y, s in selections.items()},
                "request_counts": request_counts,
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
        config.year_availability_output_json.write_text(availability.model_dump_json(indent=2), encoding="utf-8")
        config.output_json.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        logger.info(
            "Wayback index: releases=%s considered=%s versions=%s (tilemap=%s) capture_years=%s included=%s passed=%s",
            len(all_releases),
            len(filtered),
            len(version_list),
            distinct_before_collapse,
            capture_years_all,
            years_included,
            manifest.passed,
        )
        return (0 if manifest.passed else 1), config.output_json

    # --------------------------------------------------------------- download
    def download(self, config: DownloadConfig) -> tuple[int, Path]:
        return asyncio.run(self._download_async(config))

    async def _download_async(self, config: DownloadConfig) -> tuple[int, Path]:
        index_manifest = IndexManifest.model_validate_json(config.index_manifest.read_text(encoding="utf-8"))
        options = dict(config.provider_options)
        if config.bbox is None:
            raise ValueError("bbox is required for esri_wayback download")
        aoi = tiles.aoi_to_mercator_bbox(_parse_bbox(config.bbox), config.srs)
        errors: list[str] = []
        notes: list[str] = []
        try:
            zoom, zoom_notes = resolve_download_zoom(options, px_per_meter=config.px_per_meter, aoi=aoi)
            notes.extend(zoom_notes)
        except ValueError as exc:
            return self._write_download_manifest(
                config, index_manifest, assets=[], years_source_map={}, errors=[str(exc)], details={}, zoom=None
            )

        # Pad by two pixels so reprojection to target_srs never samples outside data.
        pad = 2 * tiles.resolution(zoom)
        crop = tiles.pad_bbox(aoi, pad)
        tile_range = tiles.tile_range_for_bbox(crop, zoom)
        years = list(index_manifest.years_included)
        total = tile_range.count * len(years)
        max_total = int(_opt(options, "max_total_tiles", DEFAULT_MAX_TOTAL_TILES))
        if total > max_total:
            return self._write_download_manifest(
                config,
                index_manifest,
                assets=[],
                years_source_map={},
                errors=[f"{total} tiles across {len(years)} years exceed max_total_tiles={max_total}"],
                details={},
                zoom=zoom,
            )

        assets: list[str] = []
        years_source_map: dict[int, str] = {}
        details: dict[str, Any] = {}
        semaphore = asyncio.Semaphore(max(1, config.concurrency))
        async with _http_client(options, timeout=config.timeout, concurrency=config.concurrency) as http:
            client = PoliteClient(
                http,
                sleep_min=config.sleep_min,
                sleep_max=config.sleep_max,
                retry_policy=RetryPolicy(
                    max_attempts=config.retries + 1, backoff_seconds=config.retry_delay
                ),
            )
            for year in years:
                sources = index_manifest.tile_sources_by_year.get(year) or {}
                if not sources:
                    errors.append(f"year {year}: no tile source in index manifest")
                    continue
                tile_id, template = next(iter(sorted(sources.items())))
                output_path = config.download_root / str(year) / f"{tile_id}_z{zoom}.tif"
                if output_path.exists() and output_path.stat().st_size > 0 and not config.overwrite:
                    assets.append(str(output_path))
                    years_source_map[year] = "wmts"
                    details[str(year)] = {"tile_id": tile_id, "path": str(output_path), "reused": True}
                    continue
                try:
                    detail = await self._fetch_and_stitch(
                        client, semaphore, template, tile_range, crop, output_path, options
                    )
                except (WaybackHTTPError, ValueError, OSError) as exc:
                    errors.append(f"year {year}: {exc}")
                    continue
                detail["tile_id"] = tile_id
                details[str(year)] = detail
                assets.append(str(output_path))
                years_source_map[year] = "wmts"
            request_total = client.requests_made

        details["_summary"] = {
            "zoom": zoom,
            "tiles_per_version": tile_range.count,
            "ground_gsd_m": round(tiles.ground_gsd(zoom, tiles.mercator_to_lonlat(*aoi.center)[1]), 4),
            "mercator_pixel_size_m": tiles.resolution(zoom),
            "requests": request_total,
            "notes": notes,
        }
        return self._write_download_manifest(
            config, index_manifest, assets=assets, years_source_map=years_source_map, errors=errors, details=details, zoom=zoom
        )

    async def _fetch_and_stitch(
        self,
        client: PoliteClient,
        semaphore: asyncio.Semaphore,
        template: str,
        tile_range: tiles.TileRange,
        crop: tiles.MercatorBBox,
        output_path: Path,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        z = tile_range.z

        async def fetch(xy: tuple[int, int]):
            x, y = xy
            async with semaphore:
                response = await client.get(template.format(z=z, y=y, x=x))
            if response is None:
                return xy, None
            return xy, tiles.decode_tile(response.content)

        results = await asyncio.gather(*(fetch(xy) for xy in tile_range.tiles()))
        tile_arrays = dict(results)
        missing = sum(1 for arr in tile_arrays.values() if arr is None)
        max_missing = float(_opt(options, "max_missing_tile_fraction", 0.05))
        if missing / max(1, tile_range.count) > max_missing:
            raise ValueError(
                f"{missing}/{tile_range.count} tiles missing at z{z} (max_missing_tile_fraction={max_missing})"
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
        passed = bool(assets) and not errors and len(years_included) == len(index_manifest.years_included)
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
        logger.info("Wayback download: assets=%s errors=%s passed=%s", len(assets), len(errors), passed)
        return (0 if passed else 1), config.output_json
