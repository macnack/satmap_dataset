"""Swedish national DEM via Lantmäteriet STAC-höjd (Markhöjdmodell).

Separate from Polish Geoportal WCS/NMPT and from ortofoto ``stac-bild``.
Stage contract: ``run(config) -> (exit_code, artifact_path)``.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

import httpx

from satmap_dataset.config import DemConfig
from satmap_dataset.geoportal.http import RetryPolicy
from satmap_dataset.models import DemProductAsset
from satmap_dataset.pipeline import dem as dem_common
from satmap_dataset.providers.lantmateriet import dem as lm_dem, stac
from satmap_dataset.providers.lantmateriet.provider import (
    LantmaterietProvider,
    _download_asset_with_retry,
    _parse_bbox,
)

logger = logging.getLogger("satmap_dataset.dem.lantmateriet")


def _clip_to_bbox(
    mosaic: Path,
    out_path: Path,
    *,
    bbox: tuple[float, float, float, float],
    srs: str,
) -> None:
    """Lossless-ish AOI clip via gdalwarp (-te); elev values stay in the band."""
    gdalwarp = dem_common._tool_path("gdalwarp")
    if not gdalwarp:
        raise RuntimeError(
            "Clipping Markhöjdmodell tiles to the AOI requires the GDAL CLI "
            "(gdalwarp). Install GDAL or reduce the AOI to a single tile."
        )
    xmin, ymin, xmax, ymax = bbox
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import subprocess

    try:
        subprocess.run(
            [
                gdalwarp,
                "-t_srs",
                srs,
                "-te",
                str(xmin),
                str(ymin),
                str(xmax),
                str(ymax),
                "-r",
                "bilinear",
                "-co",
                "COMPRESS=DEFLATE",
                "-overwrite",
                str(mosaic),
                str(out_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"gdalwarp AOI clip failed: {(exc.stderr or '')[-500:]}"
        ) from exc


async def _run_async(config: DemConfig) -> tuple[int, Path]:
    options = dict(config.provider_options)
    search_options = lm_dem.resolve_dem_search_options(options)
    aoi = _parse_bbox(config.bbox)
    bbox_for_search, bbox_crs = LantmaterietProvider._bbox_for_search(aoi, config.srs, options)
    retry_policy = RetryPolicy(max_attempts=config.retries, backoff_seconds=config.retry_delay)

    errors: list[str] = []
    warnings: list[str] = []
    product = "nmt"
    asset = DemProductAsset(
        product=product,
        coverage_id=str(search_options.collections[0] if search_options.collections else lm_dem.DEFAULT_STAC_HOJD_COLLECTION),
        endpoint=search_options.url,
    )

    try:
        items = await stac.search_features(
            search_options,
            bbox=bbox_for_search,
            bbox_crs=bbox_crs,
            datetime_range=(
                str(options["datetime_range"]) if options.get("datetime_range") else None
            ),
            timeout=config.timeout,
            retry_policy=retry_policy,
        )
    except httpx.HTTPError as exc:
        items = []
        errors.append(f"STAC-höjd search failed: {exc}")

    pairs = lm_dem.items_with_raster_assets(items)
    if not pairs and not errors:
        errors.append(
            "STAC-höjd returned no Markhöjdmodell raster items for the AOI. "
            f"url={search_options.url} collections={list(search_options.collections)}"
        )

    grid = dem_common._resolve_align_grid(config) if config.align_to_render else None
    resample = str(options.get("resample", "bilinear"))
    native_path = config.dem_root / "native" / f"{product}_{config.vertical_datum}.tif"

    if pairs and (config.overwrite or not native_path.exists()):
        headers = lm_dem.auth_headers(options)
        if "Authorization" not in headers:
            errors.append(
                "Markhöjdmodell asset download requires Geotorget credentials. "
                "Set SATMAP_LANTMATERIET_DEM_USERNAME/PASSWORD (preferred) or "
                "SATMAP_LANTMATERIET_USERNAME/PASSWORD for a Markhöjdmodell "
                "Nedladdning subscription (ortofoto credentials usually do not work)."
            )
        else:
            timeout = httpx.Timeout(timeout=config.timeout, connect=min(config.timeout, 20.0))
            with tempfile.TemporaryDirectory() as tmp:
                tmp_dir = Path(tmp)
                tiles: list[Path] = []
                failed: list[str] = []
                async with httpx.AsyncClient(
                    follow_redirects=True, timeout=timeout, headers=headers
                ) as client:
                    for item, stac_asset in pairs:
                        dest = tmp_dir / lm_dem.filename_for_item(item, stac_asset)
                        ok = await _download_asset_with_retry(
                            client,
                            stac_asset.href,
                            dest,
                            retries=config.retries,
                            retry_delay=config.retry_delay,
                            sleep_min=config.sleep_min,
                            sleep_max=config.sleep_max,
                        )
                        if ok:
                            tiles.append(dest)
                        else:
                            failed.append(stac_asset.href)
                asset.tile_count = len(tiles)
                if failed:
                    errors.append(f"{product}: failed to download {len(failed)} tile(s)")
                if tiles:
                    mosaic = tmp_dir / "mosaic.tif"
                    dem_common._merge_tiles(tiles, mosaic)
                    _clip_to_bbox(mosaic, native_path, bbox=aoi, srs=config.srs)
                elif not errors:
                    errors.append(f"{product}: no tiles downloaded")

    if native_path.exists() and not errors:
        normalisation = dem_common._normalise_elevation_raster(native_path)
        if normalisation:
            warnings.append(normalisation)
        if dem_common._coverage_is_empty(native_path):
            asset.errors.append("coverage empty / nodata-only for AOI")
            errors.append(f"{product}: empty coverage")
        else:
            asset.native_path = str(native_path)
            asset.native_width, asset.native_height = dem_common._raster_dims(native_path)
            asset.nodata = dem_common._read_nodata(native_path)
            if grid is not None:
                aligned_path = (
                    config.dem_root / "aligned" / f"{product}_{config.vertical_datum}.tif"
                )
                target_bbox, gw, gh = grid
                dem_common._align_to_grid(
                    native_path,
                    aligned_path,
                    target_bbox=target_bbox,
                    target_width=gw,
                    target_height=gh,
                    srs=config.srs,
                    resample=resample,
                )
                asset.aligned_path = str(aligned_path)
                asset.aligned_width, asset.aligned_height = gw, gh
            asset.passed = True
    elif not native_path.exists() and not errors:
        errors.append(f"{product}: native DEM missing at {native_path}")

    if errors and not asset.passed:
        asset.errors.extend(errors)

    passed = asset.passed and not errors
    notes = (
        "Lantmäteriet Markhöjdmodell 1 m DTM via STAC-höjd "
        f"({search_options.url}, collections={list(search_options.collections)}); "
        f"vertical datum RH2000; attribution {lm_dem.DEFAULT_ATTRIBUTION}; "
        f"license {lm_dem.DEFAULT_LICENSE}. Not year-aware composite tiles."
    )
    manifest = dem_common.build_dem_layer_manifest(
        config,
        [asset],
        transport="stac_hojd",
        years_skipped={},
        grid=grid,
        passed=passed,
        errors=errors,
        notes=notes,
    )
    manifest.warnings.extend(warnings)
    meta = dict(manifest.provider_metadata or {})
    meta.update(
        {
            "stac_hojd_url": search_options.url,
            "stac_hojd_collections": list(search_options.collections),
            "attribution": lm_dem.DEFAULT_ATTRIBUTION,
            "license": lm_dem.DEFAULT_LICENSE,
            "product_name": "Markhöjdmodell Nedladdning",
            "search_bbox_wgs84": list(bbox_for_search),
            "item_ids": [item.item_id for item, _ in pairs],
        }
    )
    manifest.provider_metadata = meta

    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    config.output_json.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    logger.info(
        "Lantmäteriet DEM: items=%s passed=%s errors=%s",
        len(pairs),
        passed,
        len(errors),
    )
    return (0 if passed else 1), config.output_json


def run(config: DemConfig) -> tuple[int, Path]:
    if config.provider != "lantmateriet":
        raise ValueError("dem_lantmateriet.run requires provider='lantmateriet'")
    return asyncio.run(_run_async(config))
