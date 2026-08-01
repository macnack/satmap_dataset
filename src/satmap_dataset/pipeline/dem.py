from __future__ import annotations

import asyncio
import logging
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

from satmap_dataset.config import DemConfig
from satmap_dataset.geoportal import wcs_client
from satmap_dataset.geoportal.http import RetryPolicy
from satmap_dataset.models import (
    DemProductAsset,
    LayerManifest,
    LayerYearAsset,
    ReferenceGrid,
)

logger = logging.getLogger("satmap_dataset.dem")


def _parse_bbox(value: str) -> tuple[float, float, float, float]:
    parts = [float(p.strip()) for p in value.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must have format xmin,ymin,xmax,ymax")
    return (parts[0], parts[1], parts[2], parts[3])


def _tool_path(name: str) -> str | None:
    return shutil.which(name)


async def _fetch_tiles_for_product(
    config: DemConfig, product: str, dest_dir: Path, *, retry_policy: RetryPolicy
) -> list[Path]:
    options = dict(config.provider_options)
    endpoint = wcs_client.endpoint_url(product, options)
    cov = wcs_client.coverage_id(product, config.vertical_datum, options)
    sub_bboxes = wcs_client.split_bbox(_parse_bbox(config.bbox), config.max_request_px)
    tiles: list[Path] = []
    for i, sub in enumerate(sub_bboxes):
        if config.sleep_max > 0:
            await asyncio.sleep(random.uniform(config.sleep_min, config.sleep_max))
        data = await wcs_client.get_coverage(
            endpoint, cov, sub, config.srs,
            options=options, timeout=config.timeout, retry_policy=retry_policy,
        )
        out = dest_dir / f"{product}_{i:04d}.tif"
        out.write_bytes(data)
        tiles.append(out)
    return tiles


def _merge_tiles(tiles: list[Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(tiles) == 1:
        shutil.copyfile(tiles[0], out_path)
        return
    gdalbuildvrt = _tool_path("gdalbuildvrt")
    gdal_translate = _tool_path("gdal_translate")
    if not gdalbuildvrt or not gdal_translate:
        raise RuntimeError(
            "Merging tiled WCS output requires the GDAL CLI (gdalbuildvrt, "
            "gdal_translate). Install GDAL or reduce the AOI below max_request_px."
        )
    vrt_path = out_path.with_suffix(".vrt")
    try:
        subprocess.run(
            [gdalbuildvrt, str(vrt_path), *[str(t) for t in tiles]],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(
            [gdal_translate, "-co", "COMPRESS=DEFLATE", str(vrt_path), str(out_path)],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"GDAL merge failed: {(exc.stderr or '')[-500:]}") from exc
    finally:
        vrt_path.unlink(missing_ok=True)


def _normalise_elevation_raster(path: Path) -> str | None:
    """Collapse a multi-band elevation response to one Float32 band, in place.

    The GUGiK NMPT service answers ``FORMAT=image/tiff`` with a 3-band Byte
    raster whose bands are identical and whose values are the elevation rounded
    to whole metres (verified 2026-08-01 against the same window fetched as
    ``image/x-aaigrid``: |diff| p95 = 0.47 m). Consumers of this pipeline read
    the declared ``DEM_F32`` profile, so the raster is rewritten as a single
    Float32 band and the lost precision is reported rather than hidden. Pass
    ``provider_options={"format": "image/x-aaigrid"}`` to fetch full float
    precision instead (~20x the bytes on the wire).

    Returns a warning string when something was changed or looks wrong, and
    None when the raster already is single-band elevation data.
    """
    try:
        import numpy as np
        import tifffile

        arr = np.asarray(tifffile.imread(str(path)))
    except Exception:  # noqa: BLE001 - diagnostics only, never fail the fetch
        logger.warning("Could not inspect %s for band normalisation.", path)
        return None

    if arr.ndim != 3 or arr.shape[-1] < 2:
        return None
    if not all(np.array_equal(arr[..., 0], arr[..., b]) for b in range(1, arr.shape[-1])):
        return (
            f"{path.name}: {arr.shape[-1]}-band response whose bands differ; left as "
            "fetched, so it is NOT the single-band Float32 the DEM_F32 profile promises"
        )

    gdal_translate = _tool_path("gdal_translate")
    if not gdal_translate:
        return (
            f"{path.name}: {arr.shape[-1]}-band elevation response could not be collapsed "
            "to single-band Float32 (GDAL CLI not found); the DEM_F32 profile is not met"
        )
    tmp_path = path.with_suffix(".normalised.tif")
    try:
        subprocess.run(
            [
                gdal_translate, "-b", "1", "-ot", "Float32",
                "-co", "COMPRESS=DEFLATE", str(path), str(tmp_path),
            ],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as exc:
        tmp_path.unlink(missing_ok=True)
        return f"{path.name}: band normalisation failed: {(exc.stderr or '')[-200:]}"
    tmp_path.replace(path)
    quantised = np.issubdtype(arr.dtype, np.integer)
    return (
        f"{path.name}: service returned {arr.shape[-1]} identical {arr.dtype} bands; "
        "collapsed to single-band Float32"
        + (
            " — elevations are quantised to 1 m, request format image/x-aaigrid for "
            "full precision" if quantised else ""
        )
    )


def _align_to_grid(
    native: Path, out_path: Path, *,
    target_bbox: tuple[float, float, float, float],
    target_width: int, target_height: int, srs: str, resample: str = "bilinear",
) -> None:
    gdalwarp = _tool_path("gdalwarp")
    if not gdalwarp:
        raise RuntimeError(
            "Aligning the DEM to the render grid requires the GDAL CLI (gdalwarp). "
            "Install GDAL or set align_to_render=false."
        )
    xmin, ymin, xmax, ymax = target_bbox
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                gdalwarp, "-t_srs", srs,
                "-te", str(xmin), str(ymin), str(xmax), str(ymax),
                "-ts", str(target_width), str(target_height),
                "-r", resample, "-co", "COMPRESS=DEFLATE", "-overwrite",
                str(native), str(out_path),
            ],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"gdalwarp alignment failed: {(exc.stderr or '')[-500:]}") from exc


def _raster_dims(path: Path) -> tuple[int | None, int | None]:
    try:
        import tifffile

        with tifffile.TiffFile(str(path)) as tif:
            page = tif.pages[0]
            return int(page.imagewidth), int(page.imagelength)
    except Exception:  # best-effort; dims are informational
        return (None, None)


def _read_nodata(path: Path) -> float | None:
    """Read the GDAL_NODATA value (TIFF tag 42113) if present."""
    try:
        import tifffile

        with tifffile.TiffFile(str(path)) as tif:
            tag = tif.pages[0].tags.get(42113)
            if tag is not None and tag.value not in (None, ""):
                return float(str(tag.value).strip())
    except Exception:  # best-effort; absence is fine
        return None
    return None


def _coverage_is_empty(path: Path) -> bool:
    try:
        import numpy as np
        import tifffile

        arr = np.asarray(tifffile.imread(str(path)), dtype="float64")
    except Exception:  # cannot read -> don't block
        logger.warning("Could not read %s for emptiness check; treating as non-empty.", path)
        return False
    if arr.size == 0:
        return True
    finite = np.isfinite(arr)
    if not finite.any():
        return True
    nodata = _read_nodata(path)
    if nodata is not None:
        valid = finite & (arr != nodata)
        return not bool(valid.any())
    return False


def _resolve_align_grid(
    config: DemConfig,
) -> tuple[tuple[float, float, float, float], int, int]:
    if config.render_manifest and Path(config.render_manifest).exists():
        manifest = LayerManifest.load(config.render_manifest)
        grid = manifest.grid
        if grid is not None:
            return (_parse_bbox(grid.bbox), int(grid.width), int(grid.height))
        if manifest.target_bbox and manifest.target_width and manifest.target_height:
            return (
                _parse_bbox(manifest.target_bbox),
                int(manifest.target_width),
                int(manifest.target_height),
            )
    bbox = _parse_bbox(config.target_bbox or config.bbox)
    if config.target_width and config.target_height:
        return (bbox, int(config.target_width), int(config.target_height))
    xmin, ymin, xmax, ymax = bbox
    width = max(1, round((xmax - xmin) * config.px_per_meter))
    height = max(1, round((ymax - ymin) * config.px_per_meter))
    return (bbox, width, height)


def _grid_to_reference(
    grid: tuple[tuple[float, float, float, float], int, int] | None, srs: str
) -> ReferenceGrid | None:
    if grid is None:
        return None
    (xmin, ymin, xmax, ymax), width, height = grid
    return ReferenceGrid(
        bbox=f"{xmin},{ymin},{xmax},{ymax}", width=width, height=height, srs=srs
    )


def build_dem_layer_manifest(
    config: DemConfig,
    product_assets: list[DemProductAsset],
    *,
    transport: str,
    years_skipped: dict[int, str],
    grid: tuple[tuple[float, float, float, float], int, int] | None,
    passed: bool,
    errors: list[str],
    notes: str | None,
) -> LayerManifest:
    """Map the product/year-centric DEM result onto the unified LayerManifest.

    Rich per-product detail (coverage_id, endpoint, native/aligned dims, godla)
    is preserved verbatim under ``provider_metadata['products']``; the flat
    LayerManifest fields carry what the orchestrator and a multi-band assembler
    need. Raster output paths are echoed unchanged.
    """
    assets: list[str] = []
    year_map: dict[int, LayerYearAsset] = {}
    for asset in product_assets:
        if asset.years:
            for ya in asset.years:
                lya = year_map.setdefault(
                    ya.year, LayerYearAsset(year=ya.year, passed=True)
                )
                if ya.native_path:
                    lya.native_paths[asset.product] = ya.native_path
                chosen = ya.aligned_path or ya.native_path
                if chosen:
                    lya.assets[asset.product] = chosen
                    assets.append(chosen)
                if ya.errors:
                    lya.errors.extend(f"{asset.product}: {e}" for e in ya.errors)
                lya.passed = lya.passed and ya.passed
        else:
            chosen = asset.aligned_path or asset.native_path
            if chosen:
                assets.append(chosen)

    years = [year_map[y] for y in sorted(year_map)]
    return LayerManifest(
        layer="dem",
        role="dem",
        stage="dem",
        provider="geoportal",
        grid=_grid_to_reference(grid, config.srs),
        bands=[asset.product for asset in product_assets],
        years_requested=config.requested_years,
        years_included=sorted(year_map),
        years_excluded_with_reason=dict(years_skipped),
        years=years,
        assets=assets,
        target_srs=config.srs,
        pixel_profile="DEM_F32",
        passed=passed,
        notes=notes,
        errors=list(errors),
        run_parameters=config.model_dump(mode="json"),
        provider_metadata={
            "products": [asset.model_dump() for asset in product_assets],
            "vertical_datum": config.vertical_datum,
            "transport": transport,
            "bbox": config.bbox,
            "align_to_render": config.align_to_render,
        },
    )


async def _run_async(config: DemConfig) -> tuple[int, Path]:
    retry_policy = RetryPolicy(max_attempts=config.retries, backoff_seconds=config.retry_delay)
    grid = _resolve_align_grid(config) if config.align_to_render else None
    resample = str(config.provider_options.get("resample", "bilinear"))
    product_assets: list[DemProductAsset] = []
    errors: list[str] = []
    warnings: list[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for product in config.products:
            options = dict(config.provider_options)
            asset = DemProductAsset(
                product=product,
                coverage_id=wcs_client.coverage_id(product, config.vertical_datum, options),
                endpoint=wcs_client.endpoint_url(product, options),
            )
            native_path = config.dem_root / "native" / f"{product}_{config.vertical_datum}.tif"
            try:
                # When reusing an existing native file, tile_count stays 0 (no tiles fetched this run).
                if not (native_path.exists() and not config.overwrite):
                    tiles = await _fetch_tiles_for_product(
                        config, product, tmp_dir, retry_policy=retry_policy
                    )
                    asset.tile_count = len(tiles)
                    _merge_tiles(tiles, native_path)
                    normalisation = _normalise_elevation_raster(native_path)
                    if normalisation:
                        logger.warning("%s", normalisation)
                        warnings.append(normalisation)
                if _coverage_is_empty(native_path):
                    asset.errors.append("coverage empty / nodata-only for AOI")
                    errors.append(f"{product}: empty coverage")
                else:
                    asset.native_path = str(native_path)
                    asset.native_width, asset.native_height = _raster_dims(native_path)
                    asset.nodata = _read_nodata(native_path)
                    if grid is not None:
                        aligned_path = (
                            config.dem_root / "aligned" / f"{product}_{config.vertical_datum}.tif"
                        )
                        target_bbox, gw, gh = grid
                        _align_to_grid(
                            native_path, aligned_path,
                            target_bbox=target_bbox, target_width=gw, target_height=gh,
                            srs=config.srs, resample=resample,
                        )
                        asset.aligned_path = str(aligned_path)
                        asset.aligned_width, asset.aligned_height = gw, gh
                    asset.passed = True
            except Exception as exc:  # noqa: BLE001 - record and continue per-product
                asset.errors.append(str(exc))
                errors.append(f"{product}: {exc}")
            product_assets.append(asset)

    passed = bool(product_assets) and all(a.passed for a in product_assets)
    manifest = build_dem_layer_manifest(
        config,
        product_assets,
        transport="wcs",
        years_skipped={},
        grid=grid,
        passed=passed,
        errors=errors,
        notes="WCS GRID1 serves a current-best 1 m composite; not year-aware.",
    )
    manifest.warnings.extend(warnings)
    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    config.output_json.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    logger.info(
        "DEM run: products=%s passed=%s errors=%s",
        [a.product for a in product_assets], passed, len(errors),
    )
    return (0 if passed else 1), config.output_json


def run(config: DemConfig) -> tuple[int, Path]:
    if config.transport == "skorowidz":
        from satmap_dataset.pipeline import dem_skorowidz

        return dem_skorowidz.run(config)
    return asyncio.run(_run_async(config))
