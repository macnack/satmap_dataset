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
from satmap_dataset.models import DemManifest, DemProductAsset

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
        from satmap_dataset.models import DatasetManifest

        manifest = DatasetManifest.model_validate_json(
            Path(config.render_manifest).read_text(encoding="utf-8")
        )
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


async def _run_async(config: DemConfig) -> tuple[int, Path]:
    retry_policy = RetryPolicy(max_attempts=config.retries, backoff_seconds=config.retry_delay)
    grid = _resolve_align_grid(config) if config.align_to_render else None
    resample = str(config.provider_options.get("resample", "bilinear"))
    product_assets: list[DemProductAsset] = []
    errors: list[str] = []

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
    manifest = DemManifest(
        provider="geoportal",
        bbox=config.bbox,
        srs=config.srs,
        vertical_datum=config.vertical_datum,
        products=product_assets,
        align_to_render=config.align_to_render,
        passed=passed,
        notes="WCS GRID1 serves a current-best 1 m composite; not year-aware.",
        errors=errors,
        run_parameters=config.model_dump(mode="json"),
    )
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
