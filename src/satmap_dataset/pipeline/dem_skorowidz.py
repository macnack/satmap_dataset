from __future__ import annotations

import asyncio
import logging
import subprocess
import tempfile
from pathlib import Path

import httpx

from satmap_dataset.config import DemConfig
from satmap_dataset.geoportal import dem_skorowidz_client
from satmap_dataset.geoportal.http import RetryPolicy
from satmap_dataset.models import DemManifest, DemProductAsset, DemYearAsset
from satmap_dataset.pipeline import dem
from satmap_dataset.providers.lantmateriet.provider import _download_asset_with_retry

logger = logging.getLogger("satmap_dataset.dem_skorowidz")


async def _download_tiles(
    urls: list[str], dest_dir: Path, config: DemConfig, retry_policy: RetryPolicy
) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    timeout = httpx.Timeout(timeout=config.timeout, connect=min(config.timeout, 20.0))
    headers = {"User-Agent": "satmap_dataset/0.1"}
    paths: list[Path] = []
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout, headers=headers) as client:
        for url in urls:
            out = dest_dir / Path(url).name
            ok = await _download_asset_with_retry(
                client, url, out,
                retries=config.retries, retry_delay=config.retry_delay,
                sleep_min=config.sleep_min, sleep_max=config.sleep_max,
            )
            if not ok:
                raise RuntimeError(f"download failed: {url}")
            paths.append(out)
    return paths


def _mosaic_asc_to_native(tiles: list[Path], out_path: Path, bbox: tuple[float, float, float, float]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    buildvrt = dem._tool_path("gdalbuildvrt")
    translate = dem._tool_path("gdal_translate")
    if not buildvrt or not translate:
        raise RuntimeError(
            "Mosaicking .asc tiles requires the GDAL CLI (gdalbuildvrt, gdal_translate). Install GDAL."
        )
    xmin, ymin, xmax, ymax = bbox
    vrt_path = out_path.with_suffix(".vrt")
    try:
        subprocess.run(
            [buildvrt, "-a_srs", "EPSG:2180", str(vrt_path), *[str(t) for t in tiles]],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(
            [
                translate, "-a_srs", "EPSG:2180",
                "-projwin", str(xmin), str(ymax), str(xmax), str(ymin),
                "-co", "COMPRESS=DEFLATE", str(vrt_path), str(out_path),
            ],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"GDAL .asc mosaic failed: {(exc.stderr or '')[-500:]}") from exc
    finally:
        vrt_path.unlink(missing_ok=True)


async def _run_async(config: DemConfig) -> tuple[int, Path]:
    retry_policy = RetryPolicy(max_attempts=config.retries, backoff_seconds=config.retry_delay)
    grid = dem._resolve_align_grid(config) if config.align_to_render else None
    resample = str(config.provider_options.get("resample", "bilinear"))
    bbox = dem._parse_bbox(config.bbox)
    requested = config.requested_years
    options = dict(config.provider_options)

    product_assets: list[DemProductAsset] = []
    years_skipped: dict[int, str] = {}

    for product in config.products:
        datum = config.vertical_datum
        asset = DemProductAsset(
            product=product,
            coverage_id=f"skorowidz:{product}:{datum}",
            endpoint=dem_skorowidz_client.endpoint(product, datum, options),
        )
        try:
            year_to_typename = await dem_skorowidz_client.year_typenames(
                product, datum, options, timeout=config.timeout, retry_policy=retry_policy
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("skorowidz capabilities failed for %s/%s: %s", product, datum, exc)
            year_to_typename = {}
        available = [y for y in requested if y in year_to_typename]

        year_assets: list[DemYearAsset] = []
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            for year in available:
                ya = DemYearAsset(year=year)
                native = (
                    config.dem_root / "skorowidz" / f"{product}_{datum}" / "native" / f"year_{year}.tif"
                )
                try:
                    _status, tiles, _bb, _acq = await dem_skorowidz_client.tiles_for_year(
                        product, datum, year, config.bbox, config.srs,
                        year_to_typename=year_to_typename, options=options,
                        timeout=config.timeout, retry_policy=retry_policy,
                    )
                    if not tiles:
                        years_skipped[year] = "no tiles in AOI"
                        continue
                    ya.godla = sorted(tiles.keys())
                    if not (native.exists() and not config.overwrite):
                        paths = await _download_tiles(
                            list(tiles.values()), tmp_dir / f"{product}_{year}", config, retry_policy
                        )
                        _mosaic_asc_to_native(paths, native, bbox)
                        ya.tile_count = len(paths)
                    ya.native_path = str(native)
                    ya.native_width, ya.native_height = dem._raster_dims(native)
                    if grid is not None:
                        aligned = (
                            config.dem_root / "skorowidz" / f"{product}_{datum}" / "aligned" / f"year_{year}.tif"
                        )
                        target_bbox, gw, gh = grid
                        dem._align_to_grid(
                            native, aligned, target_bbox=target_bbox,
                            target_width=gw, target_height=gh, srs=config.srs, resample=resample,
                        )
                        ya.aligned_path = str(aligned)
                        ya.aligned_width, ya.aligned_height = gw, gh
                    ya.passed = True
                except Exception as exc:  # noqa: BLE001
                    ya.errors.append(str(exc))
                year_assets.append(ya)

        asset.years = year_assets
        asset.passed = any(y.passed for y in year_assets)
        product_assets.append(asset)

    passed = any(y.passed for a in product_assets for y in a.years)
    manifest = DemManifest(
        provider="geoportal", bbox=config.bbox, srs=config.srs, vertical_datum=config.vertical_datum,
        transport="skorowidz", years_requested=requested, years_skipped=years_skipped,
        products=product_assets, align_to_render=config.align_to_render, passed=passed,
        notes="GUGiK skorowidz (WFS) historical NMT/NMPT; one mosaic per ALS acquisition year.",
        run_parameters=config.model_dump(mode="json"),
    )
    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    config.output_json.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    logger.info(
        "DEM skorowidz: products=%s years_done=%s skipped=%s passed=%s",
        [a.product for a in product_assets],
        sum(1 for a in product_assets for y in a.years if y.passed),
        len(years_skipped), passed,
    )
    return (0 if passed else 1), config.output_json


def run(config: DemConfig) -> tuple[int, Path]:
    return asyncio.run(_run_async(config))
