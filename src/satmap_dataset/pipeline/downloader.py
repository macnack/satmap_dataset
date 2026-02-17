from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import random
from pathlib import Path
from urllib.parse import urlparse
import logging

import aiofiles
import httpx
import tifffile

from satmap_dataset.config import DownloadConfig
from satmap_dataset.models import DatasetManifest, IndexManifest

logger = logging.getLogger("satmap_dataset.download")
WMS_ORTHO_URL = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMS/StandardResolutionTime"


@dataclass(frozen=True)
class BBox:
    min_x: float
    min_y: float
    max_x: float
    max_y: float


@dataclass(frozen=True)
class WMSTileSpec:
    year: int
    source_type: str
    x0: int
    y0: int
    width: int
    height: int
    bbox: BBox
    output_path: Path


def _read_index_manifest(path: Path) -> IndexManifest:
    return IndexManifest.model_validate_json(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: DatasetManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")


def _parse_bbox(value: str) -> BBox:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must have 4 numeric values")
    xmin, ymin, xmax, ymax = parts
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("bbox must satisfy xmin<xmax and ymin<ymax")
    return BBox(min_x=xmin, min_y=ymin, max_x=xmax, max_y=ymax)


def _bbox_dimensions_meters(bbox: BBox) -> tuple[float, float]:
    return bbox.max_x - bbox.min_x, bbox.max_y - bbox.min_y


def _wms_time_for_year(year: int) -> str:
    cest = timezone(timedelta(hours=2))
    return datetime(year, 7, 15, 12, 0, 0, tzinfo=cest).isoformat()


def _geo_key_directory_for_epsg_2180() -> tuple[int, ...]:
    return (
        1,
        1,
        0,
        4,
        1024,
        0,
        1,
        1,
        1025,
        0,
        1,
        1,
        3072,
        0,
        1,
        2180,
        3076,
        0,
        1,
        9001,
    )


def _is_valid_cached_tiff(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 256:
        return False
    try:
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            if int(page.imagewidth) <= 0 or int(page.imagelength) <= 0:
                return False
            _ = page.asarray(maxworkers=1)
        return True
    except Exception:
        try:
            from PIL import Image
            import numpy as np

            with Image.open(path) as image:
                arr = np.asarray(image)
            return arr.size > 0
        except Exception:
            return False


def _is_valid_cached_asset(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        return _is_valid_cached_tiff(path)
    return False


def _build_wfs_jobs(index_manifest: IndexManifest) -> list[tuple[int, str, str]]:
    jobs: list[tuple[int, str, str]] = []
    for year in index_manifest.years_included:
        year_sources = index_manifest.tile_sources_by_year.get(year, {})
        for tile_id in sorted(year_sources.keys()):
            url = year_sources.get(tile_id)
            if not url:
                continue
            jobs.append((year, tile_id, url))
    return jobs


def _compute_output_dimensions(bbox: BBox, px_per_meter: float) -> tuple[int, int]:
    width_m, height_m = _bbox_dimensions_meters(bbox)
    width = max(1, int(round(width_m * px_per_meter)))
    height = max(1, int(round(height_m * px_per_meter)))
    return width, height


def _iter_wms_tiles(
    *,
    year: int,
    source_type: str,
    bbox: BBox,
    px_per_meter: float,
    out_dir: Path,
    max_tile_size_px: int,
) -> list[WMSTileSpec]:
    full_width, full_height = _compute_output_dimensions(bbox, px_per_meter)
    px_x = (bbox.max_x - bbox.min_x) / float(full_width)
    px_y = (bbox.max_y - bbox.min_y) / float(full_height)

    specs: list[WMSTileSpec] = []
    for y0 in range(0, full_height, max_tile_size_px):
        tile_h = min(max_tile_size_px, full_height - y0)
        y1 = y0 + tile_h
        tile_max_y = bbox.max_y - y0 * px_y
        tile_min_y = bbox.max_y - y1 * px_y

        for x0 in range(0, full_width, max_tile_size_px):
            tile_w = min(max_tile_size_px, full_width - x0)
            x1 = x0 + tile_w
            tile_min_x = bbox.min_x + x0 * px_x
            tile_max_x = bbox.min_x + x1 * px_x

            output = out_dir / (
                f"wms_{year}_x{x0}_y{y0}_w{tile_w}_h{tile_h}.tiff"
            )
            specs.append(
                WMSTileSpec(
                    year=year,
                    source_type=source_type,
                    x0=x0,
                    y0=y0,
                    width=tile_w,
                    height=tile_h,
                    bbox=BBox(
                        min_x=tile_min_x,
                        min_y=tile_min_y,
                        max_x=tile_max_x,
                        max_y=tile_max_y,
                    ),
                    output_path=output,
                )
            )

    return specs


def _tag_wms_tile_as_geotiff(path: Path, bbox: BBox, width: int, height: int, srs: str) -> None:
    if srs.upper() != "EPSG:2180":
        raise ValueError(f"Unsupported SRS for WMS tile geotagging: {srs}")

    arr = None
    try:
        with tifffile.TiffFile(path) as tif:
            arr = tif.pages[0].asarray()
    except Exception:
        # Geoportal WMS often serves LZW-compressed TIFF; fall back to Pillow decoder
        # when tifffile lacks imagecodecs support in the runtime.
        from PIL import Image
        import numpy as np

        with Image.open(path) as image:
            arr = np.asarray(image)

    if arr.ndim == 2:
        photometric = "minisblack"
    else:
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
            photometric = "minisblack"
        else:
            if arr.shape[2] > 3:
                arr = arr[:, :, :3]
            photometric = "rgb"

    pixel_size_x = (bbox.max_x - bbox.min_x) / float(width)
    pixel_size_y = (bbox.max_y - bbox.min_y) / float(height)
    scale = (float(pixel_size_x), float(pixel_size_y), 0.0)
    tie = (0.0, 0.0, 0.0, float(bbox.min_x), float(bbox.max_y), 0.0)
    geokey = _geo_key_directory_for_epsg_2180()

    temp = path.with_suffix(path.suffix + ".tmp")
    if temp.exists():
        temp.unlink()

    tifffile.imwrite(
        temp,
        arr,
        photometric=photometric,
        compression="deflate",
        metadata=None,
        extratags=[
            (33550, "d", 3, scale, False),
            (33922, "d", 6, tie, False),
            (34735, "H", len(geokey), geokey, False),
        ],
    )
    temp.replace(path)


async def _download_with_retry(
    client: httpx.AsyncClient,
    url: str,
    output_path: Path,
    *,
    retries: int,
    retry_delay: float,
    sleep_min: float,
    sleep_max: float,
) -> bool:
    attempts = retries + 1
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, attempts + 1):
        await asyncio.sleep(random.uniform(sleep_min, sleep_max))
        try:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                async with aiofiles.open(output_path, "wb") as file_handle:
                    async for chunk in response.aiter_bytes():
                        await file_handle.write(chunk)
            return True
        except (httpx.HTTPError, OSError):
            if attempt >= attempts:
                return False
            await asyncio.sleep(retry_delay * (2 ** (attempt - 1)))
    return False


async def _download_wms_tile_with_retry(
    client: httpx.AsyncClient,
    *,
    year: int,
    bbox: BBox,
    srs: str,
    width: int,
    height: int,
    output_path: Path,
    retries: int,
    retry_delay: float,
    sleep_min: float,
    sleep_max: float,
) -> bool:
    params = {
        "SERVICE": "WMS",
        "REQUEST": "GetMap",
        "VERSION": "1.1.1",
        "LAYERS": "Raster",
        "STYLES": "",
        "SRS": srs,
        "BBOX": f"{bbox.min_x:.6f},{bbox.min_y:.6f},{bbox.max_x:.6f},{bbox.max_y:.6f}",
        "WIDTH": str(width),
        "HEIGHT": str(height),
        "FORMAT": "image/tiff",
        "TRANSPARENT": "TRUE",
        "EXCEPTIONS": "application/vnd.ogc.se_xml",
        "TIME": _wms_time_for_year(year),
    }
    attempts = retries + 1
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, attempts + 1):
        await asyncio.sleep(random.uniform(sleep_min, sleep_max))
        try:
            response = await client.get(WMS_ORTHO_URL, params=params)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            if "image" not in content_type and "tiff" not in content_type:
                raise RuntimeError(f"Expected image response, got {content_type or 'unknown'}")
            async with aiofiles.open(output_path, "wb") as file_handle:
                await file_handle.write(response.content)
            return True
        except Exception:
            if attempt >= attempts:
                return False
            await asyncio.sleep(retry_delay * (2 ** (attempt - 1)))
    return False


def _resolve_year_sets(
    index_manifest: IndexManifest,
    mode: str,
    *,
    wms_fallback_missing_years: bool,
) -> tuple[list[int], list[int], list[int]]:
    requested = sorted(set(index_manifest.years_requested))
    wfs_years = sorted(set(index_manifest.years_included))

    if mode == "wfs_render":
        return requested, wfs_years, []
    if mode == "wms_tiled":
        return requested, [], requested

    if not wms_fallback_missing_years:
        return requested, wfs_years, []
    fallback_years = sorted(set(requested) - set(wfs_years))
    return requested, wfs_years, fallback_years


def run(config: DownloadConfig) -> tuple[int, Path]:
    logger.info(
        "Download: start mode=%s index_manifest=%s root=%s concurrency=%s",
        config.mode,
        config.index_manifest,
        config.download_root,
        config.concurrency,
    )
    index_manifest = _read_index_manifest(config.index_manifest)
    requested_years, wfs_years, wms_years = _resolve_year_sets(
        index_manifest,
        config.mode,
        wms_fallback_missing_years=config.wms_fallback_missing_years,
    )

    if config.mode in {"wms_tiled", "hybrid"} and config.bbox is None:
        raise ValueError("bbox is required when mode uses WMS")

    wfs_jobs = _build_wfs_jobs(index_manifest) if wfs_years else []
    wms_specs: list[WMSTileSpec] = []
    if wms_years:
        bbox_obj = _parse_bbox(config.bbox or "")
        for year in wms_years:
            source_type = "wms" if config.mode == "wms_tiled" else "wms_fallback"
            wms_specs.extend(
                _iter_wms_tiles(
                    year=year,
                    source_type=source_type,
                    bbox=bbox_obj,
                    px_per_meter=config.px_per_meter,
                    out_dir=config.download_root / str(year),
                    max_tile_size_px=2048,
                )
            )

    logger.info(
        "Download: planned wfs_jobs=%s wms_tiles=%s years_requested=%s",
        len(wfs_jobs),
        len(wms_specs),
        len(requested_years),
    )

    results: list[tuple[Path, bool]] = []
    years_source_map: dict[int, str] = {}

    async def _run_wfs_jobs() -> list[tuple[Path, bool]]:
        if not wfs_jobs:
            return []

        queue: asyncio.Queue[tuple[int, str, str] | None] = asyncio.Queue()
        for job in wfs_jobs:
            queue.put_nowait(job)

        limits = httpx.Limits(max_connections=config.concurrency, max_keepalive_connections=config.concurrency)
        timeout = httpx.Timeout(timeout=config.timeout, connect=min(config.timeout, 20.0))

        output: list[tuple[Path, bool]] = []
        lock = asyncio.Lock()

        async def worker() -> None:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=timeout,
                limits=limits,
                headers={"User-Agent": "satmap_dataset/0.1"},
            ) as client:
                while True:
                    item = await queue.get()
                    if item is None:
                        queue.task_done()
                        return

                    year, tile_id, url = item
                    filename = Path(urlparse(url).path).name or f"{tile_id}.tif"
                    output_path = config.download_root / str(year) / filename

                    ok = False
                    if output_path.exists() and not config.overwrite:
                        ok = _is_valid_cached_asset(output_path)
                        if not ok:
                            try:
                                output_path.unlink()
                            except OSError:
                                pass

                    if not ok:
                        ok = await _download_with_retry(
                            client,
                            url,
                            output_path,
                            retries=config.retries,
                            retry_delay=config.retry_delay,
                            sleep_min=config.sleep_min,
                            sleep_max=config.sleep_max,
                        )
                        if ok:
                            ok = _is_valid_cached_asset(output_path)

                    async with lock:
                        output.append((output_path, ok))
                    queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(config.concurrency)]
        await queue.join()
        for _ in workers:
            queue.put_nowait(None)
        await asyncio.gather(*workers)
        return output

    async def _run_wms_jobs() -> list[tuple[Path, bool]]:
        if not wms_specs:
            return []

        queue: asyncio.Queue[WMSTileSpec | None] = asyncio.Queue()
        for spec in wms_specs:
            queue.put_nowait(spec)

        timeout = httpx.Timeout(timeout=config.timeout, connect=min(config.timeout, 20.0))
        output: list[tuple[Path, bool]] = []
        lock = asyncio.Lock()
        progress = {"done": 0}

        async def worker() -> None:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=timeout,
                headers={"User-Agent": "satmap_dataset/0.1"},
            ) as client:
                while True:
                    item = await queue.get()
                    if item is None:
                        queue.task_done()
                        return

                    spec = item
                    ok = False
                    if spec.output_path.exists() and not config.overwrite:
                        ok = _is_valid_cached_asset(spec.output_path)
                        if not ok:
                            try:
                                spec.output_path.unlink()
                            except OSError:
                                pass

                    if not ok:
                        ok = await _download_wms_tile_with_retry(
                            client,
                            year=spec.year,
                            bbox=spec.bbox,
                            srs=config.srs,
                            width=spec.width,
                            height=spec.height,
                            output_path=spec.output_path,
                            retries=config.retries,
                            retry_delay=config.retry_delay,
                            sleep_min=config.sleep_min,
                            sleep_max=config.sleep_max,
                        )
                        if ok:
                            try:
                                _tag_wms_tile_as_geotiff(
                                    spec.output_path,
                                    bbox=spec.bbox,
                                    width=spec.width,
                                    height=spec.height,
                                    srs=config.srs,
                                )
                                ok = _is_valid_cached_asset(spec.output_path)
                            except Exception:
                                ok = False

                    async with lock:
                        output.append((spec.output_path, ok))
                        progress["done"] += 1
                        if progress["done"] % 25 == 0 or not ok:
                            logger.info(
                                "Download: WMS progress %s/%s (latest ok=%s file=%s)",
                                progress["done"],
                                len(wms_specs),
                                ok,
                                spec.output_path.name,
                            )
                    queue.task_done()

        worker_count = max(1, min(config.concurrency, 16))
        workers = [asyncio.create_task(worker()) for _ in range(worker_count)]
        await queue.join()
        for _ in workers:
            queue.put_nowait(None)
        await asyncio.gather(*workers)
        return output

    results.extend(asyncio.run(_run_wfs_jobs()))
    results.extend(asyncio.run(_run_wms_jobs()))

    failed = [path for path, ok in results if not ok]
    assets = sorted(str(path) for path, ok in results if ok)

    for year in wfs_years:
        if any(Path(asset).parent.name == str(year) for asset in assets):
            years_source_map[year] = "wfs"
    for year in wms_years:
        if any(Path(asset).parent.name == str(year) for asset in assets):
            years_source_map[year] = "wms" if config.mode == "wms_tiled" else "wms_fallback"

    years_included_effective = sorted(years_source_map.keys())
    if config.mode == "wms_tiled":
        expected_years = requested_years
    elif config.mode == "wfs_render":
        expected_years = wfs_years
    else:
        expected_years = sorted(set(wfs_years) | set(wms_years))

    missing_year_outputs = [
        year for year in expected_years if year not in years_included_effective
    ]

    index_gate = index_manifest.passed or config.mode == "wms_tiled"
    passed = (
        index_gate
        and not failed
        and not missing_year_outputs
        and bool(assets)
    )

    manifest = DatasetManifest(
        stage="download",
        years_requested=index_manifest.years_requested,
        years_available_wfs=index_manifest.years_available_wfs,
        years_included=years_included_effective,
        years_excluded_with_reason=index_manifest.years_excluded_with_reason,
        common_tile_ids=index_manifest.common_tile_ids,
        tile_sources_by_year=index_manifest.tile_sources_by_year,
        assets=assets,
        source_manifest=str(config.index_manifest),
        mode=config.mode,
        target_bbox=config.bbox if config.mode in {"wms_tiled", "hybrid"} else None,
        target_srs=config.srs if config.mode in {"wms_tiled", "hybrid"} else None,
        profile=config.profile,
        px_per_meter=config.px_per_meter if config.mode in {"wms_tiled", "hybrid"} else None,
        years_source_map=years_source_map,
        passed=passed,
        notes=(
            f"mode={config.mode} downloaded={len(assets)} failed={len(failed)} "
            f"missing_year_outputs={missing_year_outputs}"
        ),
    )
    _write_json(config.output_json, manifest)

    logger.info(
        "Download: finished passed=%s assets=%s failed=%s output=%s",
        manifest.passed,
        len(assets),
        len(failed),
        config.output_json,
    )
    return (0 if manifest.passed else 1), config.output_json
