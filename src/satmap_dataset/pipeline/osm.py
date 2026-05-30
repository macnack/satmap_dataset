from __future__ import annotations

import asyncio
import logging
import random
from pathlib import Path

from satmap_dataset.config import OsmConfig
from satmap_dataset.models import OsmCategoryAsset, OsmManifest, OsmYearAsset
from satmap_dataset.osm import overpass_client, rasterize
from satmap_dataset.osm.ohsome_client import bbox_epsg2180_to_wgs84

logger = logging.getLogger("satmap_dataset.osm")


def _read_year_date_map(config: OsmConfig) -> dict[int, str]:
    if config.year_date_map is not None:
        return dict(config.year_date_map)
    if config.render_manifest is not None and Path(config.render_manifest).exists():
        from satmap_dataset.models import DatasetManifest

        manifest = DatasetManifest.model_validate_json(
            Path(config.render_manifest).read_text(encoding="utf-8")
        )
        result: dict[int, str] = {}
        for year, tile_acq in manifest.tile_acquisition_by_year.items():
            dates = [v.acquisition_date for v in tile_acq.values() if v.acquisition_date is not None]
            if dates:
                result[int(year)] = max(dates)
        return result
    raise ValueError(
        "OsmConfig requires either render_manifest (with tile_acquisition_by_year) "
        "or year_date_map to determine per-year snapshot dates."
    )


def _read_grid(config: OsmConfig) -> tuple[int, int]:
    if config.target_width is not None and config.target_height is not None:
        return config.target_width, config.target_height
    if config.render_manifest is not None and Path(config.render_manifest).exists():
        from satmap_dataset.models import DatasetManifest

        manifest = DatasetManifest.model_validate_json(
            Path(config.render_manifest).read_text(encoding="utf-8")
        )
        if manifest.target_width is not None and manifest.target_height is not None:
            return int(manifest.target_width), int(manifest.target_height)
    xmin, ymin, xmax, ymax = (float(x) for x in config.bbox.split(","))
    return max(1, round(xmax - xmin)), max(1, round(ymax - ymin))


async def _run_async(config: OsmConfig) -> tuple[int, Path]:
    from satmap_dataset.geoportal.http import RetryPolicy

    errors: list[str] = []
    year_assets: list[OsmYearAsset] = []

    try:
        year_date_map = _read_year_date_map(config)
    except ValueError as exc:
        errors.append(str(exc))
        manifest = OsmManifest(
            bbox=config.bbox,
            bbox_wgs84="",
            srs=config.srs,
            categories=list(config.categories),
            passed=False,
            errors=errors,
            run_parameters=config.model_dump(mode="json"),
        )
        config.output_json.parent.mkdir(parents=True, exist_ok=True)
        config.output_json.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return 1, config.output_json

    target_width, target_height = _read_grid(config)
    target_bbox = tuple(float(x) for x in config.bbox.split(","))
    bbox_wgs84 = bbox_epsg2180_to_wgs84(config.bbox)
    retry_policy = RetryPolicy(
        max_attempts=config.retries, backoff_seconds=config.retry_delay
    )
    overpass_url = config.ohsome_url or overpass_client._DEFAULT_OVERPASS_URL

    for year in sorted(year_date_map.keys()):
        snapshot_date = year_date_map[year]
        normalized = snapshot_date if snapshot_date.endswith("Z") else snapshot_date + "T00:00:00Z"
        year_asset = OsmYearAsset(year=year, snapshot_date=normalized)

        for category in config.categories:
            raster_path = config.osm_root / f"year_{year}_{category}.tif"

            if raster_path.exists() and not config.overwrite:
                year_asset.categories[category] = OsmCategoryAsset(
                    feature_count=0, raster_path=str(raster_path)
                )
                continue

            if config.sleep_max > 0:
                await asyncio.sleep(random.uniform(config.sleep_min, config.sleep_max))

            try:
                geojson = await overpass_client.get_elements_geometry(
                    bbox_wgs84,
                    category,
                    normalized,
                    overpass_url=overpass_url,
                    timeout=config.timeout,
                    retry_policy=retry_policy,
                )
                features = geojson.get("features") or []
                count = len(features)

                if count == 0:
                    year_asset.categories[category] = OsmCategoryAsset(
                        feature_count=0, raster_path=None
                    )
                else:
                    rasterize.rasterize_geojson_to_file(
                        geojson,
                        raster_path,
                        target_bbox=target_bbox,
                        target_width=target_width,
                        target_height=target_height,
                        target_srs=config.srs,
                    )
                    year_asset.categories[category] = OsmCategoryAsset(
                        feature_count=count, raster_path=str(raster_path)
                    )
            except Exception as exc:  # noqa: BLE001
                year_asset.errors.append(f"{category}: {exc}")
                errors.append(f"year={year} category={category}: {exc}")

        year_asset.passed = not year_asset.errors
        year_assets.append(year_asset)

    passed = bool(year_assets) and all(a.passed for a in year_assets)
    manifest = OsmManifest(
        bbox=config.bbox,
        bbox_wgs84=bbox_wgs84,
        srs=config.srs,
        target_width=target_width,
        target_height=target_height,
        categories=list(config.categories),
        years=year_assets,
        passed=passed,
        errors=errors,
        run_parameters=config.model_dump(mode="json"),
    )
    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    config.output_json.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    logger.info(
        "OSM run: years=%s passed=%s errors=%s",
        [a.year for a in year_assets], passed, len(errors),
    )
    return (0 if passed else 1), config.output_json


def run(config: OsmConfig) -> tuple[int, Path]:
    return asyncio.run(_run_async(config))
