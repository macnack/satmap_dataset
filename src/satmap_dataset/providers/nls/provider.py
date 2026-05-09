from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import aiofiles
import httpx

from satmap_dataset.config import DownloadConfig, IndexConfig
from satmap_dataset.models import (
    DatasetManifest,
    IndexManifest,
    YearAvailabilityReport,
    YearStatus,
)
from satmap_dataset.pipeline.validator import evaluate_year_policy
from satmap_dataset.providers.nls.auth import basic_auth_header, resolve_api_key
from satmap_dataset.providers.nls.wcs import (
    DEFAULT_COVERAGE_ID,
    DEFAULT_WCS_URL,
    build_describe_coverage_url,
    build_get_coverage_url,
    parse_describe_coverage_years,
)


logger = logging.getLogger("satmap_dataset.nls")

WCS_AOI_CAP_METERS = 2000.0


def _option(options: dict[str, Any], key: str, default: Any) -> Any:
    if key in options and options[key] not in (None, ""):
        return options[key]
    return default


def _parse_bbox(bbox: str) -> tuple[float, float, float, float]:
    parts = [float(p.strip()) for p in bbox.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must have format xmin,ymin,xmax,ymax")
    xmin, ymin, xmax, ymax = parts
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("bbox must satisfy xmin<xmax and ymin<ymax")
    return (xmin, ymin, xmax, ymax)


def _check_aoi_cap(bbox: tuple[float, float, float, float]) -> str | None:
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin
    if width > WCS_AOI_CAP_METERS or height > WCS_AOI_CAP_METERS:
        return (
            f"bbox {width:.0f}m x {height:.0f}m exceeds NLS WCS cap of "
            f"{WCS_AOI_CAP_METERS:.0f}m on either side"
        )
    return None


def _fetch_describe_coverage_xml(
    *,
    base_url: str,
    coverage_id: str,
    api_key: str,
    timeout: float = 60.0,
) -> bytes:
    url = build_describe_coverage_url(base_url, coverage_id=coverage_id)
    headers = {"Authorization": basic_auth_header(api_key), "User-Agent": "satmap_dataset/0.1"}
    with httpx.Client(timeout=timeout, headers=headers) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.content


def _make_async_client(**kwargs: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(**kwargs)


async def _download_one(
    client: httpx.AsyncClient,
    url: str,
    output_path: Path,
    *,
    retries: int,
) -> bool:
    attempts = max(1, retries + 1)
    for attempt in range(1, attempts + 1):
        try:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                async with aiofiles.open(output_path, "wb") as fp:
                    async for chunk in response.aiter_bytes():
                        await fp.write(chunk)
            return True
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (400, 401, 403, 404):
                logger.error("NLS download terminal status=%s url=%s", exc.response.status_code, url)
                return False
            if attempt >= attempts:
                return False
            await asyncio.sleep(0.5 * attempt)
        except httpx.HTTPError as exc:
            if attempt >= attempts:
                logger.error("NLS download exhausted retries url=%s err=%s", url, exc)
                return False
            await asyncio.sleep(0.5 * attempt)
    return False


def _write_failed_manifest(config: IndexConfig, error: str) -> None:
    manifest = IndexManifest(
        provider="nls",
        year_start=config.year_start,
        year_end=config.year_end,
        bbox=config.bbox,
        srs=config.srs,
        strict_years=config.strict_years,
        min_years=config.min_years,
        years_requested=config.requested_years,
        year_statuses=[],
        years_available_wfs=[],
        years_included=[],
        years_excluded_with_reason={year: error for year in config.requested_years},
        passed=False,
        errors=[error],
        run_parameters=config.model_dump(mode="json"),
    )
    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    config.output_json.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


class NlsProvider:
    name = "nls"
    default_target_srs = "EPSG:3067"

    def index(self, config: IndexConfig) -> tuple[int, Path]:
        bbox = _parse_bbox(config.bbox)
        cap_error = _check_aoi_cap(bbox)
        if cap_error is not None:
            logger.error("NLS index: %s", cap_error)
            _write_failed_manifest(config, cap_error)
            return 2, config.output_json

        options = dict(config.provider_options)
        base_url = str(_option(options, "wcs_url", DEFAULT_WCS_URL))
        coverage_id = str(_option(options, "coverage_id", DEFAULT_COVERAGE_ID))
        api_key = resolve_api_key(options, secret_path=Path(".secret"))

        try:
            xml_bytes = _fetch_describe_coverage_xml(
                base_url=base_url,
                coverage_id=coverage_id,
                api_key=api_key,
            )
        except httpx.HTTPError as exc:
            error = f"DescribeCoverage failed: {exc}"
            _write_failed_manifest(config, error)
            return 1, config.output_json

        available_years = parse_describe_coverage_years(xml_bytes)
        available_set = set(available_years)
        requested_years = config.requested_years
        years_included = [y for y in requested_years if y in available_set]
        years_excluded = {
            y: "year_not_in_wcs_describe_coverage"
            for y in requested_years
            if y not in available_set
        }
        year_statuses = [
            YearStatus(
                year=y,
                typename_exists=(y in available_set),
                feature_count=1 if y in available_set else 0,
                status="has_features" if y in available_set else "no_typename",
                reason=None if y in available_set else "year_not_in_wcs_describe_coverage",
            )
            for y in requested_years
        ]

        tile_sources_by_year: dict[int, dict[str, str]] = {}
        tile_bboxes_by_year: dict[int, dict[str, list[float]]] = {}
        for year in years_included:
            url = build_get_coverage_url(
                base_url,
                coverage_id=coverage_id,
                bbox=bbox,
                year=year,
            )
            tile_sources_by_year[year] = {f"nls_{year}": url}
            tile_bboxes_by_year[year] = {f"nls_{year}": list(bbox)}

        policy = evaluate_year_policy(
            requested_years=requested_years,
            available_years=years_included,
            strict_years=config.strict_years,
            min_years=config.min_years,
        )
        errors = list(policy.errors)
        warnings = list(policy.warnings)
        if not years_included:
            errors.append("WCS DescribeCoverage returned no years intersecting the requested range.")

        provider_metadata: dict[str, Any] = {
            "wcs_url": base_url,
            "coverage_id": coverage_id,
            "available_years_in_coverage": available_years,
            "native_srs": "EPSG:3067",
            "aoi_cap_meters": WCS_AOI_CAP_METERS,
        }

        manifest = IndexManifest(
            provider="nls",
            year_start=config.year_start,
            year_end=config.year_end,
            bbox=config.bbox,
            srs=config.srs,
            strict_years=config.strict_years,
            min_years=config.min_years,
            years_requested=requested_years,
            year_statuses=year_statuses,
            years_available_wfs=available_years,
            years_included=years_included,
            years_excluded_with_reason=years_excluded,
            common_tile_ids=[f"nls_{y}" for y in years_included],
            tile_sources_by_year=tile_sources_by_year,
            tile_bboxes_by_year=tile_bboxes_by_year,
            passed=policy.passed and bool(years_included),
            errors=errors,
            warnings=warnings,
            run_parameters=config.model_dump(mode="json"),
            provider_metadata=provider_metadata,
        )

        availability = YearAvailabilityReport(
            year_start=config.year_start,
            year_end=config.year_end,
            bbox=config.bbox,
            srs=config.srs,
            years_requested=requested_years,
            year_statuses=year_statuses,
            years_available_wfs=available_years,
            years_included=years_included,
            years_excluded_with_reason=years_excluded,
            strict_years=config.strict_years,
            min_years=config.min_years,
            passed=manifest.passed,
            errors=errors,
            warnings=warnings,
            run_parameters=config.model_dump(mode="json"),
        )

        config.output_json.parent.mkdir(parents=True, exist_ok=True)
        config.year_availability_output_json.parent.mkdir(parents=True, exist_ok=True)
        config.output_json.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        config.year_availability_output_json.write_text(
            availability.model_dump_json(indent=2), encoding="utf-8"
        )
        logger.info(
            "NLS index: years_included=%s available=%s passed=%s",
            len(years_included),
            len(available_years),
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
        api_key = resolve_api_key(options, secret_path=Path(".secret"))
        headers = {
            "Authorization": basic_auth_header(api_key),
            "User-Agent": "satmap_dataset/0.1",
        }

        timeout = httpx.Timeout(timeout=config.timeout, connect=min(config.timeout, 20.0))
        limits = httpx.Limits(
            max_connections=max(1, config.concurrency),
            max_keepalive_connections=max(1, config.concurrency),
        )

        assets: list[str] = []
        failed: list[str] = []
        years_source_map: dict[int, str] = {}
        years_included_effective: list[int] = []

        async with _make_async_client(
            timeout=timeout, limits=limits, headers=headers, follow_redirects=True
        ) as client:
            for year in index_manifest.years_included:
                sources = index_manifest.tile_sources_by_year.get(year, {})
                if not sources:
                    failed.append(f"year_{year}_no_source")
                    continue
                tile_id, url = next(iter(sources.items()))
                output_path = config.download_root / str(year) / f"{tile_id}.tif"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                ok = output_path.exists() and output_path.stat().st_size > 0 and not config.overwrite
                if not ok:
                    ok = await _download_one(client, url, output_path, retries=config.retries)
                if ok:
                    assets.append(str(output_path))
                    years_source_map[year] = "wcs"
                    years_included_effective.append(year)
                else:
                    failed.append(url)

        manifest = DatasetManifest(
            provider="nls",
            stage="download",
            mode="wcs",
            years_requested=index_manifest.years_requested,
            years_available_wfs=index_manifest.years_available_wfs,
            years_included=sorted(years_included_effective),
            years_excluded_with_reason=index_manifest.years_excluded_with_reason,
            common_tile_ids=index_manifest.common_tile_ids,
            tile_sources_by_year=index_manifest.tile_sources_by_year,
            tile_bboxes_by_year=index_manifest.tile_bboxes_by_year,
            assets=sorted(set(assets)),
            source_manifest=str(config.index_manifest),
            target_bbox=config.bbox,
            target_srs=config.srs,
            profile=config.profile,
            px_per_meter=config.px_per_meter,
            years_source_map=years_source_map,
            forced_wms_years=[],
            passed=bool(assets) and not failed,
            notes=f"provider=nls downloaded={len(assets)} failed={len(failed)}",
            run_parameters=config.model_dump(mode="json"),
            provider_metadata={"failed_urls": failed},
        )
        config.output_json.parent.mkdir(parents=True, exist_ok=True)
        config.output_json.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return (0 if manifest.passed else 1), config.output_json
