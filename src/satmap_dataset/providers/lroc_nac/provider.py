"""LROC NAC (Moon) multi-temporal provider — ODE index + download.

Sources lunar NAC observations from the PDS Orbital Data Explorer REST API.
Index enumerates every overlapping NAC observation across a lat/lon bbox and
date range; download pulls the PDS frames. Map projection (ISIS cam2map) and
render are intentionally out of scope for this provider.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from satmap_dataset.config import DownloadConfig, IndexConfig
from satmap_dataset.models import (
    IndexManifest,
    TileAcquisitionMetadata,
    YearAvailabilityReport,
    YearStatus,
)
from satmap_dataset.pipeline.validator import evaluate_year_policy
from satmap_dataset.providers.base import Provider
from satmap_dataset.providers.lroc_nac import crs, ode

logger = logging.getLogger("satmap_dataset.lroc_nac")

DEFAULT_TARGET_SRS = "IAU_2015:30100"


class LrocNacProvider(Provider):
    name = "lroc_nac"
    default_target_srs = DEFAULT_TARGET_SRS

    def index(self, config: IndexConfig) -> tuple[int, Path]:
        return asyncio.run(self._index_async(config))

    async def _index_async(self, config: IndexConfig) -> tuple[int, Path]:
        options = dict(config.provider_options)
        search_options = ode.OdeSearchOptions(
            url=str(options.get("ode_url", ode.OdeSearchOptions.url)),
            product_type=str(options.get("product_type", "CDRNAC4")),
            loc=str(options.get("loc", "f")),
            limit=int(options.get("page_limit", 100)),
            max_pages=int(options.get("max_pages", 20)),
        )
        westlon, eastlon, minlat, maxlat = crs.normalize_bbox_to_ode(config.bbox, config.srs)
        min_obtime = str(options.get("min_obtime", f"{config.year_start}-01-01"))
        max_obtime = str(options.get("max_obtime", f"{config.year_end}-12-31"))

        warnings: list[str] = []
        errors: list[str] = []
        try:
            products = await ode.search_products(
                search_options,
                westlon=westlon, eastlon=eastlon, minlat=minlat, maxlat=maxlat,
                min_obtime=min_obtime, max_obtime=max_obtime,
                timeout=float(options.get("timeout", 60.0)),
            )
        except Exception as exc:  # noqa: BLE001 — surface transport errors in manifest
            products = []
            errors.append(f"ODE search failed: {exc}")

        max_incidence = options.get("max_incidence_angle")
        if max_incidence is not None:
            limit = float(max_incidence)
            products = [
                p for p in products
                if p.incidence_angle is None or p.incidence_angle <= limit
            ]

        grouped = ode.group_products_by_year(products)
        years_available = sorted(grouped.keys())

        tile_sources_by_year: dict[int, dict[str, str]] = {}
        tile_bboxes_by_year: dict[int, dict[str, list[float]]] = {}
        tile_acquisition_by_year: dict[int, dict[str, TileAcquisitionMetadata]] = {}
        year_statuses: list[YearStatus] = []
        years_excluded: dict[int, str] = {}

        for year in config.requested_years:
            items = grouped.get(year, [])
            if not items:
                year_statuses.append(
                    YearStatus(year=year, typename_exists=year in years_available,
                               feature_count=0, status="zero_features",
                               reason="no_nac_observation")
                )
                years_excluded[year] = "no_nac_observation"
                continue
            sources: dict[str, str] = {}
            bboxes: dict[str, list[float]] = {}
            acquisition: dict[str, TileAcquisitionMetadata] = {}
            for product in items:
                if product.file_url is None:
                    continue
                sources[product.pdsid] = product.file_url
                if product.footprint_bbox is not None:
                    bboxes[product.pdsid] = list(product.footprint_bbox)
                acquisition[product.pdsid] = TileAcquisitionMetadata(
                    acquisition_date=product.observation_time,
                    publication_date=None,
                    acquisition_year=year,
                )
            if not sources:
                year_statuses.append(
                    YearStatus(year=year, typename_exists=True, feature_count=0,
                               status="zero_features", reason="no_downloadable_asset")
                )
                years_excluded[year] = "no_downloadable_asset"
                continue
            tile_sources_by_year[year] = sources
            if bboxes:
                tile_bboxes_by_year[year] = bboxes
            tile_acquisition_by_year[year] = acquisition
            year_statuses.append(
                YearStatus(year=year, typename_exists=True,
                           feature_count=len(sources), status="has_features", reason=None)
            )

        years_included = sorted(tile_sources_by_year.keys())
        policy = evaluate_year_policy(
            requested_years=config.requested_years,
            available_years=years_included,
            strict_years=config.strict_years,
            min_years=config.min_years,
        )
        if not years_included and not errors:
            errors.append("ODE returned no NAC observations for the requested bbox/date range.")

        provider_metadata: dict[str, Any] = {
            "ode_url": search_options.url,
            "product_type": search_options.product_type,
            "loc": search_options.loc,
            "min_obtime": min_obtime,
            "max_obtime": max_obtime,
            "available_years": years_available,
            "observations_per_year": {y: len(grouped[y]) for y in years_available},
            "incidence_by_tile": {
                p.pdsid: p.incidence_angle for p in products if p.incidence_angle is not None
            },
            "map_resolution_by_tile": {
                p.pdsid: p.map_resolution for p in products if p.map_resolution is not None
            },
        }

        manifest = IndexManifest(
            provider="lroc_nac",
            year_start=config.year_start, year_end=config.year_end,
            bbox=config.bbox, srs=config.srs,
            strict_years=config.strict_years, min_years=config.min_years,
            wfs_bbox_axes_swapped=False,
            years_requested=config.requested_years,
            year_statuses=year_statuses,
            years_available_wfs=years_available,
            years_included=years_included,
            years_excluded_with_reason=years_excluded,
            common_tile_ids=[],
            tile_sources_by_year=tile_sources_by_year,
            tile_bboxes_by_year=tile_bboxes_by_year,
            tile_acquisition_by_year=tile_acquisition_by_year,
            passed=policy.passed and bool(years_included),
            errors=list(errors) + list(policy.errors),
            warnings=list(warnings) + list(policy.warnings),
            run_parameters=config.model_dump(mode="json"),
            provider_metadata=provider_metadata,
        )
        availability = YearAvailabilityReport(
            year_start=config.year_start, year_end=config.year_end,
            bbox=config.bbox, srs=config.srs, wfs_bbox_axes_swapped=False,
            years_requested=manifest.years_requested,
            year_statuses=manifest.year_statuses,
            years_available_wfs=manifest.years_available_wfs,
            years_included=manifest.years_included,
            years_excluded_with_reason=manifest.years_excluded_with_reason,
            strict_years=manifest.strict_years, min_years=manifest.min_years,
            passed=manifest.passed, errors=manifest.errors, warnings=manifest.warnings,
            run_parameters=manifest.run_parameters,
        )

        config.output_json.parent.mkdir(parents=True, exist_ok=True)
        config.year_availability_output_json.parent.mkdir(parents=True, exist_ok=True)
        config.year_availability_output_json.write_text(
            availability.model_dump_json(indent=2), encoding="utf-8"
        )
        config.output_json.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        logger.info(
            "LROC NAC index: years_included=%s available=%s passed=%s",
            years_included, years_available, manifest.passed,
        )
        return (0 if manifest.passed else 1), config.output_json

    def download(self, config: DownloadConfig) -> tuple[int, Path]:
        return asyncio.run(self._download_async(config))

    async def _download_async(self, config: DownloadConfig) -> tuple[int, Path]:
        raise NotImplementedError("Implemented in Task 5")
