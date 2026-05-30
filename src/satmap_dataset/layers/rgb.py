from __future__ import annotations

from pathlib import Path

from satmap_dataset.config import ALLOWED_PROVIDERS, RunConfig
from satmap_dataset.layers.base import Layer
from satmap_dataset.models import (
    LayerManifest,
    LayerYearAsset,
    ReferenceGrid,
)
from satmap_dataset.pipeline.run_all import _run_rgb_pipeline

_RGB_BANDS = ["red", "green", "blue"]


class RgbLayer(Layer):
    """RGB orthophoto layer. Defines the shared ReferenceGrid for a location.

    Wraps the existing index -> download -> render pipeline (via
    ``run_all._run_rgb_pipeline``) and maps its render manifest onto the unified
    ``LayerManifest``. The render stage owns the actual raster files; this
    adapter only assembles provenance.
    """

    role = "rgb"
    defines_grid = True

    def __init__(self, provider_name: str) -> None:
        if provider_name not in ALLOWED_PROVIDERS:
            raise ValueError(
                f"Unknown RGB provider {provider_name!r}; expected one of "
                f"{sorted(ALLOWED_PROVIDERS)}."
            )
        self.provider_name = provider_name
        self.name = f"{provider_name}_rgb"
        self.default_native_srs = "EPSG:2180"

    def bands(self, config: RunConfig) -> list[str]:
        return list(_RGB_BANDS)

    def produce(
        self, config: RunConfig, grid: ReferenceGrid | None = None
    ) -> tuple[int, LayerManifest]:
        code, render_output = _run_rgb_pipeline(config)
        if code != 0:
            return code, LayerManifest(
                layer=self.name,
                role="rgb",
                provider=self.provider_name,
                passed=False,
                source_manifest=str(render_output),
                run_parameters=config.model_dump(mode="json"),
            )
        manifest = self._to_layer_manifest(render_output, config)
        return code, manifest

    def _to_layer_manifest(
        self, render_output: Path, config: RunConfig
    ) -> LayerManifest:
        dm = LayerManifest.model_validate_json(
            render_output.read_text(encoding="utf-8")
        )
        ref_grid = dm.grid or ReferenceGrid.from_render_manifest(dm)

        years: list[LayerYearAsset] = []
        for year in dm.years_included:
            rgb_path = _asset_for_year(dm.assets, year)
            years.append(
                LayerYearAsset(
                    year=year,
                    snapshot_date=ref_grid.year_date_map.get(year),
                    source=dm.years_source_map.get(year),
                    assets={"rgb": rgb_path} if rgb_path else {},
                    color_qc=dict(dm.color_qc_by_year.get(year, {})),
                    acquisition=dict(dm.tile_acquisition_by_year.get(year, {})),
                    passed=True,
                )
            )

        return LayerManifest(
            layer=self.name,
            role="rgb",
            stage="render",
            provider=dm.provider or self.provider_name,
            grid=ref_grid,
            bands=list(_RGB_BANDS),
            years_requested=dm.years_requested,
            years_included=dm.years_included,
            years_excluded_with_reason=dm.years_excluded_with_reason,
            years_source_map=dm.years_source_map,
            years=years,
            assets=list(dm.assets),
            source_manifest=str(render_output),
            pixel_profile=dm.pixel_profile,
            target_srs=ref_grid.srs,
            render_cache_signature=dm.render_cache_signature,
            diagnostics_report_path=dm.diagnostics_report_path,
            diagnostics_quicklook_dir=dm.diagnostics_quicklook_dir,
            passed=dm.passed,
            notes=dm.notes,
            run_parameters=dm.run_parameters,
            provider_metadata={
                **dm.provider_metadata,
                "common_tile_ids": dm.common_tile_ids,
                "years_available_wfs": dm.years_available_wfs,
                "mode": dm.mode,
                "forced_wms_years": dm.forced_wms_years,
                "resample_method": dm.resample_method,
                "render_backend": dm.render_backend,
            },
        )


def _asset_for_year(assets: list[str], year: int) -> str | None:
    token = f"year_{year}"
    for asset in assets:
        if token in Path(asset).name:
            return asset
    return None
