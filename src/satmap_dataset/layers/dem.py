from __future__ import annotations

from satmap_dataset.config import DemConfig
from satmap_dataset.layers import register_layer
from satmap_dataset.layers.base import Layer
from satmap_dataset.models import LayerManifest, ReferenceGrid
from satmap_dataset.pipeline import dem as dem_pipeline


@register_layer("dem")
class DemLayer(Layer):
    """Elevation (NMT/NMPT) layer. Aligns to a grid supplied by the orchestrator.

    Thin wrapper over the existing ``pipeline.dem.run`` (WCS + skorowidz
    transports); raster outputs and their paths are produced by that pipeline
    unchanged.
    """

    name = "dem"
    role = "dem"
    defines_grid = False

    def bands(self, config: DemConfig) -> list[str]:
        return list(config.products)

    def produce(
        self, config: DemConfig, grid: ReferenceGrid | None
    ) -> tuple[int, LayerManifest]:
        if grid is not None:
            if grid.srs.upper() != config.srs.upper():
                raise ValueError(
                    f"DEM layer cannot align to a grid in a different CRS: grid.srs="
                    f"{grid.srs!r} vs config.srs={config.srs!r}. The alignment step does "
                    "not reproject across CRS; configure the DEM in the grid's CRS."
                )
            config = config.model_copy(
                update={
                    "align_to_render": True,
                    "target_bbox": grid.bbox,
                    "target_width": grid.width,
                    "target_height": grid.height,
                    # Prefer the injected grid over any stale render_manifest path.
                    "render_manifest": None,
                }
            )
        code, output = dem_pipeline.run(config)
        manifest = LayerManifest.model_validate_json(
            output.read_text(encoding="utf-8")
        )
        return code, manifest
