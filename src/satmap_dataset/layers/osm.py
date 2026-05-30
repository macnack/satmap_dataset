from __future__ import annotations

from satmap_dataset.config import OsmConfig
from satmap_dataset.layers import register_layer
from satmap_dataset.layers.base import Layer
from satmap_dataset.models import LayerManifest, ReferenceGrid
from satmap_dataset.pipeline import osm as osm_pipeline


@register_layer("osm")
class OsmLayer(Layer):
    """OSM semantic-label layer. Rasterizes per-year category masks onto the grid.

    Thin wrapper over the existing ``pipeline.osm.run``; raster outputs and their
    paths are produced by that pipeline unchanged. The grid supplied by the
    orchestrator carries both the target dimensions and the per-year snapshot
    dates OSM needs (``grid.year_date_map``).
    """

    name = "osm"
    role = "labels"
    defines_grid = False
    provider_name = None

    def bands(self, config: OsmConfig) -> list[str]:
        return list(config.categories)

    def produce(
        self, config: OsmConfig, grid: ReferenceGrid | None
    ) -> tuple[int, LayerManifest]:
        if grid is not None:
            update = {
                "target_width": grid.width,
                "target_height": grid.height,
                # Prefer the injected grid over any stale render_manifest path.
                "render_manifest": None,
            }
            if grid.year_date_map:
                update["year_date_map"] = dict(grid.year_date_map)
            config = config.model_copy(update=update)
        code, output = osm_pipeline.run(config)
        manifest = LayerManifest.model_validate_json(
            output.read_text(encoding="utf-8")
        )
        return code, manifest
