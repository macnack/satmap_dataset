from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from satmap_dataset.models import LayerManifest, ReferenceGrid


class Layer(ABC):
    """A single modality (RGB ortho, DEM, OSM labels) that emits aligned bands.

    Generalizes the older ``Provider`` concept: a Layer owns whatever
    acquisition + alignment sub-steps it needs and returns one ``LayerManifest``
    describing its bands, provenance, and (for grid-defining layers) the shared
    ``ReferenceGrid`` that the orchestrator passes to the other layers.
    """

    name: str = "abstract"
    role: str = "abstract"  # "rgb" | "dem" | "labels"
    defines_grid: bool = False
    default_native_srs: str = "EPSG:2180"

    @abstractmethod
    def bands(self, config: Any) -> list[str]:
        """Band/asset/category identifiers this layer will emit."""

    @abstractmethod
    def produce(
        self, config: Any, grid: ReferenceGrid | None
    ) -> tuple[int, LayerManifest]:
        """Acquire + align this layer's output.

        Grid-defining layers (``defines_grid=True``) ignore an incoming grid and
        compute their own, embedding it in ``manifest.grid``. Non-grid layers
        require a non-None grid supplied by the orchestrator.
        """
