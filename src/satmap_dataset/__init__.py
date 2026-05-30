"""satmap_dataset package."""

from satmap_dataset.models import (
    IndexManifest,
    LayerManifest,
    ReferenceGrid,
    ValidationReport,
    YearAvailabilityReport,
    YearStatus,
)

__all__ = [
    "IndexManifest",
    "LayerManifest",
    "ReferenceGrid",
    "ValidationReport",
    "YearAvailabilityReport",
    "YearStatus",
]

__version__ = "0.1.0"
