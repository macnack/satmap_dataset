"""satmap_dataset package."""

from satmap_dataset.models import (
    DatasetManifest,
    IndexManifest,
    ValidationReport,
    YearAvailabilityReport,
    YearStatus,
)

__all__ = [
    "DatasetManifest",
    "IndexManifest",
    "ValidationReport",
    "YearAvailabilityReport",
    "YearStatus",
]

__version__ = "0.1.0"
