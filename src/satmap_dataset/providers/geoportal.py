from __future__ import annotations

from pathlib import Path

from satmap_dataset.config import DemConfig, DownloadConfig, IndexConfig
from satmap_dataset.pipeline import dem, downloader, index_builder
from satmap_dataset.providers.base import Provider


class GeoportalProvider(Provider):
    name = "geoportal"
    default_target_srs = "EPSG:2180"

    def index(self, config: IndexConfig) -> tuple[int, Path]:
        return index_builder.run(config)

    def download(self, config: DownloadConfig) -> tuple[int, Path]:
        return downloader.run(config)

    def dem(self, config: DemConfig) -> tuple[int, Path]:
        return dem.run(config)
