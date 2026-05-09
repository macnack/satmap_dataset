from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from satmap_dataset.config import DownloadConfig, IndexConfig


class Provider(ABC):
    """Source of orthophoto data. Each stage returns (exit_code, artifact_path)."""

    name: str = "abstract"
    default_target_srs: str = "EPSG:2180"

    @abstractmethod
    def index(self, config: IndexConfig) -> tuple[int, Path]:
        ...

    @abstractmethod
    def download(self, config: DownloadConfig) -> tuple[int, Path]:
        ...
