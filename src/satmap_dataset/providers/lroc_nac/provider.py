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

from satmap_dataset.config import DownloadConfig, IndexConfig
from satmap_dataset.providers.base import Provider

logger = logging.getLogger("satmap_dataset.lroc_nac")

DEFAULT_TARGET_SRS = "IAU_2015:30100"


class LrocNacProvider(Provider):
    name = "lroc_nac"
    default_target_srs = DEFAULT_TARGET_SRS

    def index(self, config: IndexConfig) -> tuple[int, Path]:
        return asyncio.run(self._index_async(config))

    async def _index_async(self, config: IndexConfig) -> tuple[int, Path]:
        raise NotImplementedError("Implemented in Task 4")

    def download(self, config: DownloadConfig) -> tuple[int, Path]:
        return asyncio.run(self._download_async(config))

    async def _download_async(self, config: DownloadConfig) -> tuple[int, Path]:
        raise NotImplementedError("Implemented in Task 5")
