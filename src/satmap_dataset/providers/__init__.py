from __future__ import annotations

from satmap_dataset.providers.base import Provider

__all__ = ["Provider", "get_provider"]


def get_provider(name: str) -> Provider:
    if name == "geoportal":
        from satmap_dataset.providers.geoportal import GeoportalProvider

        return GeoportalProvider()
    if name == "lantmateriet":
        from satmap_dataset.providers.lantmateriet import LantmaterietProvider

        return LantmaterietProvider()
    if name == "sentinel2":
        from satmap_dataset.providers.sentinel2 import Sentinel2Provider

        return Sentinel2Provider()
    raise ValueError(
        f"Unknown provider: {name!r}. Expected 'geoportal', 'lantmateriet', or 'sentinel2'."
    )
