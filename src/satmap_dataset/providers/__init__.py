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
    if name == "lroc_nac":
        from satmap_dataset.providers.lroc_nac import LrocNacProvider

        return LrocNacProvider()
    raise ValueError(
        f"Unknown provider: {name!r}. Expected 'geoportal', 'lantmateriet', "
        "'sentinel2', or 'lroc_nac'."
    )
