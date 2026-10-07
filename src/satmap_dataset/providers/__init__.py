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
    if name == "nls":
        from satmap_dataset.providers.nls import NlsProvider

        return NlsProvider()
    if name == "landsd_hk":
        from satmap_dataset.providers.landsd_hk import LandsdHkProvider

        return LandsdHkProvider()
    raise ValueError(
        f"Unknown provider: {name!r}. Expected 'geoportal', 'lantmateriet', "
        "'sentinel2', 'lroc_nac', 'nls', or 'landsd_hk'."
    )
