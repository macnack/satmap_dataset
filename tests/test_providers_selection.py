from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.providers import get_provider
from satmap_dataset.providers.base import Provider
from satmap_dataset.providers.geoportal import GeoportalProvider
from satmap_dataset.providers.lantmateriet import LantmaterietProvider


def test_get_provider_returns_geoportal() -> None:
    provider = get_provider("geoportal")
    assert isinstance(provider, GeoportalProvider)
    assert isinstance(provider, Provider)
    assert provider.name == "geoportal"
    assert provider.default_target_srs == "EPSG:2180"


def test_get_provider_returns_lantmateriet() -> None:
    provider = get_provider("lantmateriet")
    assert isinstance(provider, LantmaterietProvider)
    assert isinstance(provider, Provider)
    assert provider.name == "lantmateriet"
    assert provider.default_target_srs == "EPSG:3006"


def test_get_provider_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("does-not-exist")


def test_get_provider_returns_lroc_nac() -> None:
    from satmap_dataset.providers.lroc_nac import LrocNacProvider

    provider = get_provider("lroc_nac")
    assert isinstance(provider, LrocNacProvider)
    assert isinstance(provider, Provider)
    assert provider.name == "lroc_nac"
    assert provider.default_target_srs == "IAU_2015:30100"


def test_get_provider_returns_nls() -> None:
    from satmap_dataset.providers.nls import NlsProvider

    provider = get_provider("nls")
    assert isinstance(provider, NlsProvider)
    assert isinstance(provider, Provider)
    assert provider.name == "nls"
    assert provider.default_target_srs == "EPSG:3067"
