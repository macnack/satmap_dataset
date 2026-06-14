from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest
from pydantic import ValidationError

from satmap_dataset.config import DownloadConfig, IndexConfig


def test_index_config_default_provider_geoportal():
    cfg = IndexConfig(year_start=2020, year_end=2020, bbox="0,0,1,1")
    assert cfg.provider == "geoportal"
    assert cfg.provider_options == {}


def test_index_config_accepts_nls_provider_with_options():
    cfg = IndexConfig(
        year_start=2020,
        year_end=2020,
        bbox="0,0,2000,2000",
        srs="EPSG:3067",
        provider="nls",
        provider_options={"api_key": "abc"},
    )
    assert cfg.provider == "nls"
    assert cfg.provider_options == {"api_key": "abc"}


def test_index_config_rejects_unknown_provider():
    with pytest.raises(ValidationError):
        IndexConfig(year_start=2020, year_end=2020, bbox="0,0,1,1", provider="boom")


def test_download_config_carries_provider_options():
    cfg = DownloadConfig(provider="nls", provider_options={"api_key": "abc"}, bbox="0,0,1,1", srs="EPSG:3067")
    assert cfg.provider == "nls"
    assert cfg.provider_options["api_key"] == "abc"


def test_index_config_nls_rejects_default_srs_2180():
    """Default srs is EPSG:2180 (Polish). Provider='nls' must require EPSG:3067."""
    with pytest.raises(ValidationError, match="EPSG:3067"):
        IndexConfig(year_start=2020, year_end=2020, bbox="0,0,1,1", provider="nls")


def test_index_config_nls_rejects_explicit_non_3067_srs():
    with pytest.raises(ValidationError, match="EPSG:3067"):
        IndexConfig(
            year_start=2020,
            year_end=2020,
            bbox="0,0,1,1",
            srs="EPSG:4326",
            provider="nls",
        )


def test_download_config_nls_rejects_non_3067_srs():
    with pytest.raises(ValidationError, match="EPSG:3067"):
        DownloadConfig(provider="nls", bbox="0,0,1,1", srs="EPSG:2180")


def test_geoportal_config_unaffected_by_nls_srs_check():
    # Default provider is geoportal; default srs is EPSG:2180. Must remain valid.
    cfg = IndexConfig(year_start=2020, year_end=2020, bbox="0,0,1,1")
    assert cfg.provider == "geoportal"
    assert cfg.srs == "EPSG:2180"
