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
    cfg = DownloadConfig(provider="nls", provider_options={"api_key": "abc"}, bbox="0,0,1,1")
    assert cfg.provider == "nls"
    assert cfg.provider_options["api_key"] == "abc"
