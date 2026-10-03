from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.config import DemConfig


def test_lantmateriet_dem_defaults():
    cfg = DemConfig(bbox="600000,6500000,600200,6500200", provider="lantmateriet")
    assert cfg.transport == "stac_hojd"
    assert cfg.products == ["nmt"]
    assert cfg.vertical_datum == "rh2000"
    assert cfg.srs == "EPSG:3006"


def test_lantmateriet_rejects_nmpt_and_polish_transport():
    with pytest.raises(ValidationError):
        DemConfig(
            bbox="600000,6500000,600200,6500200",
            provider="lantmateriet",
            products=["nmt", "nmpt"],
        )
    with pytest.raises(ValidationError):
        DemConfig(
            bbox="600000,6500000,600200,6500200",
            provider="lantmateriet",
            transport="wcs",
        )
    with pytest.raises(ValidationError):
        DemConfig(
            bbox="600000,6500000,600200,6500200",
            provider="lantmateriet",
            vertical_datum="evrf2007",
        )


def test_stac_hojd_requires_lantmateriet_provider():
    with pytest.raises(ValidationError):
        DemConfig(bbox="0,0,10,10", transport="stac_hojd")
    with pytest.raises(ValidationError):
        DemConfig(bbox="0,0,10,10", vertical_datum="rh2000")


def test_geoportal_still_defaults_wcs():
    cfg = DemConfig(bbox="0,0,10,10")
    assert cfg.provider == "geoportal"
    assert cfg.transport == "wcs"
    assert cfg.products == ["nmt", "nmpt"]
