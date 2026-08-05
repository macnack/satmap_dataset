"""Smoke tests for studio config builders."""

from __future__ import annotations

from pathlib import Path

from satmap_dataset.config import IndexConfig, RunConfig
from satmap_dataset.studio.config_builders import (
    build_index_config,
    build_location_payload,
    build_run_config,
    resolve_base_json,
)

REPO = Path(__file__).resolve().parents[1]
POZNAN_LAT = 52.4012627
POZNAN_LON = 16.9517999


def test_build_geoportal_index_config():
    payload = build_location_payload(
        location_name="Poznan",
        center_lat=POZNAN_LAT,
        center_lon=POZNAN_LON,
        area_km2=4.0,
        provider="geoportal",
        year_start=2015,
        year_end=2020,
        px_per_meter=15.0,
        profile="reference",
    )
    base_json = resolve_base_json("geoportal", REPO)
    config = build_index_config(payload, base_json)
    assert isinstance(config, IndexConfig)
    assert config.provider == "geoportal"
    assert config.srs == "EPSG:2180"
    parts = [float(p) for p in config.bbox.split(",")]
    assert len(parts) == 4
    assert parts[0] < parts[2]


def test_build_lantmateriet_run_config():
    payload = build_location_payload(
        location_name="Kisa",
        center_lat=57.985,
        center_lon=15.629,
        area_km2=4.0,
        provider="lantmateriet",
        year_start=2010,
        year_end=2014,
        px_per_meter=5.0,
        profile="train",
    )
    base_json = resolve_base_json("lantmateriet", REPO)
    config = build_run_config(payload, base_json)
    assert isinstance(config, RunConfig)
    assert config.srs == "EPSG:3006"
