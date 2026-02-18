from __future__ import annotations

import sys
from pathlib import Path

import pytest
import typer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset import cli


def test_resolve_bbox_accepts_literal_bbox() -> None:
    bbox = cli._resolve_bbox_input(
        bbox="359700,504900,361700,506900",
        center_lat=None,
        center_lon=None,
        square_km=None,
        srs="EPSG:2180",
        required=True,
    )
    assert bbox == "359700,504900,361700,506900"


def test_resolve_bbox_rejects_mixed_literal_and_center_mode() -> None:
    with pytest.raises(typer.BadParameter):
        cli._resolve_bbox_input(
            bbox="359700,504900,361700,506900",
            center_lat=52.4012627,
            center_lon=16.9517999,
            square_km=4.0,
            srs="EPSG:2180",
            required=True,
        )


def test_resolve_bbox_requires_center_lat_and_lon() -> None:
    with pytest.raises(typer.BadParameter):
        cli._resolve_bbox_input(
            bbox=None,
            center_lat=52.4012627,
            center_lon=None,
            square_km=4.0,
            srs="EPSG:2180",
            required=True,
        )


def test_resolve_bbox_center_mode_defaults_to_4km(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_lonlat_to_epsg2180", lambda lon, lat: (360700.0, 505900.0))
    bbox = cli._resolve_bbox_input(
        bbox=None,
        center_lat=52.4012627,
        center_lon=16.9517999,
        square_km=None,
        srs="EPSG:2180",
        required=True,
    )
    assert bbox == "359700.000,504900.000,361700.000,506900.000"


def test_resolve_bbox_center_mode_requires_epsg2180() -> None:
    with pytest.raises(typer.BadParameter):
        cli._resolve_bbox_input(
            bbox=None,
            center_lat=52.4012627,
            center_lon=16.9517999,
            square_km=4.0,
            srs="EPSG:3857",
            required=True,
        )
