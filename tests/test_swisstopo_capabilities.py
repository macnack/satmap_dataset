from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.providers.swisstopo.capabilities import (
    CapabilitiesParseError,
    expand_wms_time_dimension,
    intersect_years,
    parse_time_years,
)

FIXTURE = ROOT / "tests" / "fixtures" / "swisstopo" / "wmts_capabilities_swissimage_product.xml"


def test_parse_time_years_from_fixture() -> None:
    years = parse_time_years(FIXTURE.read_text(encoding="utf-8"))
    assert years[0] == 1926
    assert years[-1] == 2025
    assert len(years) == 99
    assert 1946 in years
    assert 1928 not in years  # only gap in the early Zeitreise list


def test_parse_missing_layer_raises() -> None:
    with pytest.raises(CapabilitiesParseError, match="not found"):
        parse_time_years(FIXTURE.read_text(encoding="utf-8"), layer_id="no.such.layer")


def test_expand_wms_time_dimension() -> None:
    years = expand_wms_time_dimension("1946,1947,1949/1952/P1Y,9999")
    assert years == [1946, 1947, 1949, 1950, 1951, 1952]


def test_intersect_years() -> None:
    assert intersect_years([1950, 1951, 1952], [1950, 1952, 1954]) == [1950, 1952]
