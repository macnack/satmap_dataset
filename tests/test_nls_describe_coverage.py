from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.providers.nls.wcs import parse_describe_coverage_years


def test_parse_describe_coverage_extracts_unique_sorted_years():
    """Fixture is the real NLS DescribeCoverage response (sanitised, key-free)."""
    fixture = (
        Path(__file__).parent / "fixtures" / "nls" / "describe_coverage_ortokuva_vari.xml"
    )
    xml_bytes = fixture.read_bytes()
    years = parse_describe_coverage_years(xml_bytes)
    # Real NLS catalogue years for the colour orthophoto coverage.
    assert years == [
        1969, 1973, 2004, 2007, 2008, 2009, 2010, 2011, 2012,
        2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021,
        2022, 2023, 2024, 2025,
    ]


def test_parse_describe_coverage_handles_no_time_axis():
    xml_bytes = b'<?xml version="1.0"?><wcs:CoverageDescriptions xmlns:wcs="http://www.opengis.net/wcs/2.0"/>'
    years = parse_describe_coverage_years(xml_bytes)
    assert years == []
