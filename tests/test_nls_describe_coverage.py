from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.providers.nls.wcs import parse_describe_coverage_years


def test_parse_describe_coverage_extracts_unique_sorted_years():
    fixture = (
        Path(__file__).parent / "fixtures" / "nls" / "describe_coverage_ortokuva_vari.xml"
    )
    xml_bytes = fixture.read_bytes()
    years = parse_describe_coverage_years(xml_bytes)
    assert years == [2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]


def test_parse_describe_coverage_handles_no_time_axis():
    xml_bytes = b'<?xml version="1.0"?><wcs:CoverageDescriptions xmlns:wcs="http://www.opengis.net/wcs/2.0"/>'
    years = parse_describe_coverage_years(xml_bytes)
    assert years == []
