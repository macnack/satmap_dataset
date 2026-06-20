from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.geoportal import wfs_client


def _feature(piksel_xml: str) -> ET.Element:
    xml = f"""<wfs:member xmlns:gugik="http://www.gugik.gov.pl" xmlns:wfs="http://www.opengis.net/wfs/2.0">
      <gugik:SkorowidzOrtofomapy2024>
        <gugik:godlo>N-33-130-D-d-1-2</gugik:godlo>
        <gugik:akt_rok>2024</gugik:akt_rok>
        {piksel_xml}
        <gugik:url_do_pobrania>https://x/y_N-33-130-D-d-1-2.tif</gugik:url_do_pobrania>
      </gugik:SkorowidzOrtofomapy2024>
    </wfs:member>"""
    return ET.fromstring(xml)


def test_parse_float_or_none():
    assert wfs_client._parse_float_or_none("0.05") == 0.05
    assert wfs_client._parse_float_or_none(" 0.25 ") == 0.25
    assert wfs_client._parse_float_or_none("") is None
    assert wfs_client._parse_float_or_none(None) is None
    assert wfs_client._parse_float_or_none("abc") is None


def test_extract_metadata_includes_gsd():
    meta = wfs_client._extract_tile_acquisition_metadata(
        _feature("<gugik:piksel>0.05</gugik:piksel>"), 2024
    )
    assert meta["gsd"] == 0.05


def test_extract_metadata_missing_piksel():
    meta = wfs_client._extract_tile_acquisition_metadata(_feature(""), 2024)
    assert meta["gsd"] is None
