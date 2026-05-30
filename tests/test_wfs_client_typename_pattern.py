import re
import xml.etree.ElementTree as ET

from satmap_dataset.geoportal import wfs_client

CAPS = """<?xml version='1.0'?>
<wfs:WFS_Capabilities xmlns:wfs="http://www.opengis.net/wfs/2.0">
  <wfs:FeatureTypeList>
    <wfs:FeatureType><wfs:Name>gugik:SkorowidzNMT2012</wfs:Name></wfs:FeatureType>
    <wfs:FeatureType><wfs:Name>gugik:SkorowidzNMT2019</wfs:Name></wfs:FeatureType>
    <wfs:FeatureType><wfs:Name>gugik:SkorowidzOrtofoto2021</wfs:Name></wfs:FeatureType>
  </wfs:FeatureTypeList>
</wfs:WFS_Capabilities>"""


def test_default_pattern_extracts_orto_years():
    root = ET.fromstring(CAPS)
    mapping = wfs_client._extract_year_typenames(root)
    assert mapping == {2021: "gugik:SkorowidzOrtofoto2021"}


def test_custom_pattern_extracts_nmt_years():
    root = ET.fromstring(CAPS)
    pattern = re.compile(r"SkorowidzNMT(\d{4})", re.IGNORECASE)
    mapping = wfs_client._extract_year_typenames(root, pattern)
    assert mapping == {2012: "gugik:SkorowidzNMT2012", 2019: "gugik:SkorowidzNMT2019"}
