import asyncio

import pytest

from satmap_dataset.geoportal import dem_skorowidz_client as dsc


def test_endpoint_all_combinations():
    assert dsc.endpoint("nmt", "kron86").endswith("NumerycznyModelTerenuKRON86/WFS/Skorowidze")
    assert dsc.endpoint("nmt", "evrf2007").endswith("NumerycznyModelTerenuEVRF2007/WFS/Skorowidze")
    assert dsc.endpoint("nmpt", "kron86").endswith("NumerycznyModelPokryciaTerenuKRON86/WFS/Skorowidze")
    assert dsc.endpoint("nmpt", "evrf2007").endswith("NumerycznyModelPokryciaTerenuEVRF2007/WFS/Skorowidze")


def test_endpoint_override_and_unknown():
    opts = {"skorowidz_endpoints": {"nmt|kron86": "https://example/custom"}}
    assert dsc.endpoint("nmt", "kron86", opts) == "https://example/custom"
    with pytest.raises(ValueError):
        dsc.endpoint("foo", "kron86")
    with pytest.raises(ValueError):
        dsc.endpoint("nmt", "baddatum")


def test_typename_pattern_matches_product_years():
    pat = dsc.typename_pattern("nmt")
    assert pat.search("gugik:SkorowidzNMT2019").group(1) == "2019"
    assert pat.search("gugik:SkorowidzNMPT2019") is None
    patp = dsc.typename_pattern("nmpt")
    assert patp.search("gugik:SkorowidzNMPT2018").group(1) == "2018"


def test_year_typenames_uses_wfs_client(monkeypatch):
    async def _fake_caps(base_url, *, timeout, retry_policy, typename_pattern):
        assert "NumerycznyModelTerenuKRON86" in base_url
        assert typename_pattern.search("gugik:SkorowidzNMT2012")
        return (None, {2012: "gugik:SkorowidzNMT2012"})

    monkeypatch.setattr(dsc.wfs_client, "get_capabilities", _fake_caps)
    out = asyncio.run(dsc.year_typenames("nmt", "kron86"))
    assert out == {2012: "gugik:SkorowidzNMT2012"}
