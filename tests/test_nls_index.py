from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.config import IndexConfig
from satmap_dataset.providers.nls import provider as nls_provider
from satmap_dataset.providers.nls.provider import NlsProvider


def _fixture_xml() -> bytes:
    return (
        Path(__file__).parent / "fixtures" / "nls" / "describe_coverage_ortokuva_vari.xml"
    ).read_bytes()


def _fixture_oapif(years: list[int]) -> bytes:
    """Build a minimal OGC API Features GeoJSON response with the given kuvausvuosi values."""
    import json
    features = [
        {
            "type": "Feature",
            "id": str(i),
            "geometry": None,
            "properties": {"kuvausvuosi": str(year), "fid": i},
        }
        for i, year in enumerate(years)
    ]
    return json.dumps({"type": "FeatureCollection", "features": features}).encode("utf-8")


def _patch_fetchers(monkeypatch, available_years: list[int]):
    monkeypatch.setattr(nls_provider, "_fetch_describe_coverage_xml", lambda **kw: _fixture_xml())
    monkeypatch.setattr(
        nls_provider,
        "_fetch_oapif_items_geojson",
        lambda **kw: _fixture_oapif(available_years),
    )


def _config(tmp_path: Path, **overrides) -> IndexConfig:
    base = dict(
        year_start=2018,
        year_end=2022,
        bbox="385000,6675000,387000,6677000",
        srs="EPSG:3067",
        provider="nls",
        provider_options={"api_key": "test-key"},
        output_json=tmp_path / "index_manifest.json",
        year_availability_output_json=tmp_path / "year_availability_report.json",
    )
    base.update(overrides)
    return IndexConfig(**base)


def test_index_writes_manifest_with_one_url_per_year(monkeypatch, tmp_path):
    _patch_fetchers(monkeypatch, available_years=[2018, 2020, 2022])
    cfg = _config(tmp_path)
    exit_code, manifest_path = NlsProvider().index(cfg)
    assert exit_code == 0
    assert manifest_path == cfg.output_json
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["provider"] == "nls"
    # OAPIF says the AOI has data only in 2018, 2020, 2022 (within the
    # 2018-2022 request range), so years_included reflects that.
    assert data["years_included"] == [2018, 2020, 2022]
    for year in data["years_included"]:
        sources = data["tile_sources_by_year"][str(year)]
        assert list(sources.keys()) == [f"nls_{year}"]
        url = sources[f"nls_{year}"]
        assert "request=GetCoverage" in url
        # SUBSET=time("YYYY-12-31...) — both ( and " get URL-encoded by urlencode
        assert f"time%28%22{year}-12-31" in url or f'time("{year}-12-31' in url


def test_index_excludes_years_with_no_aoi_coverage(monkeypatch, tmp_path):
    _patch_fetchers(monkeypatch, available_years=[2018, 2020, 2022])
    cfg = _config(tmp_path)
    exit_code, manifest_path = NlsProvider().index(cfg)
    assert exit_code == 0
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    excluded = data["years_excluded_with_reason"]
    # 2019 and 2021 are in the WCS catalogue but OAPIF says no photos here.
    assert excluded["2019"] == "no_orthophoto_for_aoi_at_this_year"
    assert excluded["2021"] == "no_orthophoto_for_aoi_at_this_year"


def test_index_rejects_bbox_larger_than_2km(tmp_path):
    cfg = _config(tmp_path, bbox="385000,6675000,388000,6678000")  # 3 km square
    exit_code, _ = NlsProvider().index(cfg)
    assert exit_code != 0
    text = (tmp_path / "index_manifest.json").read_text(encoding="utf-8")
    assert "exceeds NLS WCS cap" in text or "bbox" in text.lower()


def test_index_fails_when_no_years_in_range(monkeypatch, tmp_path):
    _patch_fetchers(monkeypatch, available_years=[])
    cfg = _config(tmp_path, year_start=2030, year_end=2031)
    exit_code, _ = NlsProvider().index(cfg)
    assert exit_code != 0


def test_index_uses_default_wcs_url_when_not_overridden(monkeypatch, tmp_path):
    seen = {}

    def fake_fetch(**kwargs):
        seen.update(kwargs)
        return _fixture_xml()

    monkeypatch.setattr(nls_provider, "_fetch_describe_coverage_xml", fake_fetch)
    monkeypatch.setattr(
        nls_provider, "_fetch_oapif_items_geojson", lambda **kw: _fixture_oapif([2018, 2019, 2020, 2021, 2022])
    )
    NlsProvider().index(_config(tmp_path))
    assert seen["base_url"].endswith("/wcs/v2")
