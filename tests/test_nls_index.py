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
    monkeypatch.setattr(nls_provider, "_fetch_describe_coverage_xml", lambda **kw: _fixture_xml())
    cfg = _config(tmp_path)
    exit_code, manifest_path = NlsProvider().index(cfg)
    assert exit_code == 0
    assert manifest_path == cfg.output_json
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["provider"] == "nls"
    # Real NLS catalogue covers 2018..2022 contiguously.
    assert data["years_included"] == [2018, 2019, 2020, 2021, 2022]
    for year in data["years_included"]:
        sources = data["tile_sources_by_year"][str(year)]
        assert list(sources.keys()) == [f"nls_{year}"]
        url = sources[f"nls_{year}"]
        assert "request=GetCoverage" in url
        # SUBSET=time("YYYY-12-31...) — both ( and " get URL-encoded by urlencode
        assert f"time%28%22{year}-12-31" in url or f'time("{year}-12-31' in url


def test_index_rejects_bbox_larger_than_2km(tmp_path):
    cfg = _config(tmp_path, bbox="385000,6675000,388000,6678000")  # 3 km square
    exit_code, _ = NlsProvider().index(cfg)
    assert exit_code != 0
    text = (tmp_path / "index_manifest.json").read_text(encoding="utf-8")
    assert "exceeds NLS WCS cap" in text or "bbox" in text.lower()


def test_index_fails_when_no_years_in_range(monkeypatch, tmp_path):
    monkeypatch.setattr(nls_provider, "_fetch_describe_coverage_xml", lambda **kw: _fixture_xml())
    cfg = _config(tmp_path, year_start=2030, year_end=2031)
    exit_code, _ = NlsProvider().index(cfg)
    assert exit_code != 0


def test_index_uses_default_wcs_url_when_not_overridden(monkeypatch, tmp_path):
    seen = {}

    def fake_fetch(**kwargs):
        seen.update(kwargs)
        return _fixture_xml()

    monkeypatch.setattr(nls_provider, "_fetch_describe_coverage_xml", fake_fetch)
    NlsProvider().index(_config(tmp_path))
    assert seen["base_url"].endswith("/wcs/v2")
