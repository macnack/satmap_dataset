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


def test_index_tiles_bbox_larger_than_2km(monkeypatch, tmp_path):
    # A 4.8 km x 3 km AOI (like a full SIVL area) must be split into a grid of
    # <= 2 km cells, one GetCoverage URL per (year, cell), and still succeed.
    _patch_fetchers(monkeypatch, available_years=[2018, 2020, 2022])
    cfg = _config(tmp_path, bbox="385000,6675000,389800,6678000")  # 4800m x 3000m
    exit_code, manifest_path = NlsProvider().index(cfg)
    assert exit_code == 0
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    # 4800/2000 -> 3 cols, 3000/2000 -> 2 rows = 6 cells per year.
    for year in data["years_included"]:
        sources = data["tile_sources_by_year"][str(year)]
        assert len(sources) == 6
        for tile_id, url in sources.items():
            assert tile_id.startswith(f"nls_{year}_")
            assert "request=GetCoverage" in url
        # Every cell stays within the 2 km cap on both sides.
        for tile_id, cell in data["tile_bboxes_by_year"][str(year)].items():
            assert cell[2] - cell[0] <= 2000.0 + 1e-6
            assert cell[3] - cell[1] <= 2000.0 + 1e-6


def test_split_bbox_into_grid_covers_aoi_without_gaps_or_overshoot():
    from satmap_dataset.providers.nls.provider import _split_bbox_into_grid

    bbox = (385000.0, 6675000.0, 389800.0, 6678000.0)  # 4800m x 3000m
    cells = _split_bbox_into_grid(bbox)
    assert len(cells) == 6  # 3 cols x 2 rows
    # The union of cells must exactly cover the AOI bounds.
    assert min(c[2][0] for c in cells) == bbox[0]
    assert min(c[2][1] for c in cells) == bbox[1]
    assert max(c[2][2] for c in cells) == bbox[2]
    assert max(c[2][3] for c in cells) == bbox[3]
    # No cell exceeds the cap on either side.
    for _, _, (cx0, cy0, cx1, cy1) in cells:
        assert cx1 - cx0 <= 2000.0 + 1e-6
        assert cy1 - cy0 <= 2000.0 + 1e-6


def test_split_bbox_into_grid_single_cell_for_small_aoi():
    from satmap_dataset.providers.nls.provider import _split_bbox_into_grid

    bbox = (385000.0, 6675000.0, 387000.0, 6677000.0)  # exactly 2 km square
    cells = _split_bbox_into_grid(bbox)
    assert cells == [(0, 0, bbox)]


def test_index_fails_when_no_years_in_range(monkeypatch, tmp_path):
    _patch_fetchers(monkeypatch, available_years=[])
    cfg = _config(tmp_path, year_start=2030, year_end=2031)
    exit_code, _ = NlsProvider().index(cfg)
    assert exit_code != 0


def test_index_follows_oapif_pagination_to_collect_all_years(monkeypatch, tmp_path):
    """Year coverage is the union across all paginated OAPIF pages, not just page 1."""
    monkeypatch.setattr(nls_provider, "_fetch_describe_coverage_xml", lambda **kw: _fixture_xml())

    pages = [
        # page 1: years 2018, 2020, plus a `next` link
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {"properties": {"kuvausvuosi": "2018"}},
                    {"properties": {"kuvausvuosi": "2020"}},
                ],
                "links": [
                    {"rel": "next", "href": "https://example.test/page2", "type": "application/geo+json"},
                ],
            }
        ).encode("utf-8"),
        # page 2: years 2022; no next link
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [{"properties": {"kuvausvuosi": "2022"}}],
                "links": [],
            }
        ).encode("utf-8"),
    ]

    import httpx

    def handler(request):
        return httpx.Response(200, content=pages[1])

    transport = httpx.MockTransport(handler)
    # First page comes from _fetch_oapif_items_geojson; subsequent pages from
    # the inline httpx.Client inside _fetch_oapif_aoi_years. Patch the first
    # fetcher to return page 1 and intercept httpx.Client to serve page 2.
    monkeypatch.setattr(nls_provider, "_fetch_oapif_items_geojson", lambda **kw: pages[0])
    real_client_init = httpx.Client.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["transport"] = transport
        real_client_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", patched_init)

    cfg = _config(tmp_path)
    exit_code, manifest_path = NlsProvider().index(cfg)
    assert exit_code == 0
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Without pagination, only [2018, 2020] would survive; with it 2022 also.
    assert data["years_included"] == [2018, 2020, 2022]


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
