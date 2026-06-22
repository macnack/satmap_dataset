import json
from pathlib import Path

from satmap_dataset.config import OsmConfig
from satmap_dataset.models import LayerManifest, OsmCategoryAsset, OsmYearAsset
from satmap_dataset.pipeline import osm as osm_pipeline


def test_build_osm_layer_manifest_maps_categories_and_rasters():
    cfg = OsmConfig(
        bbox="348967,508503,349967,509503",
        srs="EPSG:2180",
        osm_root="osm_x",
        output_json="osm_x/osm_manifest.json",
        categories=["buildings", "roads"],
    )
    year_assets = [
        OsmYearAsset(
            year=2022,
            snapshot_date="2022-04-29T00:00:00Z",
            categories={
                "buildings": OsmCategoryAsset(
                    feature_count=1326, raster_path="osm_x/year_2022_buildings.tif"
                ),
                "roads": OsmCategoryAsset(feature_count=0, raster_path=None),
            },
            passed=True,
        ),
    ]
    manifest = osm_pipeline.build_osm_layer_manifest(
        cfg, year_assets, bbox_wgs84="16.778,52.421,16.792,52.430",
        target_width=15000, target_height=15000, passed=True, errors=[],
    )
    restored = LayerManifest.model_validate_json(manifest.model_dump_json())
    assert restored.role == "labels"
    assert restored.layer == "osm"
    assert restored.provider is None
    assert restored.bands == ["buildings", "roads"]
    year = restored.years[0]
    assert year.year == 2022
    assert year.feature_counts["buildings"] == 1326
    assert year.assets["buildings"] == "osm_x/year_2022_buildings.tif"
    # zero-feature category has a count but no raster asset.
    assert year.feature_counts["roads"] == 0
    assert "roads" not in year.assets
    assert restored.grid is not None and restored.grid.width == 15000
    assert restored.grid.year_date_map == {2022: "2022-04-29T00:00:00Z"}


def _patch_seams(monkeypatch, *, features_by_cat=None):
    counts = features_by_cat or {"buildings": 5, "roads": 3, "paths": 2, "green": 2, "water": 1}

    async def _fake_fetch(bbox, category, snapshot_date, **kwargs):
        n = counts.get(category, 0)
        return {
            "type": "FeatureCollection",
            "features": [{"type": "Feature", "geometry": None, "properties": {}} for _ in range(n)],
        }

    def _fake_rasterize(geojson, out_path, *, target_bbox, target_width, target_height, target_srs="EPSG:2180"):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"RASTER")

    monkeypatch.setattr(osm_pipeline.overpass_client, "get_elements_geometry", _fake_fetch)
    monkeypatch.setattr(osm_pipeline.rasterize, "rasterize_geojson_to_file", _fake_rasterize)


def _make_render_manifest(tmp_path, *, years_dates: dict) -> Path:
    tile_acq = {
        str(year): {"tile_A": {"acquisition_date": date, "publication_date": None, "acquisition_year": year}}
        for year, date in years_dates.items()
    }
    data = {
        "kind": "layer_manifest",
        "layer": "geoportal_rgb",
        "role": "rgb",
        "stage": "render",
        "target_bbox": "0,0,100,100",
        "target_width": 100,
        "target_height": 100,
        "tile_acquisition_by_year": tile_acq,
        "years_included": list(years_dates.keys()),
    }
    path = tmp_path / "dataset_manifest_render.json"
    path.write_text(json.dumps(data))
    return path


def test_run_writes_rasters_and_manifest(tmp_path, monkeypatch):
    _patch_seams(monkeypatch)
    render = _make_render_manifest(tmp_path, years_dates={2022: "2022-04-29", 2023: "2023-05-21"})
    cfg = OsmConfig(
        bbox="0,0,100,100",
        osm_root=tmp_path / "osm_x",
        output_json=tmp_path / "osm_x" / "osm_manifest.json",
        render_manifest=render,
        target_width=100,
        target_height=100,
        sleep_min=0.0, sleep_max=0.0,
    )
    code, path = osm_pipeline.run(cfg)
    assert code == 0
    manifest = LayerManifest.model_validate_json(Path(path).read_text())
    assert manifest.passed is True
    assert {a.year for a in manifest.years} == {2022, 2023}
    for year_asset in manifest.years:
        assert year_asset.passed is True
        for cat, count in year_asset.feature_counts.items():
            assert count > 0
            assert cat in year_asset.assets
            assert Path(year_asset.assets[cat]).exists()


def test_run_accepts_epsg3006_and_converts_bbox(tmp_path, monkeypatch):
    captured = {}

    async def _spy_fetch(bbox, category, snapshot_date, **kwargs):
        captured["bbox"] = bbox
        return {"type": "FeatureCollection", "features": []}

    monkeypatch.setattr(osm_pipeline.overpass_client, "get_elements_geometry", _spy_fetch)
    monkeypatch.setattr(
        osm_pipeline.rasterize, "rasterize_geojson_to_file", lambda *a, **k: None
    )
    render = _make_render_manifest(tmp_path, years_dates={2021: "2021-07-01"})
    cfg = OsmConfig(
        bbox="717646.631,7534133.478,721519.615,7538006.462",
        srs="EPSG:3006",
        osm_root=tmp_path / "osm_k",
        output_json=tmp_path / "osm_k" / "osm_manifest.json",
        render_manifest=render,
        categories=["buildings"],
        target_width=10,
        target_height=10,
        sleep_min=0.0, sleep_max=0.0,
    )
    code, path = osm_pipeline.run(cfg)
    manifest = LayerManifest.model_validate_json(Path(path).read_text())
    # The 2180-only guard must not trip for a valid Swedish CRS.
    assert not any("only supports" in e for e in manifest.errors), manifest.errors
    # The EPSG:3006 AOI must be converted to Kiruna WGS84 lon/lat, not left as
    # metres or mis-projected through the old EPSG:2180 path.
    lon_min, lat_min = (float(x) for x in captured["bbox"].split(",")[:2])
    assert 20.1 < lon_min < 20.3, captured["bbox"]
    assert 67.7 < lat_min < 68.0, captured["bbox"]


def test_run_zero_features_no_raster(tmp_path, monkeypatch):
    _patch_seams(monkeypatch, features_by_cat={"buildings": 0, "roads": 0, "paths": 0, "green": 0, "water": 0})
    render = _make_render_manifest(tmp_path, years_dates={2015: "2015-06-01"})
    cfg = OsmConfig(
        bbox="0,0,100,100",
        osm_root=tmp_path / "osm_x",
        output_json=tmp_path / "osm_x" / "osm_manifest.json",
        render_manifest=render,
        target_width=100,
        target_height=100,
        sleep_min=0.0, sleep_max=0.0,
    )
    code, path = osm_pipeline.run(cfg)
    assert code == 0
    manifest = LayerManifest.model_validate_json(Path(path).read_text())
    assert manifest.passed is True
    year = manifest.years[0]
    for cat, count in year.feature_counts.items():
        assert count == 0
        assert cat not in year.assets


def test_run_uses_acquisition_date_not_jan1(tmp_path, monkeypatch):
    captured_dates = []

    async def _spy_fetch(bbox, category, snapshot_date, **kwargs):
        captured_dates.append(snapshot_date)
        return {"type": "FeatureCollection", "features": []}

    monkeypatch.setattr(osm_pipeline.overpass_client, "get_elements_geometry", _spy_fetch)

    render = _make_render_manifest(tmp_path, years_dates={2022: "2022-04-29"})
    cfg = OsmConfig(
        bbox="0,0,100,100",
        osm_root=tmp_path / "osm_x",
        output_json=tmp_path / "osm_x" / "osm_manifest.json",
        render_manifest=render,
        categories=["buildings"],
        target_width=10,
        target_height=10,
        sleep_min=0.0, sleep_max=0.0,
    )
    osm_pipeline.run(cfg)
    assert all("2022-04-29" in d for d in captured_dates), captured_dates


def test_run_reuses_existing_raster(tmp_path, monkeypatch):
    fetch_calls = []

    async def _spy_fetch(bbox, category, snapshot_date, **kwargs):
        fetch_calls.append(category)
        return {"type": "FeatureCollection", "features": []}

    monkeypatch.setattr(osm_pipeline.overpass_client, "get_elements_geometry", _spy_fetch)

    existing = tmp_path / "osm_x" / "year_2022_buildings.tif"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"CACHED")

    render = _make_render_manifest(tmp_path, years_dates={2022: "2022-04-29"})
    cfg = OsmConfig(
        bbox="0,0,100,100",
        osm_root=tmp_path / "osm_x",
        output_json=tmp_path / "osm_x" / "osm_manifest.json",
        render_manifest=render,
        categories=["buildings"],
        target_width=10,
        target_height=10,
        overwrite=False,
        sleep_min=0.0, sleep_max=0.0,
    )
    osm_pipeline.run(cfg)
    assert not any("building" in f for f in fetch_calls)


def test_run_no_year_date_source_raises(tmp_path):
    cfg = OsmConfig(
        bbox="0,0,100,100",
        osm_root=tmp_path / "osm_x",
        output_json=tmp_path / "osm_x" / "osm_manifest.json",
        sleep_min=0.0, sleep_max=0.0,
    )
    code, path = osm_pipeline.run(cfg)
    assert code == 1
    manifest = LayerManifest.model_validate_json(Path(path).read_text())
    assert manifest.passed is False
    assert manifest.errors


def test_read_year_date_map_from_render_manifest(tmp_path):
    render = _make_render_manifest(tmp_path, years_dates={2022: "2022-04-29", 2023: "2023-05-21"})
    cfg = OsmConfig(bbox="0,0,10,10", render_manifest=render, sleep_min=0.0, sleep_max=0.0)
    result = osm_pipeline._read_year_date_map(cfg)
    assert result[2022] == "2022-04-29"
    assert result[2023] == "2023-05-21"


def test_read_year_date_map_prefers_explicit(tmp_path):
    render = _make_render_manifest(tmp_path, years_dates={2022: "2022-04-29"})
    cfg = OsmConfig(
        bbox="0,0,10,10",
        render_manifest=render,
        year_date_map={2020: "2020-07-15"},
        sleep_min=0.0, sleep_max=0.0,
    )
    result = osm_pipeline._read_year_date_map(cfg)
    assert list(result.keys()) == [2020]
    assert result[2020] == "2020-07-15"
