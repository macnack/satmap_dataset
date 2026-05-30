import json
from pathlib import Path

from satmap_dataset.config import OsmConfig
from satmap_dataset.models import OsmCategoryAsset, OsmManifest, OsmYearAsset
from satmap_dataset.pipeline import osm as osm_pipeline


def test_osm_manifest_round_trip_with_null_rasters():
    manifest = OsmManifest(
        bbox="348967,508503,349967,509503",
        bbox_wgs84="16.778,52.421,16.792,52.430",
        srs="EPSG:2180",
        target_width=15000,
        target_height=15000,
        categories=["buildings", "highways"],
        years=[
            OsmYearAsset(
                year=2022,
                snapshot_date="2022-04-29T00:00:00Z",
                categories={
                    "buildings": OsmCategoryAsset(
                        feature_count=1326,
                        raster_path="osm_x/year_2022_buildings.tif",
                    ),
                    "highways": OsmCategoryAsset(
                        feature_count=0,
                        raster_path=None,
                    ),
                },
                passed=True,
            ),
        ],
        passed=True,
    )
    blob = manifest.model_dump_json()
    restored = OsmManifest.model_validate_json(blob)
    assert restored.kind == "osm_manifest"
    assert restored.stage == "osm"
    assert restored.years[0].year == 2022
    assert restored.years[0].categories["buildings"].feature_count == 1326
    assert restored.years[0].categories["buildings"].raster_path == "osm_x/year_2022_buildings.tif"
    assert restored.years[0].categories["highways"].raster_path is None
    assert restored.passed is True


def _patch_seams(monkeypatch, *, features_by_cat=None):
    counts = features_by_cat or {"buildings": 5, "highways": 3, "landuse": 2, "water": 1}

    async def _fake_fetch(bbox, filter_str, snapshot_date, **kwargs):
        for cat, flt in osm_pipeline.ohsome_client.CATEGORY_FILTERS.items():
            if flt == filter_str:
                n = counts.get(cat, 0)
                return {
                    "type": "FeatureCollection",
                    "features": [{"type": "Feature", "geometry": None, "properties": {}} for _ in range(n)],
                }
        return {"type": "FeatureCollection", "features": []}

    def _fake_rasterize(geojson, out_path, *, target_bbox, target_width, target_height, target_srs="EPSG:2180"):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"RASTER")

    monkeypatch.setattr(osm_pipeline.ohsome_client, "get_elements_geometry", _fake_fetch)
    monkeypatch.setattr(osm_pipeline.rasterize, "rasterize_geojson_to_file", _fake_rasterize)


def _make_render_manifest(tmp_path, *, years_dates: dict) -> Path:
    tile_acq = {
        str(year): {"tile_A": {"acquisition_date": date, "publication_date": None, "acquisition_year": year}}
        for year, date in years_dates.items()
    }
    data = {
        "kind": "dataset_manifest",
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
    manifest = OsmManifest.model_validate_json(Path(path).read_text())
    assert manifest.passed is True
    assert {a.year for a in manifest.years} == {2022, 2023}
    for year_asset in manifest.years:
        assert year_asset.passed is True
        for cat, asset in year_asset.categories.items():
            assert asset.feature_count > 0
            assert asset.raster_path is not None
            assert Path(asset.raster_path).exists()


def test_run_zero_features_no_raster(tmp_path, monkeypatch):
    _patch_seams(monkeypatch, features_by_cat={"buildings": 0, "highways": 0, "landuse": 0, "water": 0})
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
    manifest = OsmManifest.model_validate_json(Path(path).read_text())
    assert manifest.passed is True
    for cat, asset in manifest.years[0].categories.items():
        assert asset.feature_count == 0
        assert asset.raster_path is None


def test_run_uses_acquisition_date_not_jan1(tmp_path, monkeypatch):
    captured_dates = []

    async def _spy_fetch(bbox, filter_str, snapshot_date, **kwargs):
        captured_dates.append(snapshot_date)
        return {"type": "FeatureCollection", "features": []}

    monkeypatch.setattr(osm_pipeline.ohsome_client, "get_elements_geometry", _spy_fetch)

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

    async def _spy_fetch(bbox, filter_str, snapshot_date, **kwargs):
        fetch_calls.append(filter_str)
        return {"type": "FeatureCollection", "features": []}

    monkeypatch.setattr(osm_pipeline.ohsome_client, "get_elements_geometry", _spy_fetch)

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
    manifest = OsmManifest.model_validate_json(Path(path).read_text())
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
