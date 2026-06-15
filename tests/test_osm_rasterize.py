import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from satmap_dataset.osm import rasterize


SAMPLE_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[16.779, 52.422], [16.780, 52.422], [16.780, 52.423], [16.779, 52.422]]],
            },
            "properties": {"building": "yes"},
        }
    ],
}


def test_rasterize_calls_ogr2ogr_and_gdal_rasterize(tmp_path, monkeypatch):
    calls = []

    def _fake_run(cmd, *, check, capture_output, text):
        calls.append(cmd[0])
        if cmd[0] == "gdal_rasterize":
            Path(cmd[-1]).touch()
        return MagicMock(returncode=0)

    monkeypatch.setattr(rasterize.subprocess, "run", _fake_run)
    monkeypatch.setattr(rasterize, "_tool_path", lambda name: f"/usr/bin/{name}")

    out = tmp_path / "test.tif"
    rasterize.rasterize_geojson_to_file(
        SAMPLE_GEOJSON, out,
        target_bbox=(348967.0, 508503.0, 349967.0, 509503.0),
        target_width=1000, target_height=1000,
    )
    assert "ogr2ogr" in calls
    assert "gdal_rasterize" in calls


def test_rasterize_gdal_rasterize_args(tmp_path, monkeypatch):
    captured = {}

    def _fake_run(cmd, *, check, capture_output, text):
        if cmd[0] == "gdal_rasterize":
            captured["cmd"] = cmd
            Path(cmd[-1]).touch()
        elif cmd[0] == "ogr2ogr":
            # ogr2ogr output is the reproj path at cmd[5]; cmd[3] is the -t_srs flag.
            Path(cmd[5]).write_text(json.dumps(SAMPLE_GEOJSON))
        return MagicMock(returncode=0)

    monkeypatch.setattr(rasterize.subprocess, "run", _fake_run)
    monkeypatch.setattr(rasterize, "_tool_path", lambda name: f"/usr/bin/{name}")

    out = tmp_path / "labels.tif"
    rasterize.rasterize_geojson_to_file(
        SAMPLE_GEOJSON, out,
        target_bbox=(100.0, 200.0, 300.0, 400.0),
        target_width=2000, target_height=2000, target_srs="EPSG:2180",
    )
    cmd = captured["cmd"]
    assert "-burn" in cmd and "255" in cmd
    ts_idx = cmd.index("-ts")
    assert cmd[ts_idx + 1] == "2000" and cmd[ts_idx + 2] == "2000"
    te_idx = cmd.index("-te")
    assert cmd[te_idx + 1] == "100.0"
    assert "-ot" in cmd and "Byte" in cmd
    assert "-co" in cmd and "COMPRESS=DEFLATE" in cmd
    assert str(out) == cmd[-1]


def test_rasterize_raises_when_gdal_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(rasterize, "_tool_path", lambda name: None)
    with pytest.raises(RuntimeError, match="gdal_rasterize"):
        rasterize.rasterize_geojson_to_file(
            SAMPLE_GEOJSON, tmp_path / "out.tif",
            target_bbox=(0.0, 0.0, 1.0, 1.0), target_width=10, target_height=10,
        )


def test_rasterize_writes_geojson_to_temp_file(tmp_path, monkeypatch):
    written_paths = []
    original_write = Path.write_text

    def _capture_write(self, data, **kwargs):
        if self.name == "src.geojson":
            written_paths.append(json.loads(data))
        return original_write(self, data, **kwargs)

    def _fake_run(cmd, *, check, capture_output, text):
        if cmd[0] == "gdal_rasterize":
            Path(cmd[-1]).touch()
        elif cmd[0] == "ogr2ogr":
            # ogr2ogr output is the reproj path at cmd[5]; cmd[3] is the -t_srs flag.
            Path(cmd[5]).write_text(json.dumps(SAMPLE_GEOJSON))
        return MagicMock(returncode=0)

    monkeypatch.setattr(rasterize.subprocess, "run", _fake_run)
    monkeypatch.setattr(rasterize, "_tool_path", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(Path, "write_text", _capture_write)

    rasterize.rasterize_geojson_to_file(
        SAMPLE_GEOJSON, tmp_path / "out.tif",
        target_bbox=(0.0, 0.0, 100.0, 100.0), target_width=100, target_height=100,
    )
    assert len(written_paths) == 1
    assert written_paths[0]["type"] == "FeatureCollection"
