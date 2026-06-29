import json
from pathlib import Path

from satmap_dataset.config import TrajectoryConfig
from satmap_dataset.models import TrajectoryManifest
from satmap_dataset.pipeline import trajectory as traj_stage


def _write_csv(tmp_path: Path) -> Path:
    p = tmp_path / "track.csv"
    p.write_text(
        "lat,lon\n51.70227,17.83960\n51.70250,17.84050\n51.70300,17.84200\n",
        encoding="utf-8",
    )
    return p


def test_run_writes_manifest_and_preview(tmp_path: Path):
    csv_path = _write_csv(tmp_path)
    out = tmp_path / "out"
    cfg = TrajectoryConfig(track_path=csv_path, output_dir=out, download=False)
    code, path = traj_stage.run(cfg)
    assert code == 0
    assert path == out / "trajectory_tiles.json"
    assert path.exists()
    manifest = TrajectoryManifest.model_validate_json(path.read_text())
    assert manifest.point_count == 3
    assert manifest.cell_count >= 1
    assert manifest.cells[0].name.startswith("track_x")
    gj = json.loads((out / "trajectory_tiles.geojson").read_text())
    assert gj["type"] == "FeatureCollection"
    assert any(f["geometry"]["type"] == "LineString" for f in gj["features"])
    assert any(f["geometry"]["type"] == "Polygon" for f in gj["features"])


def test_run_no_preview(tmp_path: Path):
    csv_path = _write_csv(tmp_path)
    out = tmp_path / "out"
    cfg = TrajectoryConfig(track_path=csv_path, output_dir=out, preview=False)
    code, _ = traj_stage.run(cfg)
    assert code == 0
    assert not (out / "trajectory_tiles.geojson").exists()
