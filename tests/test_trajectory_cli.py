import json
from pathlib import Path

from typer.testing import CliRunner

from satmap_dataset.cli import app

runner = CliRunner()


def _csv(tmp_path: Path) -> Path:
    p = tmp_path / "track.csv"
    p.write_text("lat,lon\n51.70227,17.83960\n51.70250,17.84050\n", encoding="utf-8")
    return p


def test_trajectory_flag_form(tmp_path: Path):
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        ["trajectory", "--track", str(_csv(tmp_path)), "--out", str(out), "--cell-km", "1.0"],
    )
    assert result.exit_code == 0, result.output
    manifest_path = out / "trajectory_tiles.json"
    assert manifest_path.exists()
    last_line = result.output.strip().splitlines()[-1]
    assert last_line.endswith("trajectory_tiles.json")


def test_trajectory_json_form(tmp_path: Path):
    out = tmp_path / "out"
    cfg = tmp_path / "cfg.json"
    cfg.write_text(
        json.dumps(
            {"track_path": str(_csv(tmp_path)), "output_dir": str(out), "download": False}
        ),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["trajectory-json", str(cfg)])
    assert result.exit_code == 0, result.output
    assert (out / "trajectory_tiles.json").exists()
