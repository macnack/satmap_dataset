from pathlib import Path

from satmap_dataset.config import TrajectoryConfig
from satmap_dataset.models import TrajectoryManifest
from satmap_dataset.pipeline import trajectory as traj_stage


def _csv(tmp_path: Path) -> Path:
    p = tmp_path / "track.csv"
    p.write_text("lat,lon\n51.70227,17.83960\n51.70250,17.84050\n", encoding="utf-8")
    return p


def test_download_invokes_stages_per_cell(tmp_path: Path, monkeypatch):
    calls = {"index": [], "download": []}

    def fake_index_run(cfg):
        calls["index"].append(cfg.bbox)
        out = Path(cfg.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{}", encoding="utf-8")
        return 0, out

    def fake_download_run(cfg):
        calls["download"].append(cfg.bbox)
        out = Path(cfg.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{}", encoding="utf-8")
        return 0, out

    monkeypatch.setattr(traj_stage.index_builder, "run", fake_index_run)
    monkeypatch.setattr(traj_stage.downloader, "run", fake_download_run)

    out = tmp_path / "out"
    cfg = TrajectoryConfig(track_path=_csv(tmp_path), output_dir=out, download=True)
    code, path = traj_stage.run(cfg)
    assert code == 0
    manifest = TrajectoryManifest.model_validate_json(path.read_text())
    n = manifest.cell_count
    assert len(calls["index"]) == n
    assert len(calls["download"]) == n
    assert all(c.download_status == "ok" for c in manifest.cells)


def test_download_failure_marks_cell_and_exit_1(tmp_path: Path, monkeypatch):
    def fake_index_run(cfg):
        out = Path(cfg.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{}", encoding="utf-8")
        return 0, out

    def fail_download_run(cfg):
        return 1, Path(cfg.output_json)

    monkeypatch.setattr(traj_stage.index_builder, "run", fake_index_run)
    monkeypatch.setattr(traj_stage.downloader, "run", fail_download_run)

    out = tmp_path / "out"
    cfg = TrajectoryConfig(track_path=_csv(tmp_path), output_dir=out, download=True)
    code, path = traj_stage.run(cfg)
    assert code == 1
    manifest = TrajectoryManifest.model_validate_json(path.read_text())
    assert all(c.download_status == "failed" for c in manifest.cells)


def test_download_idempotent_skip(tmp_path: Path, monkeypatch):
    out = tmp_path / "out"
    cfg0 = TrajectoryConfig(track_path=_csv(tmp_path), output_dir=out, download=False)
    code0, path0 = traj_stage.run(cfg0)
    from satmap_dataset.models import TrajectoryManifest as TM

    name = TM.model_validate_json(path0.read_text()).cells[0].name
    cell_dir = out / name
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / "dataset_manifest_download.json").write_text("{}", encoding="utf-8")

    called = {"download": 0}

    def fake_index_run(cfg):
        o = Path(cfg.output_json)
        o.parent.mkdir(parents=True, exist_ok=True)
        o.write_text("{}", encoding="utf-8")
        return 0, o

    def fake_download_run(cfg):
        called["download"] += 1
        return 0, Path(cfg.output_json)

    monkeypatch.setattr(traj_stage.index_builder, "run", fake_index_run)
    monkeypatch.setattr(traj_stage.downloader, "run", fake_download_run)

    cfg = TrajectoryConfig(track_path=_csv(tmp_path), output_dir=out, download=True)
    code, path = traj_stage.run(cfg)
    assert code == 0
    manifest = TrajectoryManifest.model_validate_json(path.read_text())
    skipped = [c for c in manifest.cells if c.download_status == "skipped"]
    assert any(c.name == name for c in skipped)
    assert called["download"] == manifest.cell_count - len(skipped)
