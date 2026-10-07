from pathlib import Path

from satmap_dataset.config import TrajectoryConfig
from satmap_dataset.models import TrajectoryManifest
from satmap_dataset.pipeline import trajectory as traj_stage


def _csv(tmp_path: Path) -> Path:
    p = tmp_path / "track.csv"
    p.write_text("lat,lon\n51.70227,17.83960\n51.70250,17.84050\n", encoding="utf-8")
    return p


class _FakeProvider:
    name = "geoportal"

    def __init__(self, *, fail_download: bool = False) -> None:
        self.fail_download = fail_download
        self.index_calls: list = []
        self.download_calls: list = []

    def index(self, cfg):
        self.index_calls.append(cfg.bbox)
        out = Path(cfg.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{}", encoding="utf-8")
        return 0, out

    def download(self, cfg):
        self.download_calls.append(cfg.bbox)
        out = Path(cfg.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{}", encoding="utf-8")
        return (1 if self.fail_download else 0), out


def test_download_invokes_stages_per_cell(tmp_path: Path, monkeypatch):
    fake = _FakeProvider()
    monkeypatch.setattr(traj_stage, "get_provider", lambda name: fake)

    out = tmp_path / "out"
    cfg = TrajectoryConfig(track_path=_csv(tmp_path), output_dir=out, download=True)
    code, path = traj_stage.run(cfg)
    assert code == 0
    manifest = TrajectoryManifest.model_validate_json(path.read_text())
    n = manifest.cell_count
    assert len(fake.index_calls) == n
    assert len(fake.download_calls) == n
    assert all(c.download_status == "ok" for c in manifest.cells)


def test_download_failure_marks_cell_and_exit_1(tmp_path: Path, monkeypatch):
    fake = _FakeProvider(fail_download=True)
    monkeypatch.setattr(traj_stage, "get_provider", lambda name: fake)

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

    fake = _FakeProvider()
    monkeypatch.setattr(traj_stage, "get_provider", lambda name: fake)

    cfg = TrajectoryConfig(track_path=_csv(tmp_path), output_dir=out, download=True)
    code, path = traj_stage.run(cfg)
    assert code == 0
    manifest = TrajectoryManifest.model_validate_json(path.read_text())
    skipped = [c for c in manifest.cells if c.download_status == "skipped"]
    assert any(c.name == name for c in skipped)
    assert len(fake.download_calls) == manifest.cell_count - len(skipped)


def test_download_uses_configured_provider(tmp_path: Path, monkeypatch):
    """Trajectory download must route through get_provider(config.provider)."""
    seen: dict[str, list] = {"names": [], "index_providers": [], "download_providers": []}

    class FakeProvider:
        name = "landsd_hk"

        def index(self, cfg):
            seen["index_providers"].append(cfg.provider)
            out = Path(cfg.output_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("{}", encoding="utf-8")
            return 0, out

        def download(self, cfg):
            seen["download_providers"].append(cfg.provider)
            out = Path(cfg.output_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("{}", encoding="utf-8")
            return 0, out

    def fake_get_provider(name: str):
        seen["names"].append(name)
        return FakeProvider()

    monkeypatch.setattr(traj_stage, "get_provider", fake_get_provider)

    out = tmp_path / "out"
    cfg = TrajectoryConfig(
        track_path=_csv(tmp_path),
        output_dir=out,
        download=True,
        provider="landsd_hk",
        srs="EPSG:3857",
        mode="wms_tiled",
        year_start=2025,
        year_end=2025,
        provider_options={"imagery_year": 2025, "zoom": 18},
        sleep_min=0.0,
        sleep_max=0.0,
    )
    code, path = traj_stage.run(cfg)
    assert code == 0
    assert seen["names"] and all(n == "landsd_hk" for n in seen["names"])
    assert seen["index_providers"] and seen["download_providers"]
    assert all(p == "landsd_hk" for p in seen["index_providers"] + seen["download_providers"])
    # landsd_hk with default EPSG:2180 should auto-switch to 3857 when srs left default —
    # here we passed 3857 explicitly; also check auto default:
    cfg2 = TrajectoryConfig(
        track_path=_csv(tmp_path),
        output_dir=tmp_path / "out2",
        provider="landsd_hk",
        mode="hybrid",
    )
    assert cfg2.srs == "EPSG:3857"
    assert cfg2.mode == "wms_tiled"
