from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from satmap_dataset.config import LrocProjectConfig
from satmap_dataset.models import LayerManifest, LrocProjectManifest
from satmap_dataset.pipeline import lroc_project


def _download_manifest(tmp_path: Path, assets: list[Path], *, provider: str = "lroc_nac") -> Path:
    for asset in assets:
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_bytes(b"PDS-IMG")
    manifest = LayerManifest(
        layer="lroc_nac_mono",
        role="rgb",
        stage="download",
        provider=provider,
        assets=[str(a) for a in assets],
        years_included=[2011],
        passed=True,
    )
    path = tmp_path / "dataset_manifest_download.json"
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return path


def _fake_which_all(name: str) -> str | None:
    return f"/opt/isis/{name}"


def _fake_run_success(argv: list[str]) -> subprocess.CompletedProcess[str]:
    # Create expected outputs for lronac2isis / cam2map so the stage can verify them.
    for arg in argv:
        if arg.startswith("to="):
            out = Path(arg.split("=", 1)[1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"CUBE")
    return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")


def test_lroc_project_config_rejects_earth_srs() -> None:
    with pytest.raises(ValidationError):
        LrocProjectConfig(
            download_manifest=Path("m.json"),
            srs="EPSG:2180",
        )


def test_lroc_project_config_accepts_lunar_srs() -> None:
    cfg = LrocProjectConfig(
        download_manifest=Path("m.json"),
        srs="IAU_2015:30100",
    )
    assert cfg.srs == "IAU_2015:30100"


def test_missing_isis_writes_manifest_exit_1(tmp_path: Path) -> None:
    img = tmp_path / "downloads" / "2011" / "M111.IMG"
    download_manifest = _download_manifest(tmp_path, [img])
    out = tmp_path / "artifacts" / "project_manifest.json"
    cfg = LrocProjectConfig(
        download_manifest=download_manifest,
        project_root=tmp_path / "projected",
        output_json=out,
        artifacts_dir=tmp_path / "artifacts",
    )

    code, path = lroc_project.run(cfg, which=lambda _name: None)
    assert code == 1
    assert path == out
    manifest = LrocProjectManifest.model_validate_json(out.read_text(encoding="utf-8"))
    assert manifest.passed is False
    assert any("ISIS" in e for e in manifest.errors)
    assert any("PATH" in e for e in manifest.errors)


def test_happy_path_argv_and_manifest(tmp_path: Path) -> None:
    img = tmp_path / "downloads" / "2011" / "M111.IMG"
    download_manifest = _download_manifest(tmp_path, [img])
    out = tmp_path / "artifacts" / "project_manifest.json"
    project_root = tmp_path / "projected"
    map_file = tmp_path / "moon.map"
    map_file.write_text("Group = Mapping\nEnd_Group\n", encoding="utf-8")
    cfg = LrocProjectConfig(
        download_manifest=download_manifest,
        project_root=project_root,
        output_json=out,
        artifacts_dir=tmp_path / "artifacts",
        map_file=map_file,
        keep_work_cubes=True,
    )

    seen: list[list[str]] = []

    def run_cmd(argv: list[str]) -> subprocess.CompletedProcess[str]:
        seen.append(list(argv))
        return _fake_run_success(list(argv))

    code, path = lroc_project.run(cfg, which=_fake_which_all, run_cmd=run_cmd)
    assert code == 0
    assert path == out

    assert len(seen) == 3
    assert seen[0][0] == "/opt/isis/lronac2isis"
    assert seen[0][1] == f"from={img}"
    assert seen[1][0] == "/opt/isis/spiceinit"
    assert seen[2][0] == "/opt/isis/cam2map"
    assert any(a.startswith("map=") for a in seen[2])

    manifest = LrocProjectManifest.model_validate_json(out.read_text(encoding="utf-8"))
    assert manifest.passed is True
    assert manifest.kind == "lroc_project_manifest"
    assert manifest.srs == "IAU_2015:30100"
    assert len(manifest.assets_projected) == 1
    assert manifest.assets_projected[0].endswith("M111_map.cub")
    assert manifest.isis_tools["cam2map"] == "/opt/isis/cam2map"
    assert not manifest.assets_failed


def test_wrong_provider_exit_1(tmp_path: Path) -> None:
    img = tmp_path / "downloads" / "2011" / "x.IMG"
    download_manifest = _download_manifest(tmp_path, [img], provider="geoportal")
    out = tmp_path / "artifacts" / "project_manifest.json"
    cfg = LrocProjectConfig(
        download_manifest=download_manifest,
        project_root=tmp_path / "projected",
        output_json=out,
    )
    code, _path = lroc_project.run(cfg, which=_fake_which_all, run_cmd=_fake_run_success)
    assert code == 1
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["passed"] is False
    assert any("lroc_nac" in e for e in data["errors"])


def test_reuse_existing_map_cube(tmp_path: Path) -> None:
    img = tmp_path / "downloads" / "2011" / "M111.IMG"
    download_manifest = _download_manifest(tmp_path, [img])
    project_root = tmp_path / "projected"
    map_cub = project_root / "2011" / "M111_map.cub"
    map_cub.parent.mkdir(parents=True, exist_ok=True)
    map_cub.write_bytes(b"EXISTING")
    out = tmp_path / "artifacts" / "project_manifest.json"
    cfg = LrocProjectConfig(
        download_manifest=download_manifest,
        project_root=project_root,
        output_json=out,
        overwrite=False,
    )

    def boom(_argv: list[str]) -> subprocess.CompletedProcess[str]:
        raise AssertionError("run_cmd should not be called on reuse")

    code, _path = lroc_project.run(cfg, which=_fake_which_all, run_cmd=boom)
    assert code == 0
    manifest = LrocProjectManifest.model_validate_json(out.read_text(encoding="utf-8"))
    assert manifest.passed is True
    assert str(map_cub) in manifest.assets_projected
    assert str(img) in manifest.assets_skipped
