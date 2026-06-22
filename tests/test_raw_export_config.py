from pathlib import Path

import pytest
from pydantic import ValidationError

from satmap_dataset.config import RawExportConfig


def _base(**over):
    payload = {"provider": "geoportal", "area": "poznan", "download_root": "downloads_poznan"}
    payload.update(over)
    return payload


def test_defaults_resolve():
    cfg = RawExportConfig(**_base())
    assert cfg.provider == "geoportal"
    assert cfg.area == "poznan"
    assert cfg.link_mode == "symlink"
    assert cfg.min_coverage is None
    assert cfg.cell_size_m is None
    assert str(cfg.raw_root)  # non-empty default


def test_sentinel2_rejected():
    with pytest.raises(ValidationError):
        RawExportConfig(**_base(provider="sentinel2"))


def test_unknown_provider_rejected():
    with pytest.raises(ValidationError):
        RawExportConfig(**_base(provider="bogus"))


def test_link_mode_validated():
    assert RawExportConfig(**_base(link_mode="copy")).link_mode == "copy"
    with pytest.raises(ValidationError):
        RawExportConfig(**_base(link_mode="hardlink"))


def test_min_coverage_bounds():
    assert RawExportConfig(**_base(min_coverage=0.5)).min_coverage == 0.5
    with pytest.raises(ValidationError):
        RawExportConfig(**_base(min_coverage=0.0))
    with pytest.raises(ValidationError):
        RawExportConfig(**_base(min_coverage=1.5))


def test_cell_size_positive():
    assert RawExportConfig(**_base(cell_size_m=2500.0)).cell_size_m == 2500.0
    with pytest.raises(ValidationError):
        RawExportConfig(**_base(cell_size_m=0.0))


def test_raw_root_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SATMAP_RAW_ROOT", str(tmp_path / "custom_raw"))
    cfg = RawExportConfig(**_base())
    assert Path(cfg.raw_root) == tmp_path / "custom_raw"
