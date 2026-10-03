from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.config import DownloadConfig, RunConfig


def test_run_config_normalizes_hybrid_to_stac_for_lantmateriet() -> None:
    cfg = RunConfig(
        year_start=2020,
        year_end=2021,
        bbox="536194.91,6426213.28,538194.91,6428213.28",
        srs="EPSG:3006",
        target_srs="EPSG:3006",
        provider="lantmateriet",
        mode="hybrid",
    )
    assert cfg.mode == "stac"


def test_download_config_accepts_stac_for_lantmateriet() -> None:
    cfg = DownloadConfig(
        index_manifest=Path("artifacts/index_manifest.json"),
        provider="lantmateriet",
        mode="stac",
        bbox="536194.91,6426213.28,538194.91,6428213.28",
        srs="EPSG:3006",
    )
    assert cfg.mode == "stac"
