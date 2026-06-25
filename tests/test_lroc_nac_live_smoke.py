from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

pytestmark = pytest.mark.skipif(
    os.environ.get("SATMAP_LIVE_TESTS") != "1",
    reason="set SATMAP_LIVE_TESTS=1 to hit the live ODE API",
)


def test_live_ode_returns_multitemporal_nac(tmp_path) -> None:
    from satmap_dataset.config import IndexConfig
    from satmap_dataset.providers.lroc_nac import LrocNacProvider

    cfg = IndexConfig(
        year_start=2009, year_end=2026,
        bbox="30.60,20.00,30.90,20.35", srs="IAU_2015:30100",
        provider="lroc_nac", min_years=2,
        provider_options={"product_type": "CDRNAC4", "max_pages": 3},
        output_json=tmp_path / "index.json",
        year_availability_output_json=tmp_path / "avail.json",
    )
    code, path = LrocNacProvider().index(cfg)
    assert code == 0, "expected ≥2 distinct NAC years over Apollo 17"
