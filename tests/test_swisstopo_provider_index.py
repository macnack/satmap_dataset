from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.config import IndexConfig
from satmap_dataset.providers import get_provider
from satmap_dataset.providers.swisstopo import provider as provider_module

FIXTURE = ROOT / "tests" / "fixtures" / "swisstopo" / "wmts_capabilities_swissimage_product.xml"


def test_get_provider_returns_swisstopo() -> None:
    p = get_provider("swisstopo")
    assert p.name == "swisstopo"
    assert p.default_target_srs == "EPSG:2056"


def test_index_intersects_requested_years(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        provider_module,
        "_fetch_capabilities_xml",
        lambda *_a, **_k: FIXTURE.read_bytes(),
    )
    config = IndexConfig(
        year_start=1946,
        year_end=1955,
        bbox="2600000,1199000,2602000,1201000",
        srs="EPSG:2056",
        strict_years=False,
        min_years=5,
        output_json=tmp_path / "index_manifest.json",
        year_availability_output_json=tmp_path / "year_availability_report.json",
        provider="swisstopo",
    )
    code, path = get_provider("swisstopo").index(config)
    assert code == 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["provider"] == "swisstopo"
    assert payload["passed"] is True
    assert payload["years_included"] == list(range(1946, 1956))
    assert payload["years_excluded_with_reason"] == {}
    assert payload["provider_metadata"]["available_year_count"] == 99


def test_index_rejects_wrong_srs() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="EPSG:2056"):
        IndexConfig(
            year_start=2000,
            year_end=2001,
            bbox="2600000,1199000,2602000,1201000",
            srs="EPSG:4326",
            provider="swisstopo",
            output_json=Path("x.json"),
            year_availability_output_json=Path("y.json"),
        )
