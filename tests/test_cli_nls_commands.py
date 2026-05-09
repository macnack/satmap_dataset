from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from typer.testing import CliRunner

from satmap_dataset.cli import app
from satmap_dataset.providers.nls import provider as nls_provider

runner = CliRunner()


def _fixture_xml() -> bytes:
    return (
        Path(__file__).parent / "fixtures" / "nls" / "describe_coverage_ortokuva_vari.xml"
    ).read_bytes()


def test_nls_index_json_invokes_provider(monkeypatch, tmp_path):
    monkeypatch.setattr(nls_provider, "_fetch_describe_coverage_xml", lambda **kw: _fixture_xml())
    cfg = {
        "year_start": 2018,
        "year_end": 2022,
        "bbox": "385000,6675000,387000,6677000",
        "srs": "EPSG:3067",
        "provider": "nls",
        "provider_options": {"api_key": "test-key"},
        "output_json": str(tmp_path / "index_manifest.json"),
        "year_availability_output_json": str(tmp_path / "year_availability_report.json"),
    }
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")
    result = runner.invoke(app, ["nls-index-json", str(config_path)])
    assert result.exit_code == 0, result.output
    data = json.loads((tmp_path / "index_manifest.json").read_text(encoding="utf-8"))
    assert data["provider"] == "nls"
    assert data["years_included"]


def test_nls_index_json_validation_error_exits_2(tmp_path):
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps({"year_start": 2030, "year_end": 2020, "bbox": "0,0,1,1"}))
    result = runner.invoke(app, ["nls-index-json", str(config_path)])
    assert result.exit_code == 2
