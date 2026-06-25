from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.config import DownloadConfig
from satmap_dataset.models import IndexManifest
from satmap_dataset.providers.lroc_nac import provider as lroc_provider


def _write_index(path: Path) -> None:
    manifest = IndexManifest(
        provider="lroc_nac", year_start=2009, year_end=2012,
        bbox="30.6,20.0,30.9,20.35", srs="IAU_2015:30100",
        strict_years=False, min_years=1, wfs_bbox_axes_swapped=False,
        years_requested=[2009, 2010, 2011, 2012], year_statuses=[],
        years_available_wfs=[2009, 2012], years_included=[2009, 2012],
        years_excluded_with_reason={}, common_tile_ids=[],
        tile_sources_by_year={
            2009: {"M101013931LC": "https://pds.example/a.IMG"},
            2012: {"M198273648LC": "https://pds.example/b.IMG"},
        },
        tile_bboxes_by_year={}, tile_acquisition_by_year={},
        passed=True, errors=[], warnings=[], run_parameters={}, provider_metadata={},
    )
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def test_download_writes_assets_and_manifest(tmp_path, monkeypatch) -> None:
    index_path = tmp_path / "index.json"
    _write_index(index_path)

    async def fake_fetch(client, url, output_path, **kwargs):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"FAKE")
        return True

    monkeypatch.setattr(lroc_provider, "_download_asset_with_retry", fake_fetch)

    cfg = DownloadConfig(
        index_manifest=index_path, download_root=tmp_path / "dl",
        srs="IAU_2015:30100", provider="lroc_nac", bbox="30.6,20.0,30.9,20.35",
        wms_fallback_missing_years=False, output_json=tmp_path / "out.json",
    )
    code, path = lroc_provider.LrocNacProvider().download(cfg)

    assert code == 0
    assert (tmp_path / "dl" / "2009" / "M101013931LC.IMG").read_bytes() == b"FAKE"
    assert (tmp_path / "dl" / "2012" / "M198273648LC.IMG").exists()
    manifest = json.loads(path.read_text())
    assert manifest["provider"] == "lroc_nac"
    assert sorted(manifest["years_included"]) == [2009, 2012]


def test_download_path_traversal_safe(tmp_path, monkeypatch) -> None:
    """pdsid values with path-traversal characters must not escape download_root."""
    index_path = tmp_path / "index.json"
    malicious_pdsid = "../../evil"
    manifest = IndexManifest(
        provider="lroc_nac", year_start=2020, year_end=2020,
        bbox="30.6,20.0,30.9,20.35", srs="IAU_2015:30100",
        strict_years=False, min_years=1, wfs_bbox_axes_swapped=False,
        years_requested=[2020], year_statuses=[],
        years_available_wfs=[2020], years_included=[2020],
        years_excluded_with_reason={}, common_tile_ids=[],
        tile_sources_by_year={2020: {malicious_pdsid: "https://pds.example/evil.IMG"}},
        tile_bboxes_by_year={}, tile_acquisition_by_year={},
        passed=True, errors=[], warnings=[], run_parameters={}, provider_metadata={},
    )
    index_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    written_paths: list[Path] = []

    async def fake_fetch(client, url, output_path, **kwargs):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"FAKE")
        written_paths.append(output_path)
        return True

    monkeypatch.setattr(lroc_provider, "_download_asset_with_retry", fake_fetch)

    download_root = tmp_path / "dl"
    cfg = DownloadConfig(
        index_manifest=index_path, download_root=download_root,
        srs="IAU_2015:30100", provider="lroc_nac", bbox="30.6,20.0,30.9,20.35",
        wms_fallback_missing_years=False, output_json=tmp_path / "out.json",
    )
    code, _ = lroc_provider.LrocNacProvider().download(cfg)

    assert code == 0
    assert written_paths, "expected at least one file to be written"
    resolved_root = download_root.resolve()
    for p in written_paths:
        resolved_p = p.resolve()
        # The written file must be inside download_root — no traversal escape.
        assert resolved_p.is_relative_to(resolved_root), (
            f"Path {resolved_p} escaped download_root {resolved_root}"
        )
        # Confirm no raw ".." component remains in the filename portion.
        assert ".." not in resolved_p.parts
