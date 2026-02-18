from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.config import RunConfig
from satmap_dataset.models import IndexManifest, YearStatus
from satmap_dataset.pipeline import run_all


def test_can_reuse_index_rejects_manifest_with_swapped_tile_bboxes() -> None:
    config = RunConfig(
        year_start=2017,
        year_end=2017,
        bbox="348760.243,508296.603,350174.457,509710.817",
        srs="EPSG:2180",
    )
    manifest = IndexManifest(
        year_start=2017,
        year_end=2017,
        bbox="348760.243,508296.603,350174.457,509710.817",
        srs="EPSG:2180",
        strict_years=False,
        min_years=1,
        wfs_bbox_axes_swapped=False,
        years_requested=[2017],
        year_statuses=[YearStatus(year=2017, typename_exists=True, feature_count=1, status="has_features")],
        years_available_wfs=[2017],
        years_included=[2017],
        years_excluded_with_reason={},
        common_tile_ids=["N-33-130-D-a-3-3"],
        tile_sources_by_year={2017: {"N-33-130-D-a-3-3": "https://example.com/a.tif"}},
        # Swapped order from legacy bug: ymin,xmin,ymax,xmax
        tile_bboxes_by_year={2017: {"N-33-130-D-a-3-3": [507954.8, 347030.43, 510336.71, 349225.86]}},
        passed=True,
        errors=[],
        warnings=[],
        run_parameters={},
    )

    assert run_all._index_manifest_has_swapped_tile_bboxes(manifest) is True
    assert run_all._can_reuse_index(manifest, config) is False
