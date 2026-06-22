"""Build a held-out TEST split manifest from ingested raw-tile cells.

Ported from sat_roma `raw_tile_pipeline/build_test_manifest.py`. Scans
`<root>/<provider>/<area>/<cellkey>/year_YYYY.tif`, picks each area's richest
equal-dimension (single-GSD) season group, materialises a homogeneous symlink
location dir, and emits a per-location `roots:` split manifest (query = newest
year, ref = older years). Equal-dimension grouping is required because the
current dataset samples co-registered (equal-size) season stacks.
"""
from __future__ import annotations

import collections
import re
from pathlib import Path

import pyvips
import yaml

_YEAR = re.compile(r"year_(\d{4})")


def _dims(tif: Path) -> tuple:
    im = pyvips.Image.new_from_file(str(tif))
    return (im.width, im.height)


def _best_cell(area: Path):
    """(loc_dir_name, cellkey, sorted_years, dims) for the area's richest
    equal-dimension season group, or None."""
    best = None
    for cell in sorted(area.glob("e*_n*")):
        if not cell.is_dir():
            continue
        by_dims = collections.defaultdict(list)
        for f in sorted(cell.glob("year_*.tif")):
            by_dims[_dims(f)].append(int(_YEAR.search(f.name).group(1)))
        for dims, years in by_dims.items():
            if best is None or len(years) > len(best[2]):
                best = (cell.name, cell.name, sorted(years), dims)
    return best


def build_test_manifest(root: Path, out: Path, min_years: int = 2) -> dict:
    """Scan ``root`` for ingested cells and write a TEST split manifest to ``out``.

    Returns the manifest dict (also written as YAML to ``out``).
    """
    root = Path(root)
    out = Path(out)
    loc_root = (root / "_manifest_locs").resolve()
    loc_root.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"roots": {"locs": str(loc_root)}}

    for area in sorted(root.glob("*/*")):
        if not area.is_dir() or area.name == "_manifest_locs":
            continue
        pick = _best_cell(area)
        if pick is None or len(pick[2]) < min_years:
            continue
        _, cellkey, years, _dims_ = pick
        provider, area_name = area.parent.name, area.name
        loc_name = f"{provider}_{area_name}_{cellkey}"
        d = loc_root / loc_name
        d.mkdir(exist_ok=True)
        for y in years:
            link = d / f"year_{y}.tif"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to((area / cellkey / f"year_{y}.tif").resolve())
        manifest[loc_name] = {
            "root": "locs",
            "test": {"query": years[-1], "ref": years[:-1][::-1]},
        }

    out.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return manifest
