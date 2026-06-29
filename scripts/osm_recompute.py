#!/usr/bin/env python3
"""Recompute OSM semantic-label masks for every location/year into the canonical
folder structure, including a combined class map per year.

For each rendered_<location>/ dir and each year_YYYY.tiff we read the grid
(size from the raster, bbox from the .tfw, CRS from the .prj — EPSG resolved via
pyproj, so ESRI-WKT .prj without an AUTHORITY tag still works), then run the OSM
pipeline for all categories and lay the outputs out as:

    rendered_<location>/osm/
        buildings/year_YYYY.tif
        roads/year_YYYY.tif
        paths/year_YYYY.tif
        green/year_YYYY.tif        # incl. multipolygon relations (large parks)
        water/year_YYYY.tif        # incl. multipolygon relations (lakes/reservoirs)
        manifests/osm_manifest_YYYY.json
        combined/year_YYYY.tif     # 0=bg 1=buildings 2=green 3=water 4=roads∪paths

Snapshot date defaults to mid-year (YYYY-07-01) — no acquisition-date manifest
is available. Relation support for water/green lives in overpass_client.py.

Idempotent: a (dir, year) whose combined/year_YYYY.tif already exists is skipped
unless --overwrite, so a run interrupted by Overpass rate-limiting just resumes.

Usage:
    python scripts/osm_recompute.py                      # poznan4 + all SIVL areas
    python scripts/osm_recompute.py <rendered_dir> ...   # explicit dirs
    python scripts/osm_recompute.py --overwrite --only sivl
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from satmap_dataset.config import OsmConfig
from satmap_dataset.models import LayerManifest
from satmap_dataset.pipeline import osm as osm_pipeline

# grid-reading helpers are shared with the simpler backfill driver.
from scripts.osm_backfill import _grid_from_tiff  # noqa: E402

SAT_DATA = Path("/home/maciej/Github/sat_data")
POZNAN4 = ["rendered_poznan", "rendered_przezmierowo", "rendered_sosnowiec", "rendered_zagan_zwirownia"]
CATEGORIES = ["buildings", "roads", "paths", "green", "water"]
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
# combined class codes; roads and paths both map to 4 (roads ∪ paths).
CLASS_CODE = {"buildings": 1, "green": 2, "water": 3, "roads": 4, "paths": 4}
GDAL_LETTER = {"buildings": "B", "green": "G", "water": "W", "roads": "R", "paths": "P"}


def _default_dirs() -> list[Path]:
    dirs = [SAT_DATA / n for n in POZNAN4]
    dirs += sorted((SAT_DATA / "sivl_maps").glob("rendered_sivl_area*"),
                   key=lambda p: int(re.search(r"\d+", p.name).group()))
    return dirs


def _build_combined(osm_dir: Path, year: int) -> tuple[bool, str]:
    """Run gdal_calc over the per-category masks that exist → combined/year.tif."""
    present = {cat: osm_dir / cat / f"year_{year}.tif"
               for cat in CATEGORIES if (osm_dir / cat / f"year_{year}.tif").exists()}
    out = osm_dir / "combined" / f"year_{year}.tif"
    if not present:
        return False, "no class masks (empty AOI)"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    letters = {GDAL_LETTER[c]: p for c, p in present.items()}

    def gt(letter: str) -> str:
        return f"({letter}>0)" if letter in letters else "0"

    def eq0(letter: str) -> str:
        return f"({letter}==0)" if letter in letters else "1"

    roads_or = "+".join(f"({l}>0)" for l in ("R", "P") if l in letters) or "0"
    roads_eq0 = "*".join(f"({l}==0)" for l in ("R", "P") if l in letters) or "1"
    # priority: buildings(1) > water(3) > roads/paths(4) > green(2)
    calc = (f"{gt('B')}*1 "
            f"+ {eq0('B')}*{gt('W')}*3 "
            f"+ {eq0('B')}*{eq0('W')}*(({roads_or})>0)*4 "
            f"+ {eq0('B')}*{eq0('W')}*{roads_eq0}*{gt('G')}*2")
    cmd = ["gdal_calc.py"]
    for letter, path in letters.items():
        cmd += [f"-{letter}", str(path)]
    cmd += ["--outfile", str(out), "--calc", calc,
            "--type", "Byte", "--co", "COMPRESS=DEFLATE", "--hideNoData", "--quiet"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return False, f"gdal_calc failed: {(proc.stderr or '')[-200:]}"
    return True, f"combined from {sorted(present)}"


def recompute_year(rendered: Path, year: int, *, overwrite: bool) -> dict:
    osm_dir = rendered / "osm"
    combined = osm_dir / "combined" / f"year_{year}.tif"
    if combined.exists() and not overwrite:
        return {"dir": rendered.name, "year": year, "status": "skip (combined exists)"}

    width, height, bbox, epsg = _grid_from_tiff(rendered / f"year_{year}.tiff")
    if epsg is None:
        return {"dir": rendered.name, "year": year, "status": "skip (no EPSG in .prj)"}
    bbox_str = ",".join(f"{v:.3f}" for v in bbox)

    staging = osm_dir / "_staging" / str(year)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    cfg = OsmConfig(
        bbox=bbox_str, target_bbox=bbox_str, target_width=width, target_height=height,
        srs=f"EPSG:{epsg}", categories=CATEGORIES,
        year_date_map={year: f"{year}-07-01"},
        osm_root=staging, output_json=staging / "osm_manifest.json",
        overpass_url=OVERPASS_URL, overwrite=True,
        sleep_min=1.0, sleep_max=2.5, retries=6, retry_delay=5.0, timeout=90.0,
    )
    exit_code, artifact = osm_pipeline.run(cfg)
    man = LayerManifest.load(artifact)
    counts = dict(man.years[0].feature_counts) if man.years else {}

    # move masks into per-category subfolders: year_YYYY_<cat>.tif -> <cat>/year_YYYY.tif
    moved = {}
    for cat in CATEGORIES:
        src = staging / f"year_{year}_{cat}.tif"
        if src.exists():
            dest = osm_dir / cat / f"year_{year}.tif"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            moved[cat] = str(dest)

    # rewrite manifest asset paths to the final layout, write to manifests/
    data = json.loads(artifact.read_text())
    data["assets"] = [moved[c] for c in CATEGORIES if c in moved]
    for y in data.get("years", []):
        if isinstance(y.get("assets"), dict):
            y["assets"] = {c: moved[c] for c in y["assets"] if c in moved}
    man_dir = osm_dir / "manifests"
    man_dir.mkdir(parents=True, exist_ok=True)
    (man_dir / f"osm_manifest_{year}.json").write_text(json.dumps(data, indent=2))

    shutil.rmtree(staging, ignore_errors=True)
    ok, note = _build_combined(osm_dir, year)

    return {"dir": rendered.name, "year": year, "grid": f"{width}x{height}",
            "epsg": epsg, "exit": exit_code, "counts": counts,
            "combined": ok, "note": note, "errors": man.errors[:1]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", type=Path, help="rendered_* dirs (default: poznan4 + SIVL)")
    ap.add_argument("--only", choices=["poznan4", "sivl"], help="restrict default set")
    ap.add_argument("--overwrite", action="store_true", help="recompute even if combined exists")
    args = ap.parse_args()

    if args.dirs:
        dirs = args.dirs
    else:
        dirs = _default_dirs()
        if args.only == "poznan4":
            dirs = [SAT_DATA / n for n in POZNAN4]
        elif args.only == "sivl":
            dirs = [d for d in dirs if "sivl" in d.name]

    results = []
    for d in dirs:
        if not d.exists():
            print(f"MISSING {d}")
            continue
        print(f"== {d.name} ==")
        for tiff in sorted(d.glob("year_*.tiff")):
            m = re.match(r"year_(\d{4})\.tiff$", tiff.name)
            if not m:
                continue
            r = recompute_year(d, int(m.group(1)), overwrite=args.overwrite)
            results.append(r)
            if "status" in r:
                print(f"  {r['year']}: {r['status']}")
            else:
                print(f"  {r['year']} [{r['grid']} EPSG:{r['epsg']}] exit={r['exit']} "
                      f"combined={r['combined']} counts={r['counts']}"
                      + (f" ERR={r['errors']}" if r['errors'] else ""))

    done = [r for r in results if r.get("combined")]
    failed = [r for r in results if r.get("exit") not in (0, None) or (r.get("exit") == 0 and not r.get("combined") and "status" not in r)]
    skipped = [r for r in results if "status" in r]
    print(f"\nDONE: {len(done)} combined built, {len(failed)} failed/partial, "
          f"{len(skipped)} skipped, {len(results)} total")
    for r in failed:
        print(f"  ISSUE {r['dir']}/{r['year']}: exit={r.get('exit')} note={r.get('note')} {r.get('errors')}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
