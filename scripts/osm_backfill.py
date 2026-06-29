#!/usr/bin/env python3
"""Backfill OSM semantic-label masks aligned 1:1 to rendered orthophoto grids.

For each rendered_<location>/ dir we read every year_YYYY.tiff's grid (size from
the raster, bbox from the .tfw, CRS from the .prj) and run the OSM pipeline stage
per-year, burning category masks (year_YYYY_<category>.tif) into the SAME dir so
they sit next to the ortho TIFFs. Snapshot date defaults to mid-year (YYYY-07-01)
since no acquisition-date manifest is available.

Idempotent: skips a (dir, year) whose masks already exist (overwrite=False), so a
run interrupted by Overpass rate-limiting can simply be re-run to resume.

Usage:
    python scripts/osm_backfill.py <rendered_dir> [<rendered_dir> ...] \
        [--categories buildings,roads,green,water]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pyvips

from satmap_dataset.config import OsmConfig
from satmap_dataset.pipeline import osm as osm_pipeline

_EPSG_RE = re.compile(r'AUTHORITY\["EPSG","(\d+)"\]\s*\]\s*$')


def _read_tfw(tfw_path: Path) -> tuple[float, float, float, float, float, float]:
    vals = [float(line.strip()) for line in tfw_path.read_text().splitlines() if line.strip()]
    a, d, b, e, c, f = vals[:6]
    return a, d, b, e, c, f


def _epsg_from_prj(prj_path: Path) -> int | None:
    text = prj_path.read_text()
    matches = re.findall(r'AUTHORITY\["EPSG","(\d+)"\]', text)
    if matches:
        return int(matches[-1])
    # ESRI-style .prj (e.g. EUREF_FIN_TM35FIN) has no AUTHORITY tag; let pyproj
    # identify the EPSG code from the projection parameters instead.
    try:
        import pyproj
        return pyproj.CRS.from_wkt(text).to_epsg(min_confidence=20)
    except Exception:  # noqa: BLE001
        return None


def _grid_from_tiff(tiff: Path) -> tuple[int, int, tuple[float, float, float, float], int | None]:
    im = pyvips.Image.new_from_file(str(tiff), access="sequential")
    width, height = im.width, im.height
    a, d, b, e, c, f = _read_tfw(tiff.with_suffix(".tfw"))
    # c,f = center of upper-left pixel; convert to outer bbox corners.
    xmin = c - a / 2.0
    ymax = f - e / 2.0  # e is negative
    xmax = xmin + width * a
    ymin = ymax + height * e
    epsg = _epsg_from_prj(tiff.with_suffix(".prj"))
    return width, height, (xmin, ymin, xmax, ymax), epsg


def backfill_dir(rendered: Path, categories: list[str]) -> list[dict]:
    results: list[dict] = []
    tiffs = sorted(rendered.glob("year_*.tiff"))
    for tiff in tiffs:
        m = re.match(r"year_(\d{4})\.tiff$", tiff.name)
        if not m:
            continue
        year = int(m.group(1))
        width, height, bbox, epsg = _grid_from_tiff(tiff)
        if epsg is None:
            results.append({"dir": rendered.name, "year": year, "skipped": "no EPSG in .prj"})
            print(f"  SKIP {rendered.name}/{year}: could not read EPSG from .prj")
            continue
        bbox_str = ",".join(f"{v:.3f}" for v in bbox)
        cfg = OsmConfig(
            bbox=bbox_str,
            target_bbox=bbox_str,
            target_width=width,
            target_height=height,
            srs=f"EPSG:{epsg}",
            categories=categories,
            year_date_map={year: f"{year}-07-01"},
            osm_root=rendered,
            output_json=rendered / f"osm_manifest_{year}.json",
            overpass_url="https://overpass-api.de/api/interpreter",
            overwrite=False,
            sleep_min=1.0,
            sleep_max=2.5,
            retries=6,
            retry_delay=5.0,
            timeout=90.0,
        )
        exit_code, artifact = osm_pipeline.run(cfg)
        from satmap_dataset.models import LayerManifest
        man = LayerManifest.load(artifact)
        counts = man.years[0].feature_counts if man.years else {}
        results.append({
            "dir": rendered.name, "year": year, "grid": f"{width}x{height}",
            "exit": exit_code, "counts": dict(counts),
            "errors": man.errors[:2],
        })
        print(f"  {rendered.name}/{year} [{width}x{height}] exit={exit_code} counts={dict(counts)}"
              + (f" ERRORS={man.errors[:1]}" if man.errors else ""))
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--categories", default="buildings,roads,green,water")
    args = ap.parse_args()
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]

    all_results: list[dict] = []
    for d in args.dirs:
        if not d.exists():
            print(f"MISSING dir: {d}")
            continue
        print(f"== {d} ==")
        all_results.extend(backfill_dir(d, cats))

    ok = sum(1 for r in all_results if r.get("exit") == 0)
    failed = [r for r in all_results if r.get("exit") not in (0, None)]
    skipped = [r for r in all_results if r.get("skipped")]
    print(f"\nDONE: {ok} ok, {len(failed)} failed, {len(skipped)} skipped, {len(all_results)} total")
    for r in failed:
        print(f"  FAIL {r['dir']}/{r['year']}: {r.get('errors')}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
