#!/usr/bin/env python3
"""Flatten world_window raw-export cells into sat_data `*_gmix` layout.

satmap_dataset's `raw-export --cell-mode world_window` writes co-registered,
equal-dim season stacks to:

    <raw_root>/<provider>/<area>/<cellkey>/year_YYYY.{tif,tfw,prj}

sat_roma's training data root (`~/Github/sat_data`) consumes flat cell dirs:

    <dest>/<provider>_<area>_<cellkey>_gmix/year_YYYY.{tif,tfw,prj}

This script materialises the second layout from the first (copy by default, or
symlink), matching the existing `geoportal_poznan_15km2_*_gmix` cells.

Example:
    python scripts/flatten_gmix.py \
        --provider geoportal --area wroclaw_15km2
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import unicodedata
from pathlib import Path

CELLKEY_RE = re.compile(r"^e\d+_n\d+$")
YEAR_SUFFIXES = (".tif", ".tfw", ".prj")


def _default_raw_root() -> Path:
    env = os.environ.get("SATMAP_RAW_ROOT")
    return Path(env).expanduser() if env else Path("~/Github/sat_data_raw").expanduser()


def _slugify(value: str) -> str:
    s = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()
    return re.sub(r"_+", "_", s)


def find_cells(area_dir: Path) -> list[Path]:
    """World_window cell dirs are named e<easting>_n<northing> and hold year_*.tif."""
    cells = []
    for sub in sorted(area_dir.iterdir()):
        if sub.is_dir() and CELLKEY_RE.match(sub.name) and any(sub.glob("year_*.tif")):
            cells.append(sub)
    return cells


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--location-json", type=Path, default=None,
                   help="Location JSON; derives --provider (default geoportal) and --area (slug of location_name).")
    p.add_argument("--provider", default=None)
    p.add_argument("--area", default=None, help="Area slug, e.g. wroclaw_15km2")
    p.add_argument("--raw-root", type=Path, default=_default_raw_root(),
                   help="world_window export root (default: $SATMAP_RAW_ROOT or ~/Github/sat_data_raw)")
    p.add_argument("--dest", type=Path, default=Path("~/Github/sat_data").expanduser(),
                   help="sat_data root that receives <provider>_<area>_<cellkey>_gmix dirs")
    p.add_argument("--link-mode", choices=("copy", "symlink"), default="copy")
    p.add_argument("--overwrite", action="store_true", help="replace existing gmix cell dirs")
    args = p.parse_args()

    if args.location_json is not None:
        loc = json.loads(args.location_json.read_text(encoding="utf-8"))
        if args.provider is None:
            args.provider = loc.get("provider", "geoportal")
        if args.area is None:
            args.area = _slugify(str(loc["location_name"]))
    if args.provider is None:
        args.provider = "geoportal"
    if args.area is None:
        print("error: provide --area or --location-json", file=sys.stderr)
        return 2

    area_dir = args.raw_root / args.provider / args.area
    if not area_dir.is_dir():
        print(f"error: export area not found: {area_dir}", file=sys.stderr)
        print("       run raw-export with --cell-mode world_window first.", file=sys.stderr)
        return 1

    cells = find_cells(area_dir)
    if not cells:
        print(f"error: no world_window cells (e<e>_n<n>/ with year_*.tif) under {area_dir}", file=sys.stderr)
        return 1

    args.dest.mkdir(parents=True, exist_ok=True)
    written = 0
    for cell in cells:
        out = args.dest / f"{args.provider}_{args.area}_{cell.name}_gmix"
        if out.exists():
            if not args.overwrite:
                print(f"skip (exists): {out.name}")
                continue
            shutil.rmtree(out)
        out.mkdir(parents=True)
        for src in sorted(cell.iterdir()):
            if src.suffix.lower() in YEAR_SUFFIXES and src.name.startswith("year_"):
                dst = out / src.name
                if args.link_mode == "symlink":
                    dst.symlink_to(src.resolve())
                else:
                    shutil.copy2(src, dst)
        n = len(list(out.glob("year_*.tif")))
        print(f"{'linked' if args.link_mode == 'symlink' else 'copied'}: {out.name}  ({n} years)")
        written += 1

    print(f"\ndone: {written} gmix cell(s) -> {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
