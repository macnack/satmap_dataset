#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _slugify_location_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    lowered = ascii_only.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    slug = re.sub(r"_+", "_", slug)
    if not slug:
        raise ValueError(f"Cannot build slug from location_name={value!r}")
    return slug


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run 'satmap_dataset.cli index' sequentially for all location configs."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: autodetected from script location).",
    )
    parser.add_argument("--year-start", type=int, default=2015, help="First year (inclusive).")
    parser.add_argument("--year-end", type=int, default=2025, help="Last year (inclusive).")
    parser.add_argument(
        "--locations-dir",
        type=Path,
        default=None,
        help="Directory with location JSON files (default: configs/run/locations).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue indexing remaining locations when one location returns non-zero exit code.",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=None,
        help="Base config JSON to read default area_km2/square_km (default: configs/run/base.json).",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    base_config_path = (
        args.base_config.resolve()
        if args.base_config is not None
        else (repo_root / "configs" / "run" / "base.json")
    )
    base_cfg = _load_json(base_config_path)
    base_area_km2 = base_cfg.get("area_km2", base_cfg.get("square_km", 4.0))
    locations_dir = (
        args.locations_dir.resolve()
        if args.locations_dir is not None
        else (repo_root / "configs" / "run" / "locations")
    )
    if not locations_dir.exists():
        print(f"Locations directory not found: {locations_dir}", file=sys.stderr)
        return 2

    location_files = sorted(locations_dir.glob("*.json"))
    if not location_files:
        print(f"No location configs found in: {locations_dir}", file=sys.stderr)
        return 2

    failures: list[tuple[str, int]] = []

    for cfg_path in location_files:
        cfg = _load_json(cfg_path)
        name = cfg_path.stem

        location_name = cfg.get("location_name")
        center_lat = cfg.get("center_lat")
        center_lon = cfg.get("center_lon")
        area_km2 = cfg.get("area_km2", cfg.get("square_km", base_area_km2))
        if location_name is None or center_lat is None or center_lon is None:
            print(
                f"Skipping {cfg_path}: missing location_name, center_lat, or center_lon",
                file=sys.stderr,
            )
            return 2

        slug = _slugify_location_name(str(location_name))
        artifacts_dir = repo_root / f"artifacts_{slug}"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        output_json = artifacts_dir / "index_manifest.json"
        availability_json = artifacts_dir / "year_availability_report.json"

        cmd = [
            sys.executable,
            "-m",
            "satmap_dataset.cli",
            "index",
            "--year-start",
            str(args.year_start),
            "--year-end",
            str(args.year_end),
            "--center-lat",
            str(center_lat),
            "--center-lon",
            str(center_lon),
            "--square-km",
            str(area_km2),
            "--srs",
            "EPSG:2180",
            "--output-json",
            str(output_json),
            "--year-availability-output-json",
            str(availability_json),
        ]

        print(f"\n=== INDEX {name} ===")
        print(" ".join(cmd))
        rc = subprocess.call(cmd, cwd=str(repo_root))
        if rc != 0:
            print(f"Index failed for {name} (exit={rc})", file=sys.stderr)
            failures.append((name, rc))
            if not args.continue_on_error:
                return rc

    if failures:
        print("\nDONE with failures:")
        for name, rc in failures:
            print(f"- {name}: exit={rc}")
        return 1

    print("\nDONE: indexed all locations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
