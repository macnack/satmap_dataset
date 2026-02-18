#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _merge(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    merged = dict(base)
    merged.update(override)
    return merged


def _slugify_location_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    lowered = ascii_only.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    slug = re.sub(r"_+", "_", slug)
    if not slug:
        raise ValueError(f"Cannot build slug from location_name={value!r}")
    return slug


def _apply_location_paths_policy(merged: dict[str, object], repo_root: Path) -> None:
    location_name = merged.get("location_name")
    if location_name is None:
        return
    slug = _slugify_location_name(str(location_name))
    merged.setdefault("download_root", str(repo_root / f"downloads_{slug}"))
    merged.setdefault("render_root", str(repo_root / f"rendered_{slug}"))
    merged.setdefault("artifacts_dir", str(repo_root / f"artifacts_{slug}"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge base + override JSON config (shallow merge).")
    parser.add_argument("--base", type=Path, required=True, help="Path to base JSON config.")
    parser.add_argument("--override", type=Path, required=True, help="Path to override JSON config.")
    parser.add_argument("--out", type=Path, required=True, help="Path to output merged JSON config.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repo root used to generate default paths from location_name.",
    )
    args = parser.parse_args()

    base = _read_json(args.base)
    override = _read_json(args.override)
    merged = _merge(base, override)
    _apply_location_paths_policy(merged, args.repo_root.resolve())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
