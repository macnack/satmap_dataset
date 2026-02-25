#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


KINDS = ("downloads", "rendered", "artifacts")


@dataclass(frozen=True)
class RootEntry:
    location_file: Path
    location_name: str
    kind: str
    path: Path


def _slugify_location_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    lowered = ascii_only.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    slug = re.sub(r"_+", "_", slug)
    if not slug:
        raise ValueError(f"Cannot build slug from location_name={value!r}")
    return slug


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_location_payload(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Location file must contain a JSON object: {path}")
    return raw


def _path_for_kind(payload: dict[str, object], kind: str, repo_root: Path) -> Path:
    mapping = {
        "downloads": "download_root",
        "rendered": "render_root",
        "artifacts": "artifacts_dir",
    }
    key = mapping[kind]
    value = payload.get(key)
    if value:
        return Path(str(value)).expanduser().resolve()
    location_name = payload.get("location_name")
    if not location_name:
        raise ValueError("Missing both explicit path and location_name in location JSON")
    slug = _slugify_location_name(str(location_name))
    return (repo_root / f"{kind}_{slug}").resolve()


def collect_entries(locations_dir: Path, kinds: Iterable[str]) -> list[RootEntry]:
    repo_root = _resolve_repo_root()
    entries: list[RootEntry] = []
    for location_file in sorted(locations_dir.glob("location_*.json")):
        payload = _load_location_payload(location_file)
        location_name = str(payload.get("location_name") or location_file.stem)
        for kind in kinds:
            root = _path_for_kind(payload, kind, repo_root)
            entries.append(
                RootEntry(
                    location_file=location_file,
                    location_name=location_name,
                    kind=kind,
                    path=root,
                )
            )
    return entries


def _print_list(entries: list[RootEntry], include_missing: bool) -> int:
    filtered = [entry for entry in entries if include_missing or entry.path.exists()]
    print(f"Roots summary: {len(filtered)} entries")
    for entry in filtered:
        status = "exists" if entry.path.exists() else "missing"
        rel_file = entry.location_file.name
        print(f"{entry.kind:10} {status:7} {rel_file:16} {entry.path}")
    return 0


def _move_entries(entries: list[RootEntry], target_dir: Path, execute: bool) -> int:
    target_dir = target_dir.expanduser().resolve()
    print(f"Move target: {target_dir}")
    print("Mode: execute" if execute else "Mode: dry-run")
    if execute:
        target_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    skipped = 0
    for entry in entries:
        source = entry.path
        if not source.exists():
            skipped += 1
            print(f"skip missing: {source}")
            continue
        destination = target_dir / source.name
        if destination.exists():
            raise FileExistsError(f"Destination already exists: {destination}")
        print(f"move {source} -> {destination}")
        if execute:
            shutil.move(str(source), str(destination))
            moved += 1
    print(f"done: moved={moved} skipped_missing={skipped}")
    return 0


def _delete_entries(entries: list[RootEntry], execute: bool) -> int:
    print("Mode: execute" if execute else "Mode: dry-run")
    deleted = 0
    skipped = 0
    for entry in entries:
        target = entry.path
        if not target.exists():
            skipped += 1
            print(f"skip missing: {target}")
            continue
        if not target.is_dir():
            raise NotADirectoryError(f"Refusing to delete non-directory path: {target}")
        print(f"delete {target}")
        if execute:
            shutil.rmtree(target)
            deleted += 1
    print(f"done: deleted={deleted} skipped_missing={skipped}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage downloads/rendered/artifacts dirs for location JSON sets.")
    parser.add_argument("action", choices=("list", "move", "delete"))
    parser.add_argument("--locations-dir", required=True, type=Path, help="Directory with location_*.json files.")
    parser.add_argument(
        "--kind",
        default="all",
        choices=("all", *KINDS),
        help="Which root kind to target.",
    )
    parser.add_argument("--include-missing", action="store_true", help="Show missing paths in list action.")
    parser.add_argument("--target-dir", type=Path, help="Destination parent dir for move action.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform move/delete. Without this flag the script runs in dry-run mode.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    locations_dir = args.locations_dir.expanduser().resolve()
    if not locations_dir.is_dir():
        raise FileNotFoundError(f"locations dir not found: {locations_dir}")

    kinds = KINDS if args.kind == "all" else (args.kind,)
    entries = collect_entries(locations_dir, kinds)
    if not entries:
        print(f"No location_*.json files found in: {locations_dir}")
        return 0

    if args.action == "list":
        return _print_list(entries, include_missing=bool(args.include_missing))
    if args.action == "move":
        if args.target_dir is None:
            raise ValueError("--target-dir is required for move action")
        return _move_entries(entries, target_dir=args.target_dir, execute=bool(args.execute))
    if args.action == "delete":
        return _delete_entries(entries, execute=bool(args.execute))

    raise ValueError(f"Unsupported action: {args.action}")


if __name__ == "__main__":
    raise SystemExit(main())
