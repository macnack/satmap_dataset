"""Collapse Wayback releases into distinct imagery versions and bucket by capture year.

Most Wayback releases re-serve the same pixels at any given place. Esri's
``tilemap`` endpoint answers "which release do these pixels come from" per
tile (``select``), which is how the Wayback app computes *local changes*. We
walk releases newest → oldest, jumping straight past every release that
re-serves an older one, so the request count scales with the number of
distinct versions rather than the ~200 releases.

Pure functions here take already-fetched data so they are unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Awaitable, Callable, Iterable, Sequence

from satmap_dataset.providers.esri_wayback.catalog import WaybackRelease


@dataclass(frozen=True)
class TilemapResult:
    """Parsed ``tilemap`` response for one release/tile."""

    has_data: bool
    select_release: int | None = None

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "TilemapResult":
        data = payload.get("data") or [0]
        select = payload.get("select") or []
        return cls(
            has_data=bool(data and int(data[0]) == 1),
            select_release=int(select[0]) if select else None,
        )


TilemapFetcher = Callable[[WaybackRelease, tuple[int, int, int]], Awaitable[TilemapResult]]


async def walk_local_changes(
    releases: Sequence[WaybackRelease],
    tile: tuple[int, int, int],
    fetch: TilemapFetcher,
    *,
    max_requests: int,
) -> tuple[list[int], int]:
    """Return release numbers whose pixels differ at ``tile`` (newest first) and request count.

    ``releases`` must be sorted newest first. A release whose ``select`` points
    at an older release re-serves that release's pixels; the walk jumps to the
    first release older than the origin.
    """
    order = {r.release_num: i for i, r in enumerate(releases)}
    origins: list[int] = []
    idx = 0
    requests = 0
    while idx < len(releases) and requests < max_requests:
        release = releases[idx]
        result = await fetch(release, tile)
        requests += 1
        if not result.has_data:
            idx += 1
            continue
        origin = result.select_release or release.release_num
        if origin not in origins:
            origins.append(origin)
        origin_idx = order.get(origin)
        if origin_idx is None:
            # Origin is outside the filtered window (older than it): nothing
            # older inside the window can differ from it at this tile.
            break
        idx = max(idx, origin_idx) + 1
    return origins, requests


def group_by_content_hash(
    releases: Sequence[WaybackRelease],
    observations: dict[int, tuple[str, int | None]],
) -> list[int]:
    """Distinct versions from per-release ``(sha256, redirect_origin)`` of one probe tile.

    Releases with identical tile bytes collapse to the oldest release carrying
    those bytes (or the redirect origin when the server revealed it).
    """
    by_hash: dict[str, list[WaybackRelease]] = {}
    origin_by_hash: dict[str, int] = {}
    for release in releases:
        obs = observations.get(release.release_num)
        if obs is None:
            continue
        digest, origin = obs
        by_hash.setdefault(digest, []).append(release)
        if origin is not None:
            origin_by_hash.setdefault(digest, origin)
    out: list[int] = []
    for digest, group in by_hash.items():
        oldest = min(group, key=lambda r: r.sort_key)
        out.append(origin_by_hash.get(digest, oldest.release_num))
    return out


@dataclass
class CaptureInfo:
    capture_date: str | None = None  # ISO YYYY-MM-DD at AOI center
    source: str | None = None  # SRC_DESC
    provider_name: str | None = None  # NICE_DESC
    resolution_m: float | None = None  # SRC_RES
    accuracy_m: float | None = None  # SRC_ACC (99999 = unknown)
    captures_in_aoi: list[dict[str, Any]] = field(default_factory=list)

    @property
    def signature(self) -> tuple[tuple[Any, ...], ...] | None:
        rows = self.captures_in_aoi or (
            [{"capture_date": self.capture_date, "source": self.source, "resolution_m": self.resolution_m}]
            if self.capture_date
            else []
        )
        if not rows:
            return None
        return tuple(sorted((r.get("capture_date"), r.get("source"), r.get("resolution_m")) for r in rows))


def parse_src_date(value: Any) -> str | None:
    """``SRC_DATE`` is an int ``YYYYMMDD``; return ISO or None for junk."""
    if value in (None, "", 0):
        return None
    text = str(int(value)) if isinstance(value, (int, float)) else str(value).strip()
    if len(text) != 8 or not text.isdigit():
        return None
    try:
        return date(int(text[:4]), int(text[4:6]), int(text[6:8])).isoformat()
    except ValueError:
        return None


def capture_from_features(
    point_features: Iterable[dict[str, Any]],
    envelope_features: Iterable[dict[str, Any]] = (),
) -> CaptureInfo:
    info = CaptureInfo()
    for feature in point_features:
        attrs = feature.get("attributes") or {}
        iso = parse_src_date(attrs.get("SRC_DATE"))
        if iso is None:
            continue
        info.capture_date = iso
        info.source = attrs.get("SRC_DESC")
        info.provider_name = attrs.get("NICE_DESC")
        info.resolution_m = _float_or_none(attrs.get("SRC_RES"))
        acc = _float_or_none(attrs.get("SRC_ACC"))
        info.accuracy_m = None if acc is not None and acc >= 99999 else acc
        break
    seen: set[tuple[Any, ...]] = set()
    for feature in envelope_features:
        attrs = feature.get("attributes") or {}
        row = {
            "capture_date": parse_src_date(attrs.get("SRC_DATE")),
            "source": attrs.get("SRC_DESC"),
            "resolution_m": _float_or_none(attrs.get("SRC_RES")),
        }
        key = (row["capture_date"], row["source"], row["resolution_m"])
        if row["capture_date"] and key not in seen:
            seen.add(key)
            info.captures_in_aoi.append(row)
    info.captures_in_aoi.sort(key=lambda r: (r["capture_date"] or "", r["source"] or ""))
    return info


def _float_or_none(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class ImageryVersion:
    release: WaybackRelease
    capture: CaptureInfo = field(default_factory=CaptureInfo)
    represented_release_nums: list[int] = field(default_factory=list)
    collapsed_release_nums: list[int] = field(default_factory=list)

    @property
    def capture_date_source(self) -> str:
        return "metadata" if self.capture.capture_date else "release_date"

    @property
    def effective_capture_date(self) -> str:
        return self.capture.capture_date or self.release.release_date

    @property
    def capture_year(self) -> int:
        return int(self.effective_capture_date[:4])

    @property
    def deduped(self) -> bool:
        return len(self.represented_release_nums) > 1 or bool(self.collapsed_release_nums)

    def as_dict(self) -> dict[str, Any]:
        return {
            "release_num": self.release.release_num,
            "layer_identifier": self.release.identifier,
            "release_date": self.release.release_date,
            "capture_date": self.capture.capture_date,
            "capture_date_source": self.capture_date_source,
            "capture_year": self.capture_year,
            "source": self.capture.source,
            "provider_name": self.capture.provider_name,
            "resolution_m": self.capture.resolution_m,
            "accuracy_m": self.capture.accuracy_m,
            "captures_in_aoi": list(self.capture.captures_in_aoi),
            "deduped": self.deduped,
            "represented_release_nums": list(self.represented_release_nums),
            "collapsed_release_nums": list(self.collapsed_release_nums),
        }


def assign_represented_releases(
    version_release_nums: Iterable[int],
    releases: Sequence[WaybackRelease],
    all_releases: dict[int, WaybackRelease],
) -> list[ImageryVersion]:
    """Build versions (newest first); each filtered release maps to the newest version not newer than it."""
    versions = sorted(
        (ImageryVersion(release=all_releases[n]) for n in set(version_release_nums) if n in all_releases),
        key=lambda v: v.release.sort_key,
        reverse=True,
    )
    for release in releases:
        for version in versions:
            if version.release.sort_key <= release.sort_key:
                version.represented_release_nums.append(release.release_num)
                break
    return versions


def collapse_same_capture(versions: list[ImageryVersion]) -> list[ImageryVersion]:
    """Merge versions whose AOI capture signature is identical (re-processed pixels).

    Keeps the newest release of each group; older ones are recorded in
    ``collapsed_release_nums``. Versions without capture metadata never merge.
    """
    kept: list[ImageryVersion] = []
    by_sig: dict[tuple[Any, ...], ImageryVersion] = {}
    for version in sorted(versions, key=lambda v: v.release.sort_key, reverse=True):
        sig = version.capture.signature
        if sig is None:
            kept.append(version)
            continue
        keeper = by_sig.get(sig)
        if keeper is None:
            by_sig[sig] = version
            kept.append(version)
            continue
        keeper.collapsed_release_nums.append(version.release.release_num)
        keeper.collapsed_release_nums.extend(version.collapsed_release_nums)
        keeper.represented_release_nums.extend(version.represented_release_nums)
    return kept


@dataclass
class YearSelection:
    year: int
    selected: ImageryVersion
    alternatives: list[ImageryVersion]

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_release_num": self.selected.release.release_num,
            "capture_date": self.selected.effective_capture_date,
            "capture_date_source": self.selected.capture_date_source,
            "alternatives": [
                {
                    "release_num": v.release.release_num,
                    "release_date": v.release.release_date,
                    "capture_date": v.effective_capture_date,
                    "capture_date_source": v.capture_date_source,
                }
                for v in self.alternatives
            ],
            "rule": "latest_capture_date_then_latest_release",
        }


def select_per_capture_year(
    versions: Iterable[ImageryVersion], requested_years: Iterable[int]
) -> dict[int, YearSelection]:
    """Bucket versions by capture year; pick the latest capture (then latest release)."""
    wanted = set(requested_years)
    buckets: dict[int, list[ImageryVersion]] = {}
    for version in versions:
        if version.capture_year in wanted:
            buckets.setdefault(version.capture_year, []).append(version)
    out: dict[int, YearSelection] = {}
    for year, items in sorted(buckets.items()):
        ranked = sorted(
            items,
            key=lambda v: (v.effective_capture_date, v.release.release_date, v.release.release_num),
            reverse=True,
        )
        out[year] = YearSelection(year=year, selected=ranked[0], alternatives=ranked[1:])
    return out
