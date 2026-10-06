from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.providers.esri_wayback import versions
from satmap_dataset.providers.esri_wayback.catalog import WaybackRelease, sort_newest_first


def _releases() -> list[WaybackRelease]:
    # Release numbers deliberately non-chronological, like the live service.
    rows = [
        (500, "2026-08-05"),
        (120, "2026-03-25"),
        (900, "2025-06-01"),
        (310, "2024-03-07"),
        (77, "2022-01-10"),
        (10, "2014-02-20"),
    ]
    return sort_newest_first(WaybackRelease(n, f"WB_{d[:4]}_R01", d) for n, d in rows)


def _fake_tilemap(table: dict[int, versions.TilemapResult], calls: list[int]):
    async def fetch(release: WaybackRelease, _tile):
        calls.append(release.release_num)
        return table[release.release_num]

    return fetch


def test_tilemap_from_json() -> None:
    assert versions.TilemapResult.from_json({"data": [1], "select": [22869]}) == versions.TilemapResult(True, 22869)
    assert versions.TilemapResult.from_json({"data": [1]}) == versions.TilemapResult(True, None)
    assert versions.TilemapResult.from_json({"data": [0]}).has_data is False


def test_walk_local_changes_jumps_past_reserved_releases() -> None:
    T = versions.TilemapResult
    table = {
        500: T(True, 120),  # newest release re-serves 120
        120: T(True, None),
        900: T(True, 77),  # 900 and 310 re-serve 77
        310: T(True, 77),
        77: T(True, None),
        10: T(True, None),
    }
    calls: list[int] = []
    origins, n = asyncio.run(
        versions.walk_local_changes(_releases(), (17, 1, 1), _fake_tilemap(table, calls), max_requests=50)
    )
    assert origins == [120, 77, 10]
    # 500 -> jump to after 120 (900) -> jump to after 77 (10) -> done; 310 never queried.
    assert calls == [500, 900, 10]
    assert n == 3


def test_walk_skips_releases_without_data_and_stops_at_out_of_window_origin() -> None:
    T = versions.TilemapResult
    table = {500: T(False), 120: T(True, None), 900: T(True, 4242)}
    calls: list[int] = []
    origins, _ = asyncio.run(
        versions.walk_local_changes(_releases()[:3], (17, 1, 1), _fake_tilemap(table, calls), max_requests=50)
    )
    assert origins == [120, 4242]
    assert calls == [500, 120, 900]


def test_walk_respects_request_cap() -> None:
    T = versions.TilemapResult
    table = {r.release_num: T(True, None) for r in _releases()}
    _, n = asyncio.run(versions.walk_local_changes(_releases(), (17, 1, 1), _fake_tilemap(table, []), max_requests=2))
    assert n == 2


def test_group_by_content_hash_prefers_redirect_origin_then_oldest() -> None:
    rel = _releases()
    obs = {
        500: ("aaa", 120),
        120: ("aaa", None),
        900: ("bbb", None),
        310: ("bbb", None),
        77: ("ccc", None),
    }
    assert sorted(versions.group_by_content_hash(rel, obs)) == sorted([120, 310, 77])


def test_parse_src_date_and_capture_features() -> None:
    assert versions.parse_src_date(20230905) == "2023-09-05"
    assert versions.parse_src_date("20230905") == "2023-09-05"
    assert versions.parse_src_date(0) is None
    assert versions.parse_src_date(20231399) is None
    point = [{"attributes": {"SRC_DATE": 20230905, "SRC_DESC": "Poland Orthos 2023", "NICE_DESC": "GUGiK", "SRC_RES": 0.05, "SRC_ACC": 99999}}]
    env = point + [{"attributes": {"SRC_DATE": 20230329, "SRC_DESC": "WV03", "SRC_RES": 0.3}}, point[0]]
    info = versions.capture_from_features(point, env)
    assert info.capture_date == "2023-09-05"
    assert info.provider_name == "GUGiK"
    assert info.accuracy_m is None  # 99999 = unknown
    assert [c["source"] for c in info.captures_in_aoi] == ["WV03", "Poland Orthos 2023"]


def _version(num: int, release_date: str, capture: str | None, source: str = "WV02") -> versions.ImageryVersion:
    info = versions.CaptureInfo(capture_date=capture, source=source, resolution_m=0.5)
    if capture:
        info.captures_in_aoi = [{"capture_date": capture, "source": source, "resolution_m": 0.5}]
    return versions.ImageryVersion(release=WaybackRelease(num, "WB", release_date), capture=info, represented_release_nums=[num])


def test_collapse_same_capture_keeps_newest_release() -> None:
    a = _version(1, "2020-01-01", "2019-05-01")
    b = _version(2, "2021-01-01", "2019-05-01")  # same capture, re-processed
    c = _version(3, "2022-01-01", None)  # no metadata: never merged
    d = _version(4, "2023-01-01", None)
    kept = versions.collapse_same_capture([a, b, c, d])
    assert [v.release.release_num for v in kept] == [4, 3, 2]
    assert kept[2].collapsed_release_nums == [1]
    assert kept[2].deduped is True
    assert sorted(kept[2].represented_release_nums) == [1, 2]


def test_select_per_capture_year_is_deterministic_with_alternatives() -> None:
    early = _version(1, "2019-03-01", "2018-04-01")
    late = _version(2, "2019-09-01", "2018-08-01")
    tie = _version(3, "2020-01-01", "2018-08-01", source="GE01")
    fallback = _version(4, "2021-06-01", None)  # capture year from release date
    sel = versions.select_per_capture_year([early, late, tie, fallback], range(2015, 2026))
    assert sorted(sel) == [2018, 2021]
    assert sel[2018].selected.release.release_num == 3  # same capture date -> newer release wins
    assert [v.release.release_num for v in sel[2018].alternatives] == [2, 1]
    assert sel[2021].selected.capture_date_source == "release_date"
    payload = sel[2018].as_dict()
    assert payload["selected_release_num"] == 3
    assert [a["release_num"] for a in payload["alternatives"]] == [2, 1]


def test_assign_represented_releases_maps_each_release_to_version() -> None:
    rel = _releases()
    by_num = {r.release_num: r for r in rel}
    out = versions.assign_represented_releases([120, 77, 10], rel, by_num)
    assert {v.release.release_num: v.represented_release_nums for v in out} == {
        120: [500, 120],
        77: [900, 310, 77],
        10: [10],
    }
