from __future__ import annotations

from pathlib import Path

from satmap_dataset.providers.lroc_nac import isis


def test_build_lronac2isis_cmd() -> None:
    cmd = isis.build_lronac2isis_cmd(Path("a.IMG"), Path("a.cub"), binary="/opt/isis/lronac2isis")
    assert cmd == ["/opt/isis/lronac2isis", "from=a.IMG", "to=a.cub"]


def test_build_spiceinit_cmd() -> None:
    cmd = isis.build_spiceinit_cmd(Path("a.cub"), binary="spiceinit")
    assert cmd == ["spiceinit", "from=a.cub"]


def test_build_cam2map_cmd_with_map_file() -> None:
    cmd = isis.build_cam2map_cmd(
        Path("a.cub"),
        Path("a_map.cub"),
        map_file=Path("moon.map"),
        binary="cam2map",
    )
    assert cmd == [
        "cam2map",
        "from=a.cub",
        "to=a_map.cub",
        "map=moon.map",
    ]


def test_missing_isis_tools() -> None:
    tools = {name: None for name in isis.REQUIRED_TOOLS}
    assert isis.missing_isis_tools(tools) == list(isis.REQUIRED_TOOLS)
    assert not isis.isis_available(tools)


def test_resolve_isis_tools_uses_which() -> None:
    def fake_which(name: str) -> str | None:
        if name == "spiceinit":
            return "/usr/bin/spiceinit"
        return None

    tools = isis.resolve_isis_tools(which=fake_which)
    assert tools["spiceinit"] == Path("/usr/bin/spiceinit")
    assert tools["cam2map"] is None
    assert isis.missing_isis_tools(tools) == ["lronac2isis", "cam2map"]
