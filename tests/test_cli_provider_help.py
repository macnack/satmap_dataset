from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _cli_env() -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not existing else f"{src_path}{os.pathsep}{existing}"
    # Keep Typer/Rich help plain and wide so option names are not wrapped/colored away.
    env["NO_COLOR"] = "1"
    env["FORCE_COLOR"] = "0"
    env["TERM"] = "dumb"
    env["COLUMNS"] = "200"
    return env


def _strip_ansi(text: str) -> str:
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", text)


@pytest.mark.parametrize("subcommand", ["index", "download", "run"])
def test_cli_subcommand_help_lists_provider_option(subcommand: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "satmap_dataset.cli", subcommand, "--help"],
        cwd=ROOT,
        env=_cli_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    flat = " ".join(_strip_ansi(result.stdout).split())
    assert "--provider" in flat
    assert "geoportal" in flat
    assert "lantmateriet" in flat
    assert "landsd_hk" in flat


def test_cli_index_rejects_unknown_provider() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "satmap_dataset.cli",
            "index",
            "--year-start",
            "2015",
            "--year-end",
            "2016",
            "--bbox",
            "0,0,1,1",
            "--provider",
            "opendata",
        ],
        cwd=ROOT,
        env=_cli_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    combined = (result.stdout + result.stderr).lower()
    assert "provider" in combined
