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
    return env


@pytest.mark.parametrize(
    "args",
    [
        ["--help"],
        ["index", "--help"],
        ["download", "--help"],
        ["render", "--help"],
        ["mosaic", "--help"],
        ["validate", "--help"],
        ["run", "--help"],
    ],
)
def test_cli_help_smoke(args: list[str]) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "satmap_dataset.cli", *args],
        cwd=ROOT,
        env=_cli_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Usage" in result.stdout


def test_index_help_marks_swap_as_deprecated() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "satmap_dataset.cli", "index", "--help"],
        cwd=ROOT,
        env=_cli_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Deprecated" in result.stdout
    assert "legacy option" in result.stdout
