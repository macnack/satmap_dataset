from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satmap_dataset.providers.lantmateriet.provider import _resolve_search_options


def test_resolve_search_options_prefers_basic_auth_over_bearer() -> None:
    opts = _resolve_search_options(
        {
            "username": "lm_user",
            "password": "secret",
            "api_key": "bearer-token",
        }
    )
    assert opts.authorization is not None
    assert opts.authorization.startswith("Basic ")
    assert "bearer-token" not in opts.authorization


def test_resolve_search_options_falls_back_to_bearer() -> None:
    opts = _resolve_search_options({"api_key": "bearer-token"})
    assert opts.authorization == "Bearer bearer-token"
