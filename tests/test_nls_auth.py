from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from satmap_dataset.providers.nls.auth import basic_auth_header, resolve_api_key


def test_resolve_api_key_prefers_provider_options(monkeypatch, tmp_path):
    monkeypatch.setenv("SATMAP_NLS_API_KEY", "from-env")
    secret = tmp_path / ".secret"
    secret.write_text("from-secret\n", encoding="utf-8")
    key = resolve_api_key({"api_key": "from-options"}, secret_path=secret)
    assert key == "from-options"


def test_resolve_api_key_falls_back_to_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SATMAP_NLS_API_KEY", "from-env")
    secret = tmp_path / ".secret"
    secret.write_text("from-secret\n", encoding="utf-8")
    key = resolve_api_key({}, secret_path=secret)
    assert key == "from-env"


def test_resolve_api_key_falls_back_to_secret_file(monkeypatch, tmp_path):
    monkeypatch.delenv("SATMAP_NLS_API_KEY", raising=False)
    secret = tmp_path / ".secret"
    secret.write_text("from-secret\n", encoding="utf-8")
    key = resolve_api_key({}, secret_path=secret)
    assert key == "from-secret"


def test_resolve_api_key_missing_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("SATMAP_NLS_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="No NLS API key"):
        resolve_api_key({}, secret_path=tmp_path / "nope")


def test_basic_auth_header_uses_api_key_username():
    header = basic_auth_header("MYKEY")
    # base64("api-key:MYKEY") == "YXBpLWtleTpNWUtFWQ=="
    assert header == "Basic YXBpLWtleTpNWUtFWQ=="
