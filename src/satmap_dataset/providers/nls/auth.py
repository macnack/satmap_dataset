from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any


ENV_VAR = "SATMAP_NLS_API_KEY"


def resolve_api_key(
    provider_options: dict[str, Any],
    *,
    secret_path: Path | None = None,
) -> str:
    """Resolve the NLS open-data API key.

    Order: provider_options["api_key"] -> env var SATMAP_NLS_API_KEY ->
    single-line .secret file at secret_path. Raises RuntimeError if none set.
    """
    candidate = provider_options.get("api_key")
    if candidate:
        return str(candidate).strip()
    env_value = os.environ.get(ENV_VAR)
    if env_value:
        return env_value.strip()
    if secret_path is not None and secret_path.is_file():
        text = secret_path.read_text(encoding="utf-8").strip()
        if text:
            return text
    raise RuntimeError(
        "No NLS API key found. Set provider_options['api_key'], "
        f"env var {ENV_VAR}, or place a single-line .secret file at the project root."
    )


def basic_auth_header(api_key: str) -> str:
    token = base64.b64encode(f"api-key:{api_key}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"
