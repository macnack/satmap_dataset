"""satmap-web — FastAPI + React browser UI for satmap_dataset."""

from __future__ import annotations

__all__ = ["create_app"]


def create_app():
    from satmap_dataset.web.app import create_app as _create

    return _create()
