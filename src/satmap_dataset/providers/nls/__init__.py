from __future__ import annotations

__all__ = ["NlsProvider"]


def __getattr__(name: str):
    if name == "NlsProvider":
        from satmap_dataset.providers.nls.provider import NlsProvider as _NlsProvider

        return _NlsProvider
    raise AttributeError(name)
