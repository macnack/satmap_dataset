from __future__ import annotations

import importlib
from typing import Callable

from satmap_dataset.layers.base import Layer

__all__ = ["Layer", "get_layer", "register_layer"]

_RGB_SUFFIX = "_rgb"

# Registry of non-RGB layers. Modules under satmap_dataset.layers self-register
# on import via the @register_layer decorator; get_layer triggers a lazy import
# of layers/<name>.py so adding a source needs no edit here.
_FACTORIES: dict[str, Callable[[], Layer]] = {}


def register_layer(name: str) -> Callable[[type[Layer]], type[Layer]]:
    def decorator(cls: type[Layer]) -> type[Layer]:
        _FACTORIES[name] = cls
        return cls

    return decorator


def get_layer(name: str) -> Layer:
    """Resolve a layer by registry name.

    RGB layers use the convention ``<provider>_rgb`` (e.g. ``geoportal_rgb``).
    Other modalities (``dem``, ``osm``) self-register from their module.
    """
    if name.endswith(_RGB_SUFFIX):
        from satmap_dataset.layers.rgb import RgbLayer

        provider = name[: -len(_RGB_SUFFIX)]
        return RgbLayer(provider)

    if name not in _FACTORIES:
        try:
            importlib.import_module(f"satmap_dataset.layers.{name}")
        except ModuleNotFoundError:
            pass
    factory = _FACTORIES.get(name)
    if factory is None:
        raise ValueError(
            f"Unknown layer: {name!r}. Expected '<provider>_rgb', 'dem', or 'osm'."
        )
    return factory()
