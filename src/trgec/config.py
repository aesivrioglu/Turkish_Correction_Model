"""YAML yapılandırma yükleyici. 'inherits' anahtarıyla temel dosyadan miras alır."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Yapılandırmayı yükler; 'inherits' varsa önce o dosyayı yükleyip üzerine yazar."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    parent = config.pop("inherits", None)
    if parent:
        config = _merge(load_config(path.parent / parent), config)
    return config
