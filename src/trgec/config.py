"""YAML yapılandırma yükleyici. 'inherits' anahtarıyla temel dosyadan miras alır."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .formats import MODE_LABEL


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


def adapter_run_name(run_name: str, model_id: str, smoke: bool = False) -> str:
    """Bir eğitim koşusunun klasör adı: '<çalışma>__<model>' (duman testinde 'smoke__<model>').

    train.py adaptörü bu adla yazar; run_pilot.py ve predict.py aynı adla bulur.
    """
    return f"{'smoke' if smoke else run_name}__{model_id.split('/')[-1]}"


def zero_shot_run_name(model_id: str, smoke: bool = False) -> str:
    """Eğitimsiz (adaptörsüz) bir modelin tahmin klasörünün adı: 'zeroshot__<model>'."""
    return f"{'smoke_zeroshot' if smoke else 'zeroshot'}__{model_id.split('/')[-1]}"


def prediction_run_name(base: str, mode: str) -> str:
    """Bir tahmin koşusunun klasör adı.

    "Düzelt ve etiketle" talimatı varsayılandır ve ad değişmez. "Yalnızca düzelt"
    talimatıyla alınan tahminler '<ad>__correct' klasörüne yazılır; böylece aynı
    modelin iki talimatla alınmış tahminleri birbirinin üstüne yazılmaz.
    """
    return base if mode == MODE_LABEL else f"{base}__{mode}"
