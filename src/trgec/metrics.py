"""Temel metrikler."""

from __future__ import annotations

from typing import Sequence

from .text import squeeze_spaces


def prf(tp: int, fp: int, fn: int, beta: float = 0.5) -> tuple[float, float, float]:
    """Precision, recall ve F-beta. GEC'te beta=0.5: precision iki kat ağırlıklı."""
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    if precision + recall == 0:
        return precision, recall, 0.0
    b2 = beta * beta
    f_score = (1 + b2) * precision * recall / (b2 * precision + recall)
    return precision, recall, f_score


def overcorrection_rate(sources: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Hatasız cümlelerde modelin değişiklik yapma oranı.

    sources yalnızca hatasız (altın düzeltmesi olmayan) cümleleri içermelidir.
    """
    if len(sources) != len(hypotheses):
        raise ValueError("sources ve hypotheses aynı uzunlukta olmalı")
    if not sources:
        return 0.0
    changed = sum(
        1 for s, h in zip(sources, hypotheses) if squeeze_spaces(s) != squeeze_spaces(h)
    )
    return changed / len(sources)
