"""Sızıntı taraması: eğitim ve test verisi arasında yinelenen cümleler.

OSCAR bir web taramasıdır; film yorumları ve tweetler eğitim verisinin içinde
bulunabilir. Bu yüzden yalnızca aynı kaynaktan gelen ayrımlar değil, bütün
eğitim-test çiftleri taranır.
"""

from __future__ import annotations

from typing import Iterable

from .text import normalize_for_match


def find_overlap(train: Iterable[str], test: Iterable[str]) -> list[str]:
    """İki kümede de geçen cümlelerin normalleştirilmiş hallerini döndürür."""
    test_keys = {normalize_for_match(s) for s in test}
    test_keys.discard("")
    return sorted({normalize_for_match(s) for s in train} & test_keys)


def drop_overlap(train: Iterable[str], test: Iterable[str]) -> list[str]:
    """Test kümesinde de geçen cümleleri eğitim kümesinden çıkarır."""
    test_keys = {normalize_for_match(s) for s in test}
    return [s for s in train if normalize_for_match(s) not in test_keys]
