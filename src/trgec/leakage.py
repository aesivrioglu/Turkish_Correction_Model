"""Sızıntı taraması: eğitim ve test verisi arasında yinelenen cümleler.

OSCAR bir web taramasıdır; film yorumları ve tweetler eğitim verisinin içinde
bulunabilir. Bu yüzden yalnızca aynı kaynaktan gelen ayrımlar değil, bütün
eğitim-test çiftleri taranır.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

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


def match_keys(texts: Iterable[str]) -> set[str]:
    """Metinlerin normalleştirilmiş hallerinin kümesi. Boş anahtar atılır."""
    keys = {normalize_for_match(text) for text in texts}
    keys.discard("")
    return keys


def split_by_overlap(
    records: Iterable[dict[str, Any]],
    keys: set[str],
    fields: Sequence[str] = ("source", "target"),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Kayıtları (temiz, sızıntılı) olarak ikiye ayırır.

    Bir kaydın fields içindeki alanlarından herhangi biri keys içindeyse kayıt
    sızıntılı sayılır. Hedef de taranır: eğitimdeki düzeltilmiş cümle bir test
    cümlesiyle aynıysa model o cümlenin cevabını görmüş olur.
    """
    kept: list[dict[str, Any]] = []
    leaked: list[dict[str, Any]] = []
    for record in records:
        if any(normalize_for_match(record[field]) in keys for field in fields):
            leaked.append(record)
        else:
            kept.append(record)
    return kept, leaked


def dedupe_records(records: Iterable[dict[str, Any]], field: str = "source") -> tuple[list[dict[str, Any]], int]:
    """Aynı (normalleştirilmiş) alanı taşıyan kayıtlardan yalnızca ilkini tutar.

    Döndürür: (tekil kayıtlar, atılan yineleme sayısı). Sıra korunur.
    """
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    dropped = 0
    for record in records:
        key = normalize_for_match(record[field])
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        unique.append(record)
    return unique, dropped
