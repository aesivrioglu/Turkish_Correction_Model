"""Türkçe metin yardımcıları.

Dikkat: Python'un str.lower() işlevi Türkçe için yanlıştır.
"I".lower() "i" verir (doğrusu "ı"), "İ".lower() ise "i" + birleşen nokta verir.
Bu yüzden karşılaştırma yapan her yerde tr_lower kullanılmalı.
"""

from __future__ import annotations

import re
import unicodedata

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


def tr_lower(text: str) -> str:
    """Türkçe kurallarına göre küçük harfe çevirir (I -> ı, İ -> i)."""
    return text.replace("I", "ı").replace("İ", "i").lower()


def tr_upper(text: str) -> str:
    """Türkçe kurallarına göre büyük harfe çevirir (i -> İ, ı -> I)."""
    return text.replace("i", "İ").replace("ı", "I").upper()


def squeeze_spaces(text: str) -> str:
    """Art arda boşlukları teke indirir, baş ve sondaki boşluğu siler."""
    return _WS_RE.sub(" ", text).strip()


def normalize_for_match(text: str) -> str:
    """Sızıntı ve yineleme taraması için gevşek eşleştirme biçimi.

    Unicode'u NFC'ye çevirir, Türkçe küçük harfe indirir, noktalamayı siler.
    Yalnızca karşılaştırma içindir; eğitim ya da skorlama verisine uygulanmaz.
    """
    text = unicodedata.normalize("NFC", text)
    text = tr_lower(text)
    text = _PUNCT_RE.sub(" ", text)
    return squeeze_spaces(text)


def tokenize(text: str) -> list[str]:
    """Boşluktan böler. GECTurk M2 dosyaları boşlukla ayrılmış token kullanır."""
    return text.split()
