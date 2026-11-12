"""Emniyet kemeri: model cümleyi fazla değiştirdiyse orijinali koru.

LLM'ler düzeltmek yerine cümleyi yeniden yazmaya eğilimlidir. Bu son işlem adımı
yeniden eğitim gerektirmez. Eşik yalnızca doğrulama setinde ayarlanır, test
setlerinde ayarlanmaz (bkz. K13).
"""

from __future__ import annotations

from .edits import token_edit_distance
from .text import squeeze_spaces, tokenize


def change_ratio(source: str, hypothesis: str) -> float:
    """Değişen token sayısının kaynak cümlenin token sayısına oranı."""
    n_source = max(1, len(tokenize(source)))
    return token_edit_distance(source, hypothesis) / n_source


def apply_guard(source: str, hypothesis: str, max_ratio: float) -> tuple[str, bool]:
    """(çıktı, kemer_devreye_girdi_mi) döndürür.

    Hipotez boşsa ya da değişim oranı max_ratio'yu aşıyorsa kaynak cümle döner.
    """
    source = squeeze_spaces(source)
    hypothesis = squeeze_spaces(hypothesis)
    if not hypothesis:
        return source, True
    if change_ratio(source, hypothesis) > max_ratio:
        return source, True
    return hypothesis, False
