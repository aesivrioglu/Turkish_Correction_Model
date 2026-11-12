"""Kaynak ve hipotez cümle arasındaki token düzeyinde farklar.

Bu modül yalnızca iç kullanım içindir (emniyet kemeri, aşırı düzeltme oranı,
hızlı hata incelemesi). Raporlanan P, R ve F0.5 değerleri için her test setinin
yayımlanmış skorunda kullanılan resmi araç kullanılır (bkz. K12).
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

from .text import tokenize


@dataclass(frozen=True)
class TokenEdit:
    """Kaynak cümlede [start, end) aralığındaki tokenların hedefle değişimi."""

    start: int
    end: int
    source: tuple[str, ...]
    target: tuple[str, ...]


def token_edits(source: str, hypothesis: str) -> list[TokenEdit]:
    """İki cümle arasındaki değişiklikleri sırayla döndürür."""
    src, hyp = tokenize(source), tokenize(hypothesis)
    matcher = SequenceMatcher(a=src, b=hyp, autojunk=False)
    return [
        TokenEdit(i1, i2, tuple(src[i1:i2]), tuple(hyp[j1:j2]))
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    ]


def token_edit_distance(source: str, hypothesis: str) -> int:
    """Token düzeyinde Levenshtein mesafesi (ekleme, silme, değiştirme = 1)."""
    src, hyp = tokenize(source), tokenize(hypothesis)
    prev = list(range(len(hyp) + 1))
    for i, s_tok in enumerate(src, start=1):
        curr = [i]
        for j, h_tok in enumerate(hyp, start=1):
            cost = 0 if s_tok == h_tok else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]
