"""Token uzunluğu sınırını ölçerek belirleme.

Sınır tahminle konmaz (bkz. K11). Pilotta seçilen modelin tokenizer'ıyla her
eğitim örneğinin (talimat + cümle + hedef) uzunluğu ölçülür ve bu işlevle
sınır seçilir.
"""

from __future__ import annotations

import math
from typing import Sequence


def suggest_max_len(lengths: Sequence[int], quantile: float = 0.95, step: int = 32) -> int:
    """Uzunlukların verilen yüzdeliğini step'in katına yukarı yuvarlar."""
    if not lengths:
        raise ValueError("Uzunluk listesi boş")
    ordered = sorted(lengths)
    index = min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)
    return int(math.ceil(ordered[max(0, index)] / step) * step)


def truncation_rate(lengths: Sequence[int], max_len: int) -> float:
    """max_len sınırında kesilecek örneklerin oranı."""
    if not lengths:
        return 0.0
    return sum(1 for n in lengths if n > max_len) / len(lengths)
