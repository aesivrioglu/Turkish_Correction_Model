"""Bootstrap güven aralıkları.

Girdi, cümle başına (tp, fp, fn) sayılarıdır. Bunlar resmi skorlayıcının cümle
bazlı çıktısından alınır. 300 cümlelik sette iki model arasındaki küçük farklar
tesadüf olabilir; fark yalnızca güven aralığı sıfırı içermiyorsa anlamlı sayılır.
"""

from __future__ import annotations

import random
from typing import Sequence

from .metrics import prf

Counts = Sequence[tuple[int, int, int]]


def _f_score(counts: Counts, indices: Sequence[int], beta: float) -> float:
    tp = sum(counts[i][0] for i in indices)
    fp = sum(counts[i][1] for i in indices)
    fn = sum(counts[i][2] for i in indices)
    return prf(tp, fp, fn, beta)[2]


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("Boş liste")
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def bootstrap_ci(
    counts: Counts, n_boot: int = 1000, beta: float = 0.5, alpha: float = 0.05, seed: int = 42
) -> dict[str, float]:
    """Tek bir sistemin F-beta skoru için yüzdelik güven aralığı."""
    rng = random.Random(seed)
    n = len(counts)
    all_idx = range(n)
    scores = sorted(
        _f_score(counts, [rng.randrange(n) for _ in all_idx], beta) for _ in range(n_boot)
    )
    return {
        "score": _f_score(counts, all_idx, beta),
        "low": _percentile(scores, alpha / 2),
        "high": _percentile(scores, 1 - alpha / 2),
    }


def paired_bootstrap(
    counts_a: Counts,
    counts_b: Counts,
    n_boot: int = 1000,
    beta: float = 0.5,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """A ve B sistemlerinin F-beta farkı (A - B) için eşleştirilmiş bootstrap.

    Aynı cümleler iki sistem için de birlikte örneklenir. p_a_not_better,
    yeniden örneklemelerin ne kadarında A'nın B'den iyi olmadığını gösterir.
    """
    if len(counts_a) != len(counts_b):
        raise ValueError("İki sistem aynı cümle kümesinde ölçülmeli")
    rng = random.Random(seed)
    n = len(counts_a)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(_f_score(counts_a, idx, beta) - _f_score(counts_b, idx, beta))
    diffs.sort()
    all_idx = range(n)
    score_a = _f_score(counts_a, all_idx, beta)
    score_b = _f_score(counts_b, all_idx, beta)
    return {
        "score_a": score_a,
        "score_b": score_b,
        "diff": score_a - score_b,
        "low": _percentile(diffs, alpha / 2),
        "high": _percentile(diffs, 1 - alpha / 2),
        "p_a_not_better": sum(1 for d in diffs if d <= 0) / n_boot,
    }
