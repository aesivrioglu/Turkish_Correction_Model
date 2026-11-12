"""Katmanlı örnekleme.

GECTurk'teki 25 kuralın sıklığı çok dengesizdir. Rastgele küçük bir alt küme
nadir kurallardan yalnızca birkaç örnek bırakır (bkz. K10). Bu işlev önce her
kural için bir alt sınırı garanti eder, sonra kotayı rastgele doldurur.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any, Sequence


def rule_counts(examples: Sequence[dict[str, Any]], rules_key: str = "rules") -> Counter:
    """Her kuralın kaç örnekte geçtiğini sayar."""
    counts: Counter = Counter()
    for ex in examples:
        counts.update(set(ex[rules_key]))
    return counts


def stratified_sample(
    examples: Sequence[dict[str, Any]],
    n_total: int,
    clean_ratio: float = 0.5,
    min_per_rule: int = 50,
    seed: int = 42,
    rules_key: str = "rules",
) -> list[dict[str, Any]]:
    """Kural başına alt sınırlı alt küme seçer.

    examples:     her biri rules_key altında kural listesi taşıyan sözlükler.
                  Boş liste hatasız cümle demektir.
    n_total:      hedef toplam örnek sayısı.
    clean_ratio:  hatasız cümlelerin hedef oranı (GECTurk'te 0.5).
    min_per_rule: her kuraldan en az bu kadar örnek (kuralın toplamı daha azsa hepsi).

    Alt sınırlar kotadan önceliklidir: kurallar çoksa ve n_total küçükse sonuç
    n_total'ı biraz aşabilir.
    """
    if not 0.0 <= clean_ratio <= 1.0:
        raise ValueError("clean_ratio 0 ile 1 arasında olmalı")
    rng = random.Random(seed)

    clean = [i for i, ex in enumerate(examples) if not ex[rules_key]]
    dirty = [i for i, ex in enumerate(examples) if ex[rules_key]]
    n_clean = min(len(clean), round(n_total * clean_ratio))
    n_dirty = min(len(dirty), n_total - n_clean)

    by_rule: dict[str, list[int]] = defaultdict(list)
    for i in dirty:
        for rule in set(examples[i][rules_key]):
            by_rule[rule].append(i)

    chosen: set[int] = set()
    # En nadir kuraldan başla; sık kurallar zaten kotadan yeterince örnek alır.
    for rule in sorted(by_rule, key=lambda r: (len(by_rule[r]), r)):
        pool = by_rule[rule]
        have = sum(1 for i in pool if i in chosen)
        need = min(min_per_rule, len(pool)) - have
        if need > 0:
            candidates = [i for i in pool if i not in chosen]
            rng.shuffle(candidates)
            chosen.update(candidates[:need])

    remaining = [i for i in dirty if i not in chosen]
    rng.shuffle(remaining)
    chosen.update(remaining[: max(0, n_dirty - len(chosen))])

    picked = sorted(chosen) + rng.sample(clean, n_clean)
    rng.shuffle(picked)
    return [examples[i] for i in picked]
