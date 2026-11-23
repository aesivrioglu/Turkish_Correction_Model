"""Örnekleme: katmanlı (GECTurk) ve rezervuar (OSCAR).

GECTurk'teki 25 kuralın sıklığı çok dengesizdir. Rastgele küçük bir alt küme
nadir kurallardan yalnızca birkaç örnek bırakır (bkz. K10). stratified_sample
önce her kural için bir alt sınırı garanti eder, sonra kotayı rastgele doldurur.

OSCAR eğitim verisi 2,3 milyon satırdır ve diske kaydedilmeden akış halinde
okunur. reservoir_sample, uzunluğu baştan bilinmeyen böyle bir akıştan tek
geçişte eşit olasılıklı rastgele örnek seçer.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any, Iterable, Sequence, TypeVar

T = TypeVar("T")


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


class Reservoir:
    """Bir akıştan, tamamını bellekte tutmadan k öğelik rastgele örnek seçer.

    Yöntem (Algoritma R): ilk k öğe doğrudan alınır. n. öğe (n > k) geldiğinde
    0 ile n-1 arasında rastgele bir sayı çekilir; sayı k'dan küçükse o sıradaki
    öğenin yerine yenisi konur. Böylece akış bittiğinde her öğenin seçilmiş
    olma olasılığı eşittir: k / toplam.

    Sınıf olmasının nedeni: aynı akış tek geçişte birden çok bağımsız örneğe
    (farklı tohumlarla) beslenebilsin.
    """

    def __init__(self, k: int, seed: int = 42) -> None:
        if k < 0:
            raise ValueError("k negatif olamaz")
        self.k = k
        self.seen = 0
        self._rng = random.Random(seed)
        self._items: list[Any] = []

    def offer(self, item: Any) -> None:
        """Akıştaki bir sonraki öğeyi sunar."""
        self.seen += 1
        if self.seen <= self.k:
            self._items.append(item)
        else:
            slot = self._rng.randrange(self.seen)
            if slot < self.k:
                self._items[slot] = item

    def result(self) -> list[Any]:
        """Seçilen öğeleri karıştırılmış sırayla döndürür. Akış bitince bir kez çağrılır."""
        items = list(self._items)
        self._rng.shuffle(items)
        return items


def reservoir_sample(items: Iterable[T], k: int, seed: int = 42) -> list[T]:
    """Bir akıştan k öğelik rastgele örnek seçer (bkz. Reservoir).

    Akışta k'dan az öğe varsa hepsi döner. Aynı akış ve aynı tohum hep aynı
    sonucu verir; sıra akıştaki sırayı yansıtmaz.
    """
    reservoir = Reservoir(k, seed)
    for item in items:
        reservoir.offer(item)
    return reservoir.result()


def mix_with_clean(
    errors: Sequence[T],
    clean: Sequence[T],
    n_total: int,
    clean_ratio: float,
    seed: int = 42,
) -> list[T]:
    """Hatalı ve hatasız örnekleri verilen oranda karıştırır.

    Hatasız örnek sayısı round(n_total * clean_ratio); kalanı hatalı örnektir.
    İki listeden de baştan alınır. Böylece oran değişse bile seçilen hatalı
    örnekler iç içe kalır (%50'deki 6.000 örnek, %25'teki 9.000'in ilk 6.000'idir).
    """
    if not 0.0 <= clean_ratio <= 1.0:
        raise ValueError("clean_ratio 0 ile 1 arasında olmalı")
    n_clean = round(n_total * clean_ratio)
    n_errors = n_total - n_clean
    if len(clean) < n_clean or len(errors) < n_errors:
        raise ValueError(
            f"Yeterli örnek yok: {n_errors} hatalı ve {n_clean} hatasız gerekli, "
            f"{len(errors)} hatalı ve {len(clean)} hatasız var"
        )
    mixed = list(errors[:n_errors]) + list(clean[:n_clean])
    random.Random(seed).shuffle(mixed)
    return mixed
