"""Pilot koşusunun küçük yardımcıları: bellek taşmasını tanıma ve grup boyutunu küçültme.

T4'ün belleği 15 GB'dir. Bir eğitim adımının bellek ihtiyacı kabaca
(grup boyutu x en uzun örnek x sözlük boyutu) ile büyür: model her konum için
sözlükteki bütün tokenlara bir puan üretir. Sözlüğü 151 bin token olan Qwen3
ile 41 bin token olan ufakzeka aynı grup boyutunu kaldıramaz. Bu yüzden grup
boyutu elle sabitlenmez; taşarsa yarıya indirilip yeniden denenir.
"""

from __future__ import annotations

OOM_MARKERS = ("CUDA out of memory", "OutOfMemoryError")


def is_out_of_memory(log_text: str) -> bool:
    """Günlükte GPU belleğinin taştığını gösteren bir satır var mı?"""
    return any(marker in log_text for marker in OOM_MARKERS)


def halve_batch(batch_size: int, grad_accum: int) -> tuple[int, int] | None:
    """Grup boyutunu yarıya indirir, gradyan biriktirmeyi iki katına çıkarır.

    Çarpımları (etkin grup boyutu) değişmez: model yine 16 örnekte bir güncellenir,
    yalnızca bu 16 örnek GPU'dan daha küçük parçalar halinde geçer. Sonuç aynı
    kalır, bellek ihtiyacı yarıya iner, süre biraz uzar. Grup boyutu 1 ise
    küçültülecek yer kalmamıştır; None döner.
    """
    if batch_size <= 1:
        return None
    return batch_size // 2, grad_accum * (batch_size // (batch_size // 2))
