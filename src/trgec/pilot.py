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


STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_NAN = "nan"
STATUS_NO_LEARNING = "no_learning"


def training_status(train_ok: bool, loss_first: float | None, loss_last: float | None, nan_step: int | None = None) -> str:
    """Bir adayın eğitimi sağlıklı mı? Tahminleri skorlanmaya değer mi?

    Eğitim betiğinin hatasız bitmesi yetmez: kayıp hiç düşmediyse model bir şey
    öğrenmemiştir ve tahminleri anlamsızdır (pilotta Mecellem: kayıp baştan sona
    11,93 = ln(sözlük boyutu), yani model her tokena eşit olasılık veriyor).
    Kayıp NaN olduysa eğitim yarıda durmuştur (pilotta Kanarya, fp16).
    Yalnızca "ok" dönen adaylar skorlanır.
    """
    if nan_step is not None or loss_first != loss_first or loss_last != loss_last:  # NaN kendine eşit değildir
        return STATUS_NAN
    if not train_ok or loss_first is None or loss_last is None:
        return STATUS_FAILED
    if loss_last >= loss_first:
        return STATUS_NO_LEARNING
    return STATUS_OK
