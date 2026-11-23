"""Değerlendirme katmanı: ölçüte özgü dönüşümler, tahminlerin hizalanması, skor özeti.

K24 gereği data/processed altındaki metin doğal haldedir; bir ölçütün yayımlanmış
skoruyla karşılaştırılabilmek için gereken dönüşümler yalnızca burada uygulanır.
Böylece hangi sayının hangi dönüşümle üretildiği tek yerden okunur.

Tweet seti (Ersoy ve Yıldız 2024): altın cümleler küçük harfle başlar ve cümle
noktalaması içermez; mT5 ise büyük harfle başlayan, noktalamalı cümle üretir.
Yazarlar bu yüzden altın cümlenin ilk harfini büyütür ve model çıktısındaki
noktalamayı siler. Ayrıntılar makalede yok; seçenekler Kapı A'da 79,0
yeniden üretilerek sabitlenir.
"""

from __future__ import annotations

import unicodedata
from typing import Any, Sequence

from .bootstrap import bootstrap_ci
from .metrics import overcorrection_rate, prf
from .text import squeeze_spaces, tr_upper

CAPITALIZE_TURKISH = "turkish"  # i -> İ, ı -> I (Türkçe kuralı)
CAPITALIZE_ASCII = "ascii"  # Python str.upper(): i -> I
CAPITALIZE_PYTHON = "python_capitalize"  # Python str.capitalize(): ilk harf büyür, gerisi küçülür
CAPITALIZE_NONE = "none"
CAPITALIZE_MODES = (CAPITALIZE_TURKISH, CAPITALIZE_ASCII, CAPITALIZE_PYTHON, CAPITALIZE_NONE)


def capitalize_first(text: str, mode: str = CAPITALIZE_TURKISH) -> str:
    """Cümlenin ilk karakterini büyütür. Harf değilse (rakam, '#') cümle değişmez."""
    if mode not in CAPITALIZE_MODES:
        raise ValueError(f"Bilinmeyen büyütme modu: {mode}")
    if not text or mode == CAPITALIZE_NONE:
        return text
    if mode == CAPITALIZE_PYTHON:
        return text.capitalize()
    first = tr_upper(text[0]) if mode == CAPITALIZE_TURKISH else text[0].upper()
    return first + text[1:]


def strip_punctuation(text: str, keep: str = "") -> str:
    """Noktalama karakterlerini (Unicode 'P' sınıfı) siler; keep içindekilere dokunmaz.

    Noktalama boşlukla değiştirilir, sonra boşluklar sıkıştırılır: 'geldim,gittim'
    -> 'geldim gittim'. keep içindeki karakterler yerinde kalır: kesme işareti
    korunursa "Adana'da" bölünmez.
    """
    cleaned = "".join(
        ch if ch in keep or not unicodedata.category(ch).startswith("P") else " " for ch in text
    )
    return squeeze_spaces(cleaned)


def tweet_protocol(
    golds: Sequence[str],
    hypotheses: Sequence[str],
    capitalize: str = CAPITALIZE_TURKISH,
    keep_punct: str = "'’#",
) -> tuple[list[str], list[str]]:
    """Tweet setinin iki dönüşümünü uygular: (dönüştürülmüş altın, dönüştürülmüş hipotez)."""
    return (
        [capitalize_first(gold, capitalize) for gold in golds],
        [strip_punctuation(hypothesis, keep_punct) for hypothesis in hypotheses],
    )


def align_predictions(records: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tahminleri test kayıtlarının sırasına dizer.

    Her test kaydı için tam bir tahmin olmalıdır. Eksik, fazla ya da yinelenen
    kimlik hata verir: sessizce atlanan bir cümle skoru olduğundan iyi gösterir.
    """
    by_id: dict[str, dict[str, Any]] = {}
    for prediction in predictions:
        if prediction["id"] in by_id:
            raise ValueError(f"Yinelenen tahmin kimliği: {prediction['id']}")
        by_id[prediction["id"]] = prediction
    wanted = [record["id"] for record in records]
    missing = [identifier for identifier in wanted if identifier not in by_id]
    extra = set(by_id) - set(wanted)
    if missing or extra:
        raise ValueError(f"Tahmin dosyası test setiyle uyuşmuyor: {len(missing)} eksik, {len(extra)} fazla kimlik")
    return [by_id[identifier] for identifier in wanted]


def summarize(
    counts: Sequence[tuple[int, int, int]],
    beta: float = 0.5,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Cümle bazlı (tp, fp, fn) sayılarından toplam skor ve %95 güven aralığı üretir."""
    tp = sum(count[0] for count in counts)
    fp = sum(count[1] for count in counts)
    fn = sum(count[2] for count in counts)
    precision, recall, f_score = prf(tp, fp, fn, beta)
    interval = bootstrap_ci(counts, n_boot=n_boot, beta=beta, seed=seed)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f_beta": round(f_score, 4),
        "f_beta_low": round(interval["low"], 4),
        "f_beta_high": round(interval["high"], 4),
    }


def clean_sentence_stats(sources: Sequence[str], golds: Sequence[str], hypotheses: Sequence[str]) -> dict[str, Any]:
    """Hatasız cümlelerde (kaynak = altın) modelin değişiklik yapma oranı."""
    clean = [(source, hypothesis) for source, gold, hypothesis in zip(sources, golds, hypotheses) if squeeze_spaces(source) == squeeze_spaces(gold)]
    rate = overcorrection_rate([s for s, _ in clean], [h for _, h in clean]) if clean else None
    return {"clean_sentences": len(clean), "overcorrection_rate": None if rate is None else round(rate, 4)}
