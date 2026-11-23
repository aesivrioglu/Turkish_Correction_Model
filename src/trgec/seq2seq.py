"""Seq2seq temel modelleriyle (mT5) çıkarımın modelden bağımsız parçaları.

Buradaki işlevler torch ya da transformers istemez; böylece testleri model
indirmeden çalışır. Modeli yükleyen ve üreten kod scripts/predict_seq2seq.py
içindedir.
"""

from __future__ import annotations

from typing import Any, Sequence

from .text import squeeze_spaces


def length_sorted_batches(lengths: Sequence[int], batch_size: int) -> list[list[int]]:
    """Örnekleri uzunluğa göre sıralayıp gruplara böler; her grup satır numaralarının listesidir.

    Bir gruptaki cümleler en uzununun boyuna kadar dolgu (pad) tokenıyla
    uzatılır. Benzer uzunluktaki cümleler aynı gruba düşerse dolgu ve boşa giden
    hesap azalır. En uzun grup başa konur: bellek yetmeyecekse çalışma ilk
    grupta durur, yarım saat sonra değil. Eşit uzunlukta özgün sıra korunur,
    yani sonuç her çalıştırmada aynıdır.
    """
    if batch_size < 1:
        raise ValueError("batch_size en az 1 olmalı")
    order = sorted(range(len(lengths)), key=lambda index: (-lengths[index], index))
    return [order[start : start + batch_size] for start in range(0, len(order), batch_size)]


def clean_output(text: str) -> str:
    """Model çıktısını tek satıra indirir: satır sonları ve yinelenen boşluklar tek boşluk olur.

    Tahmin dosyası (JSONL) ve ERRANT-TR satır temellidir; çıktıda kalan bir
    satır sonu cümle sayısını kaydırır.
    """
    return squeeze_spaces(text.replace("\r", " ").replace("\n", " "))


def hit_length_limit(token_ids: Sequence[int], eos_token_id: int) -> bool:
    """Model cümleyi bitirmeden çıktı sınırına takıldı mı?

    Model yazmayı kendisi bitirdiğinde bitiş tokenını (EOS) üretir. Çıktıda EOS
    yoksa üretim max_new_tokens sınırında kesilmiştir, yani cümlenin sonu eksiktir.
    """
    return eos_token_id not in token_ids


def generation_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Bir çıkarım koşusunun özetini çıkarır: kesilen girdi ve çıktı sayısı, token sayıları."""
    n = len(rows)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "truncated_inputs": sum(1 for row in rows if row["truncated_input"]),
        "hit_length_limit": sum(1 for row in rows if row["hit_length_limit"]),
        "empty_outputs": sum(1 for row in rows if not row["hypothesis"]),
        "unchanged": sum(1 for row in rows if row["hypothesis"] == row["source"]),
        "input_tokens_max": max(row["input_tokens"] for row in rows),
        "output_tokens_max": max(row["output_tokens"] for row in rows),
        "input_tokens_mean": round(sum(row["input_tokens"] for row in rows) / n, 1),
        "output_tokens_mean": round(sum(row["output_tokens"] for row in rows) / n, 1),
    }
