"""Veri kayıtlarından eğitim ve çıkarım örnekleri.

Bir kayıt (data/processed/*.jsonl satırı) üç adımda modele verilecek metne döner:

1. Mesajlar: talimat + cümle (kullanıcı) ve beklenen cevap (asistan).
   GECTurk kayıtları "düzelt ve etiketle", diğerleri "yalnızca düzelt"
   talimatını alır (K9). Biçim trgec.formats içindedir (K8).
2. Metin: mesajlar modelin kendi sohbet şablonundan geçirilir. Şablonu olmayan
   modelde (ör. Kanarya) sabit ve basit bir şablon kullanılır.
3. İstem ve cevap ayrımı: metin ikiye bölünür. Eğitimde kayıp yalnızca cevap
   kısmında hesaplanır; çıkarımda modele yalnızca istem kısmı verilir. İkisi
   aynı işlevden geldiği için eğitim ve çıkarım aynı istemi görür.

Bu modül torch ya da transformers içe aktarmaz; tokenizer dışarıdan verilir.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence

from .convert import ORIGIN_GECTURK
from .formats import MODE_CORRECT, MODE_LABEL, Edit, build_messages

# Sohbet şablonu olmayan modeller için. '###' kullanılmaz: çıktı biçiminin ayracıdır (K8).
FALLBACK_PROMPT = "Kullanıcı: {user}\nAsistan:\n"


@dataclass(frozen=True)
class Rendered:
    """Bir örneğin modele verilecek metni.

    prompt:             modelin göreceği kısım (talimat, cümle, cevabın başladığı yer)
    answer:             modelin üretmesi beklenen kısım, bitiş işaretiyle birlikte;
                        çıkarım girdisinde None
    add_special_tokens: tokenizer'a bu metin verilirken özel token eklensin mi?
                        Sohbet şablonu başlangıç tokenını kendisi yazar; o yüzden
                        şablonlu metinde False olur.
    """

    prompt: str
    answer: str | None
    add_special_tokens: bool


def record_mode(record: dict[str, Any]) -> str:
    """Kaydın eğitimde alacağı talimat: GECTurk etiketli, diğerleri etiketsiz (K9)."""
    return MODE_LABEL if record["origin"] == ORIGIN_GECTURK else MODE_CORRECT


def record_edits(record: dict[str, Any]) -> list[Edit]:
    """Kaydın düzeltme listesi (yalnızca GECTurk kayıtlarında dolu)."""
    return [Edit(edit["source"], edit["target"], edit["rule"]) for edit in record.get("extra", {}).get("edits", [])]


def training_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    """Kayıttan eğitim örneği: kullanıcı mesajı ve beklenen asistan cevabı."""
    return build_messages(record["source"], record_mode(record), corrected=record["target"], edits=record_edits(record))


def inference_messages(sentence: str, mode: str = MODE_LABEL) -> list[dict[str, str]]:
    """Çıkarım girdisi: yalnızca kullanıcı mesajı. Varsayılan talimat "düzelt ve etiketle" (K9)."""
    return build_messages(sentence, mode)


def render(tokenizer: Any, messages: Sequence[dict[str, str]]) -> Rendered:
    """Mesajları modelin sohbet şablonuyla metne çevirir ve istem ile cevabı ayırır.

    İstem, şablonun "şimdi asistan konuşacak" işaretiyle biter
    (add_generation_prompt). Qwen3'te düşünme modu kapatılır (enable_thinking;
    diğer şablonlar bu ayarı yok sayar). Cevap, tam metinden istem çıkarılarak
    bulunur; böylece şablonun cevaptan sonra yazdığı bitiş işareti de cevaba
    dahil olur ve model nerede duracağını öğrenir. Bitiş işareti modelden modele
    değişir ('<|im_end|>', '<|eot_id|>') ve her zaman tokenizer'ın EOS'u değildir.

    İki durumda cevap, içerik + EOS olarak kurulur: tam metin istemle
    başlamıyorsa (şablon cevabı farklı sarıyorsa) ve şablon cevaptan sonra
    hiçbir bitiş işareti yazmıyorsa (Cosmos GPT-2: yalnızca satır sonu). İkinci
    durumda EOS eklenmezse model durmayı öğrenemez ve yazmaya devam eder.
    """
    messages = list(messages)
    user = messages[0]["content"]
    answer = messages[1]["content"] if len(messages) > 1 else None
    eos = getattr(tokenizer, "eos_token", None) or ""

    if not getattr(tokenizer, "chat_template", None):
        prompt = FALLBACK_PROMPT.format(user=user)
        return Rendered(prompt, None if answer is None else answer + eos, add_special_tokens=True)

    prompt = tokenizer.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if answer is None:
        return Rendered(prompt, None, add_special_tokens=False)
    full = tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)
    rendered_answer = full[len(prompt) :] if full.startswith(prompt) else ""
    if rendered_answer.strip() == answer.strip() or not rendered_answer:
        rendered_answer = answer + eos
    return Rendered(prompt, rendered_answer, add_special_tokens=False)


_PROBE = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]


def stop_text(tokenizer: Any) -> str:
    """Bu modelde cevabın bittiğini gösteren işaret (ör. '<|im_end|>').

    Çıkarımda üretim bu işarette durdurulur. Eğitim örneklerinin cevabı da aynı
    işaretle bittiği için ikisi tutarlıdır. İşaret, deneme amaçlı kısa bir
    cevabın sonuna şablonun ne eklediğine bakılarak bulunur.
    """
    answer = render(tokenizer, _PROBE).answer or ""
    return answer[len(_PROBE[1]["content"]) :].strip()


def mix_records(
    gecturk: Sequence[dict[str, Any]],
    oscar: Sequence[dict[str, Any]],
    limit_per_source: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Eğitim kümesini kurar: iki kaynağı birleştirir ve karıştırır.

    D1 için oscar boş verilir. limit_per_source verilirse her kaynaktan yalnızca
    ilk o kadar kayıt alınır (pilot mini eğitimler; alt küme dosyaları zaten
    karıştırılmış olduğu için ilk N kayıt rastgele bir örnektir). Karıştırma
    tohumludur: aynı girdi hep aynı sırayı verir.
    """
    chosen = list(gecturk[:limit_per_source]) + list(oscar[:limit_per_source])
    random.Random(seed).shuffle(chosen)
    return chosen


def drop_over_length(items: Sequence[Any], lengths: Sequence[int], ceiling: int) -> tuple[list[Any], list[Any]]:
    """Token sayısı tavanı aşan örnekleri ayırır: (kalanlar, elenenler).

    Örnek kırpılmaz, elenir (K11): kırpılmış bir hedef modele yarım cevap vermeyi öğretir.
    """
    if len(items) != len(lengths):
        raise ValueError("items ve lengths aynı uzunlukta olmalı")
    kept = [item for item, n in zip(items, lengths) if n <= ceiling]
    dropped = [item for item, n in zip(items, lengths) if n > ceiling]
    return kept, dropped


IGNORE_INDEX = -100  # PyTorch'un çapraz entropi kaybı bu etiketi taşıyan konumları hesaba katmaz


def encode(tokenizer: Any, rendered: Sequence[Rendered]) -> list[dict[str, Any]]:
    """Eğitim örneklerini token dizisine çevirir ve kaybın hesaplanacağı konumları işaretler.

    Her örnek için:
        input_ids  istem tokenları + cevap tokenları
        labels     istem konumlarında IGNORE_INDEX, cevap konumlarında cevabın tokenları
        length     toplam token sayısı (benzer uzunlukları aynı gruba koymak için)

    Model her konumda bir sonraki tokenı tahmin eder. labels yalnızca cevap
    konumlarında dolu olduğu için model talimatı ya da hatalı cümleyi ezberlemeye
    değil, düzeltilmiş cevabı üretmeye göre cezalandırılır. İstem ve cevap ayrı
    ayrı tokenize edilir: çıkarımda da model istemi hazır alır ve cevabı kendi
    üretir; eğitimdeki sınır bununla birebir aynı olur.
    """
    if not rendered:
        return []
    if any(item.answer is None for item in rendered):
        raise ValueError("Eğitim örneğinde cevap olmalı")
    special = rendered[0].add_special_tokens
    prompts = tokenizer([item.prompt for item in rendered], add_special_tokens=special)["input_ids"]
    answers = tokenizer([item.answer for item in rendered], add_special_tokens=False)["input_ids"]
    return [
        {
            "input_ids": list(prompt) + list(answer),
            "labels": [IGNORE_INDEX] * len(prompt) + list(answer),
            "length": len(prompt) + len(answer),
        }
        for prompt, answer in zip(prompts, answers)
    ]


def pad_batch(features: Sequence[dict[str, Any]], pad_token_id: int) -> dict[str, list[list[int]]]:
    """Bir gruptaki örnekleri en uzununun boyuna kadar sağdan doldurur.

    attention_mask dolgu konumlarında 0'dır (model oraya bakmaz); labels dolgu
    konumlarında IGNORE_INDEX'tir (kayba girmez). Dolgu sağdadır: eğitimde
    cevap, istemin hemen ardından gelmelidir.
    """
    width = max(len(feature["input_ids"]) for feature in features)
    batch: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
    for feature in features:
        n, gap = len(feature["input_ids"]), width - len(feature["input_ids"])
        batch["input_ids"].append(list(feature["input_ids"]) + [pad_token_id] * gap)
        batch["attention_mask"].append([1] * n + [0] * gap)
        batch["labels"].append(list(feature["labels"]) + [IGNORE_INDEX] * gap)
    return batch
