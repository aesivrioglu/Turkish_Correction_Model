"""Token uzunluğu ölçümü: aday modellerin tokenizer'ıyla tam eğitim örneklerinin uzunluğu.

Görev tanımı: docs/gorevler/03_pilot_ve_model_secimi.md, adım 3 (K11).

Her eğitim kaydı gerçek eğitim örneğine çevrilir (talimat + cümle + hedef,
modelin sohbet şablonuyla) ve token sayısı ölçülür. Model indirilmez; yalnızca
tokenizer dosyaları okunur. Çıktı: her aday için ortalama, %95 ve %99'luk
dilimler, en uzun örnek, tavanı (max_len_ceiling) aşan örnek sayısı ve K11'in
yöntemiyle önerilen max_len.

Çıktı dosyası: outputs/lengths/<calisma>.json

Örnek:
    python scripts/measure_lengths.py --config configs/d2_gecturk_oscar.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from trgec.config import load_config
from trgec.convert import read_jsonl
from trgec.examples import mix_records, render, stop_text, training_messages
from trgec.lengths import suggest_max_len, truncation_rate


def load_training_records(config: dict) -> list[dict]:
    """Yapılandırmadaki karışıma (mix) göre eğitim kayıtlarını okur: D1 yalnızca GECTurk, D2 ikisi."""
    processed = Path(config["paths"]["processed"])
    mix = config["mix"]
    gecturk = read_jsonl(processed / "gecturk_train_12k.jsonl") if mix.get("gecturk") else []
    oscar = read_jsonl(processed / "oscar_train_12k.jsonl") if mix.get("oscar") else []
    return mix_records(gecturk, oscar, mix.get("limit_per_source"), seed=config["seed"])


def quantile(sorted_values: list[int], q: float) -> int:
    return sorted_values[min(len(sorted_values) - 1, max(0, round(q * len(sorted_values)) - 1))]


def describe(lengths: list[int], ceiling: int) -> dict[str, Any]:
    ordered = sorted(lengths)
    return {
        "n": len(ordered),
        "mean": round(sum(ordered) / len(ordered), 1),
        "p95": quantile(ordered, 0.95),
        "p99": quantile(ordered, 0.99),
        "max": ordered[-1],
        "over_384": sum(1 for n in ordered if n > 384),
        "over_ceiling": sum(1 for n in ordered if n > ceiling),
    }


def measure(model_id: str, records: list[dict], ceiling: int) -> dict[str, Any]:
    """Bir adayın tokenizer'ıyla bütün eğitim örneklerini ölçer."""
    from transformers import AutoTokenizer  # yalnızca burada gerekir

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    rendered = [render(tokenizer, training_messages(record)) for record in records]
    special = rendered[0].add_special_tokens
    lengths = [len(ids) for ids in tokenizer([r.prompt + r.answer for r in rendered], add_special_tokens=special)["input_ids"]]
    answer_lengths = [len(ids) for ids in tokenizer([r.answer for r in rendered], add_special_tokens=False)["input_ids"]]
    words = sum(len(record["source"].split()) for record in records)
    source_tokens = sum(len(ids) for ids in tokenizer([record["source"] for record in records], add_special_tokens=False)["input_ids"])

    result: dict[str, Any] = {
        "model": model_id,
        "has_chat_template": bool(tokenizer.chat_template),
        "eos_token": tokenizer.eos_token,
        "stop_text": stop_text(tokenizer),
        "stop_token_ids": tokenizer(stop_text(tokenizer), add_special_tokens=False)["input_ids"],
        "tokens_per_word": round(source_tokens / words, 2),
        "all": describe(lengths, ceiling),
        "answer": describe(answer_lengths, ceiling),
        "suggested_max_len": min(ceiling, suggest_max_len(lengths)),  # K11: %95'lik dilim, tavanı geçemez
        "truncation_rate_at_suggested": round(truncation_rate(lengths, min(ceiling, suggest_max_len(lengths))), 4),
        "example_prompt_tail": rendered[0].prompt[-60:],
        "example_answer": rendered[0].answer,
    }
    for origin in sorted({record["origin"] for record in records}):
        result[origin] = describe([n for n, record in zip(lengths, records) if record["origin"] == origin], ceiling)
    return result


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/d2_gecturk_oscar.yaml")
    parser.add_argument("--models", nargs="*", help="ölçülecek modeller (varsayılan: model.candidates)")
    args = parser.parse_args()

    config = load_config(args.config)
    ceiling = config["training"]["max_len_ceiling"]
    records = load_training_records(config)
    models = args.models or config["model"]["candidates"]
    print(f"{config['run_name']}: {len(records)} eğitim örneği, tavan {ceiling} token\n")

    results = []
    for model_id in models:
        result = measure(model_id, records, ceiling)
        results.append(result)
        stats = result["all"]
        print(
            f"{model_id}\n"
            f"  şablon: {'var' if result['has_chat_template'] else 'YOK (yedek şablon)'} | kelime başına token: {result['tokens_per_word']}\n"
            f"  ortalama {stats['mean']} | %95 {stats['p95']} | %99 {stats['p99']} | en uzun {stats['max']} | "
            f"384'ü aşan {stats['over_384']} | {ceiling}'yi aşan {stats['over_ceiling']}\n"
            f"  önerilen max_len (K11): {result['suggested_max_len']} (kesilecek oran {result['truncation_rate_at_suggested']})\n"
            f"  bitiş işareti: {result['stop_text']!r} (token: {result['stop_token_ids']}; tokenizer EOS'u {result['eos_token']!r})\n"
            f"  istemin sonu: {result['example_prompt_tail']!r}\n"
            f"  ilk cevap   : {result['example_answer']!r}\n",
            flush=True,
        )

    out = Path(config["paths"]["outputs"]) / "lengths" / f"{config['run_name']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": args.config, "n": len(records), "ceiling": ceiling, "results": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"yazıldı: {out}")


if __name__ == "__main__":
    main()
