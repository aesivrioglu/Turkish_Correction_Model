"""Temel model çıkarımı: yayımlanmış bir seq2seq modelin (OSCAR GEC mT5) tahminlerini JSONL olarak yazar.

Görev tanımı: docs/gorevler/02_degerlendirme_hatti.md (Kapı A)

Bu betik eğitim yapmaz; yayımlanmış ağırlıkları indirip cümleleri düzelttirir.
Amaç kendi değerlendirme hattımızın yayımlanmış skoru (tweetlerde 79,0) yeniden
üretip üretmediğini görmektir. Kendi modellerimizin çıkarımı scripts/predict.py
içinde olacak (görev 03).

Çıktı:  outputs/predictions/<calisma>/<set>.jsonl
            {"id", "source", "hypothesis", "raw_output", "input_tokens",
             "output_tokens", "truncated_input", "hit_length_limit"}
        outputs/predictions/<calisma>/<set>.meta.json   koşunun bütün ayarları ve özeti

Üretim ayarları (ışın sayısı, çıktı uzunluğu) yazarlarca yayımlanmadı. Bu yüzden
varsayılanları yoktur: ya yapılandırmada (baselines.oscar_mt5) ya da komut
satırında açıkça verilmelidir; her koşunun ayarı meta dosyasına yazılır.

Örnek (yerelde, CPU):
    python scripts/predict_seq2seq.py --testset oscar_dev --limit 5 --num-beams 1 --max-new-tokens 256
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from trgec.config import load_config
from trgec.convert import read_jsonl, write_jsonl
from trgec.seq2seq import clean_output, generation_summary, hit_length_limit, length_sorted_batches

BASELINE_KEY = "oscar_mt5"


def resolve_settings(config: dict, args: argparse.Namespace) -> dict[str, Any]:
    """Komut satırı ile yapılandırmayı birleştirir; komut satırı önceliklidir."""
    baseline = config.get("baselines", {}).get(BASELINE_KEY, {})
    settings = {
        "model": args.model or baseline.get("hf_id"),
        "revision": args.revision or baseline.get("revision"),
        "num_beams": args.num_beams if args.num_beams is not None else baseline.get("num_beams"),
        "max_new_tokens": args.max_new_tokens if args.max_new_tokens is not None else baseline.get("max_new_tokens"),
        "max_input_tokens": args.max_input_tokens,
        "prefix": args.prefix,
        "decode_cleanup": not args.no_cleanup,
        "batch_size": args.batch_size,
    }
    missing = [key for key in ("model", "num_beams", "max_new_tokens") if settings[key] is None]
    if missing:
        raise SystemExit(
            f"Eksik ayar: {', '.join(missing)}. Yayımlanmış değeri olmadığı için varsayılan yok; "
            "komut satırında ya da configs/base.yaml içinde (baselines.oscar_mt5) ver."
        )
    return settings


def load_model(model_id: str, revision: str | None, device: str):
    """Tokenizer'ı ve modeli yükler. Ağır kütüphaneler yalnızca burada içe aktarılır."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, revision=revision)
    model.to(device)
    model.eval()  # dropout kapanır; aynı girdi her zaman aynı çıktıyı verir
    return tokenizer, model, device


def predict(records: list[dict], tokenizer, model, device: str, settings: dict[str, Any]) -> list[dict]:
    """Kayıtların kaynak cümlelerini modele verir; kayıt sırasını koruyarak sonuçları döndürür."""
    import torch

    inputs = [settings["prefix"] + record["source"] for record in records]
    # Önce her cümlenin token sayısı ölçülür: hem gruplama hem de "girdi kesildi mi" sorusu için.
    token_counts = [len(ids) for ids in tokenizer(inputs)["input_ids"]]
    limit = settings["max_input_tokens"]
    eos_id, pad_id = tokenizer.eos_token_id, tokenizer.pad_token_id

    rows: list[dict | None] = [None] * len(records)
    batches = length_sorted_batches(token_counts, settings["batch_size"])
    start, done = time.time(), 0
    for batch in batches:
        encoded = tokenizer(
            [inputs[index] for index in batch],
            return_tensors="pt",
            padding=True,
            truncation=limit is not None,
            max_length=limit,
        ).to(device)
        with torch.inference_mode():  # gradyan tutulmaz: daha az bellek, daha hızlı
            generated = model.generate(
                **encoded,
                num_beams=settings["num_beams"],
                max_new_tokens=settings["max_new_tokens"],
                do_sample=False,  # rastgele örnekleme yok; sonuç yinelenebilir
            )
        # Çözme iki biçimde yapılır. "Temizleme" (tokenizer'ın varsayılanı) noktalamadan önceki
        # boşluğu siler: "%40'ının , orgazmla" -> "%40'ının, orgazmla". Ham çıktı da saklanır ki
        # hangisinin yayımlanmış skoru verdiği çıkarım yinelenmeden denenebilsin.
        cleaned = tokenizer.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        raw = tokenizer.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        chosen = cleaned if settings["decode_cleanup"] else raw
        for index, ids, text, raw_text in zip(batch, generated.tolist(), chosen, raw):
            rows[index] = {
                "id": records[index]["id"],
                "source": records[index]["source"],
                "hypothesis": clean_output(text),
                "raw_output": clean_output(raw_text),
                "input_tokens": token_counts[index],
                "output_tokens": sum(1 for token in ids if token != pad_id),
                "truncated_input": limit is not None and token_counts[index] > limit,
                "hit_length_limit": hit_length_limit(ids, eos_id),
            }
        done += len(batch)
        print(f"  {done}/{len(records)} cümle, {time.time() - start:.0f} sn", flush=True)
    return [row for row in rows if row is not None]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--testset", required=True, help="data/processed altındaki setin adı (ör. tweets, oscar_dev)")
    parser.add_argument("--run-name", help="çıktı klasörünün adı (varsayılan: ayarlardan üretilir)")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--model", help="Hugging Face kimliği (varsayılan: baselines.oscar_mt5.hf_id)")
    parser.add_argument("--revision", help="modelin sabit sürümü (commit)")
    parser.add_argument("--num-beams", type=int, help="ışın sayısı; 1 = açgözlü kod çözme")
    parser.add_argument("--max-new-tokens", type=int, help="üretilecek en fazla token")
    parser.add_argument("--max-input-tokens", type=int, help="girdi bu sayıdan uzunsa kesilir (varsayılan: kesme yok)")
    parser.add_argument("--prefix", default="", help="her girdinin başına eklenecek görev öneki (varsayılan: yok)")
    parser.add_argument("--no-cleanup", action="store_true", help="çözmede noktalama önündeki boşlukları silme (hipotez = ham çıktı)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--limit", type=int, help="yalnızca ilk N cümle (deneme için; değerlendirmeye verilemez)")
    args = parser.parse_args()

    config = load_config(args.config)
    settings = resolve_settings(config, args)
    data_path = Path(config["paths"]["processed"]) / f"{args.testset}.jsonl"
    if not data_path.exists():
        raise SystemExit(f"{data_path} yok. Önce: python scripts/prepare_data.py --step convert (ya da sample)")
    records = read_jsonl(data_path)
    if args.limit:
        records = records[: args.limit]

    run_name = args.run_name or f"mt5_b{settings['num_beams']}_n{settings['max_new_tokens']}"
    if args.limit and not args.run_name:
        run_name += f"_ilk{args.limit}"
    print(f"{args.testset}: {len(records)} cümle | {settings['model']} | ayarlar: {settings}")

    tokenizer, model, device = load_model(settings["model"], settings["revision"], args.device)
    start = time.time()
    rows = predict(records, tokenizer, model, device, settings)
    seconds = time.time() - start

    import torch
    import transformers

    out_dir = Path(config["paths"]["outputs"]) / "predictions" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(rows, out_dir / f"{args.testset}.jsonl")
    meta = {
        "testset": args.testset,
        "run": run_name,
        "partial": bool(args.limit),
        **settings,
        "device": device,
        "seconds": round(seconds, 1),
        "sentences_per_second": round(len(rows) / seconds, 2) if seconds else None,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "summary": generation_summary(rows),
    }
    (out_dir / f"{args.testset}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    summary = meta["summary"]
    print(f"\nyazıldı: {out_dir / (args.testset + '.jsonl')} | {seconds:.0f} sn ({meta['sentences_per_second']} cümle/sn, {device})")
    print(
        f"  çıktı sınırına takılan: {summary['hit_length_limit']} | kesilen girdi: {summary['truncated_inputs']} | "
        f"boş çıktı: {summary['empty_outputs']} | değişmeyen: {summary['unchanged']} | "
        f"en uzun girdi/çıktı: {summary['input_tokens_max']}/{summary['output_tokens_max']} token"
    )
    if summary["hit_length_limit"]:
        print("  UYARI: sınıra takılan cümlelerin sonu eksik; --max-new-tokens değerini büyüt.")


if __name__ == "__main__":
    main()
