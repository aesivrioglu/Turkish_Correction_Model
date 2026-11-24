"""Çıkarım: ince ayarlı bir modelin (temel model + LoRA adaptörü) tahminlerini JSONL olarak yazar.

Görev tanımı: docs/gorevler/03_pilot_ve_model_secimi.md, adım 5.

Her cümle için model, eğitimde gördüğü istemin aynısını alır ("düzelt ve
etiketle" talimatı, K9) ve cevabı kendi üretir. Cevap trgec.formats.parse_output
ile düzeltilmiş cümleye ve düzeltme listesine ayrılır; biçim bozuksa
well_formed False olur ve bozuk çıktı oranına girer. Emniyet kemeri (K14)
yapılandırmada eşik varsa uygulanır; kemersiz çıktı da saklanır.

--mode correct ile model "yalnızca düzelt" talimatını alır: cevapta liste
beklenmez, ilk satır düzeltilmiş cümle sayılır. Tahminler '<calisma>__correct'
klasörüne yazılır. --adapter verilmezse temel model eğitimsiz haliyle çalışır
(klasör: 'zeroshot__<model>'); bu, eğitimin ne kattığını ölçmek içindir (K21).

Çıktı:  outputs/predictions/<calisma>/<set>.jsonl
            {"id", "source", "raw_output", "hypothesis", "hypothesis_no_guard",
             "edits", "well_formed", "guard_triggered", "hit_length_limit",
             "input_tokens", "output_tokens"}
        outputs/predictions/<calisma>/<set>.meta.json

Dosya doğrudan scripts/evaluate.py'ye verilebilir.

Örnek:
    python scripts/predict.py --testset oscar_dev --adapter outputs/adapters/pilot__Kumru-2B
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from trgec.config import load_config, prediction_run_name, zero_shot_run_name
from trgec.convert import read_jsonl, write_jsonl
from trgec.examples import inference_messages, render, stop_text, use_plain_prompt
from trgec.formats import MODE_CORRECT, MODE_LABEL, parse_output
from trgec.guard import apply_guard
from trgec.modeling import restore_tied_output_layer
from trgec.seq2seq import clean_output, length_sorted_batches, output_budget


def base_model_of(adapter_dir: Path) -> str:
    """Adaptörün hangi temel model için eğitildiğini adaptörün kendi ayar dosyasından okur."""
    return json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))["base_model_name_or_path"]


def load_model(model_id: str, adapter_dir: Path | None, device: str, load_in_4bit: bool, plain_prompt_models: list[str] | None = None):
    """Temel modeli yükler, varsa adaptörü üstüne takar.

    GPU'da model eğitimdeki gibi 4-bit yüklenir; CPU'da (yerel deneme) fp32.
    Tokenizer adaptör klasöründen okunur: eğitimde kullanılan sohbet şablonu orada kayıtlıdır.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir) if adapter_dir else model_id)
    use_plain_prompt(tokenizer, model_id, plain_prompt_models)  # eğitimdeki istemle aynı olsun
    if device == "cuda":
        from transformers import BitsAndBytesConfig

        quantization = None
        if load_in_4bit:
            quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization, dtype=torch.float16, device_map={"": 0})
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    # Eğitimdeki onarımın aynısı: çıktı katmanı boş yüklenen modelde (Mecellem) gömme katmanına bağlanır.
    restore_tied_output_layer(model)
    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    return tokenizer, model


def predict(records: list[dict], tokenizer, model, device: str, settings: dict[str, Any]) -> list[dict]:
    """Kayıtların kaynak cümlelerini düzelttirir; kayıt sırasını koruyarak sonuçları döndürür."""
    import torch

    mode = settings["mode"]
    prompts = [render(tokenizer, inference_messages(record["source"], mode)) for record in records]
    special = prompts[0].add_special_tokens
    token_counts = [len(ids) for ids in tokenizer([p.prompt for p in prompts], add_special_tokens=special)["input_ids"]]
    source_tokens = [len(ids) for ids in tokenizer([record["source"] for record in records], add_special_tokens=False)["input_ids"]]

    # Üretim, cevabın bittiğini gösteren işarette durur. Bu işaret her zaman tokenizer'ın
    # EOS'u değildir (rapor 03, bölüm 11); ikisi de durdurucu olarak verilir.
    stop_ids = tokenizer(stop_text(tokenizer), add_special_tokens=False)["input_ids"]
    stop_tokens = {stop_ids[0]} if len(stop_ids) == 1 else set()
    if tokenizer.eos_token_id is not None:
        stop_tokens.add(tokenizer.eos_token_id)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    # Dolgu solda: üretim istemin hemen ardından başlamalı, dolgunun ardından değil.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = pad_token_id

    rows: list[dict | None] = [None] * len(records)
    start, done = time.time(), 0
    for batch in length_sorted_batches(token_counts, settings["batch_size"]):
        encoded = tokenizer([prompts[i].prompt for i in batch], add_special_tokens=special, return_tensors="pt", padding=True).to(device)
        # Çıktı sınırı gruptaki en uzun cümleye göre belirlenir (trgec.seq2seq.output_budget).
        budget = output_budget(
            max(source_tokens[i] for i in batch),
            settings["max_new_tokens"],
            settings["max_new_tokens_per_source_token"],
            settings["max_new_tokens_margin"],
        )
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=budget,
                do_sample=False,  # açgözlü kod çözme: aynı girdi hep aynı çıktıyı verir
                num_beams=1,
                eos_token_id=sorted(stop_tokens),
                pad_token_id=pad_token_id,
            )
        new_tokens = generated[:, encoded["input_ids"].shape[1] :].tolist()
        for index, ids in zip(batch, new_tokens):
            source = records[index]["source"]
            produced = [token for token in ids if token != pad_token_id or token in stop_tokens]
            raw = tokenizer.decode(ids, skip_special_tokens=True).strip()
            # "Yalnızca düzelt" talimatında liste beklenmez; ayraç yoksa ilk satır düzeltilmiş cümledir.
            parsed = parse_output(raw, expect_edits=mode == MODE_LABEL)
            corrected = clean_output(parsed.corrected)
            hypothesis, guard_triggered = corrected, False
            if settings["max_change_ratio"] is not None:
                hypothesis, guard_triggered = apply_guard(source, corrected, settings["max_change_ratio"])
            rows[index] = {
                "id": records[index]["id"],
                "source": source,
                "raw_output": raw,
                "hypothesis": hypothesis,
                "hypothesis_no_guard": corrected,
                "edits": [{"source": e.source, "target": e.target, "rule": e.rule} for e in parsed.edits],
                "well_formed": parsed.well_formed,
                "guard_triggered": guard_triggered,
                "hit_length_limit": not any(token in stop_tokens for token in ids),
                "input_tokens": token_counts[index],
                "output_tokens": len(produced),
            }
        done += len(batch)
        print(f"  {done}/{len(records)} cümle, {time.time() - start:.0f} sn", flush=True)
    return [row for row in rows if row is not None]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--testset", help="data/processed altındaki setin adı (ör. oscar_dev, gecturk_dev)")
    source.add_argument("--records-file", help="ortak biçimde herhangi bir JSONL dosyası (deneme için)")
    parser.add_argument("--adapter", help="LoRA adaptörünün klasörü (verilmezse temel model olduğu gibi kullanılır)")
    parser.add_argument("--model", help="temel modelin Hugging Face kimliği (varsayılan: adaptörün kayıtlı temel modeli)")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--mode", default=MODE_LABEL, choices=[MODE_LABEL, MODE_CORRECT], help="talimat: label = düzelt ve etiketle (varsayılan, K9), correct = yalnızca düzelt")
    parser.add_argument("--run-name", help="çıktı klasörünün adı (varsayılan: adaptör klasörünün adı; --mode correct ise sonuna '__correct')")
    parser.add_argument("--max-new-tokens", type=int, help="üretilecek en fazla token (varsayılan: inference.max_new_tokens)")
    parser.add_argument("--batch-size", type=int, help="varsayılan: inference.batch_size")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--limit", type=int, help="yalnızca ilk N cümle (deneme için; değerlendirmeye verilemez)")
    args = parser.parse_args()

    import torch

    config = load_config(args.config)
    adapter_dir = Path(args.adapter) if args.adapter else None
    model_id = args.model or (base_model_of(adapter_dir) if adapter_dir else None)
    if not model_id:
        raise SystemExit("Model belli değil: --adapter ya da --model ver.")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    settings = {
        "mode": args.mode,
        "max_new_tokens": args.max_new_tokens or config["inference"]["max_new_tokens"],
        "max_new_tokens_per_source_token": config["inference"]["max_new_tokens_per_source_token"],
        "max_new_tokens_margin": config["inference"]["max_new_tokens_margin"],
        "batch_size": args.batch_size or config["inference"]["batch_size"],
        "max_change_ratio": config["guard"]["max_change_ratio"],
    }

    if args.testset:
        name, path = args.testset, Path(config["paths"]["processed"]) / f"{args.testset}.jsonl"
    else:
        path = Path(args.records_file)
        name = path.stem
    records = read_jsonl(path)
    if args.limit:
        records = records[: args.limit]
    run_name = args.run_name or prediction_run_name(adapter_dir.name if adapter_dir else zero_shot_run_name(model_id), args.mode)
    print(f"{name}: {len(records)} cümle | {model_id} | adaptör: {adapter_dir} | {device} | {settings}")

    tokenizer, model = load_model(model_id, adapter_dir, device, config["training"]["load_in_4bit"], config["model"].get("plain_prompt"))
    start = time.time()
    rows = predict(records, tokenizer, model, device, settings)
    seconds = time.time() - start

    out_dir = Path(config["paths"]["outputs"]) / "predictions" / run_name
    write_jsonl(rows, out_dir / f"{name}.jsonl")
    n = len(rows)
    summary = {
        "n": n,
        "malformed": sum(1 for row in rows if not row["well_formed"]),
        "hit_length_limit": sum(1 for row in rows if row["hit_length_limit"]),
        "empty": sum(1 for row in rows if not row["hypothesis_no_guard"]),
        "unchanged": sum(1 for row in rows if row["hypothesis"] == row["source"]),
        "guard_triggered": sum(1 for row in rows if row["guard_triggered"]),
        "input_tokens_max": max(row["input_tokens"] for row in rows),
        "output_tokens_max": max(row["output_tokens"] for row in rows),
    }
    meta = {
        "testset": name,
        "run": run_name,
        "partial": bool(args.limit),
        "model": model_id,
        "adapter": str(adapter_dir) if adapter_dir else None,
        **settings,
        "device": device,
        "seconds": round(seconds, 1),
        "sentences_per_second": round(n / seconds, 2) if seconds else None,
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2) if device == "cuda" else None,
        "summary": summary,
    }
    (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nyazıldı: {out_dir / (name + '.jsonl')} | {seconds:.0f} sn ({meta['sentences_per_second']} cümle/sn)")
    print(
        f"  bozuk çıktı: {summary['malformed']} | sınıra takılan: {summary['hit_length_limit']} | boş: {summary['empty']} | "
        f"değişmeyen: {summary['unchanged']} | kemer devreye girdi: {summary['guard_triggered']}"
    )


if __name__ == "__main__":
    main()
