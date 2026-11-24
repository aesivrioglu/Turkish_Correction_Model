"""QLoRA eğitimi. Yerelde CPU üzerinde duman testi, Kaggle'da gerçek eğitim.

Görev tanımı: docs/gorevler/03_pilot_ve_model_secimi.md, adım 4.

Ne yapar:
    1. Yapılandırmadaki karışıma göre eğitim kayıtlarını okur (D1, D2 ya da pilot).
    2. Her kaydı modelin sohbet şablonuyla istem + cevaba çevirir (trgec.examples);
       token tavanını aşan örnekleri eler (kırpmaz, K11).
    3. Modeli 4-bit yükler (QLoRA) ve LoRA adaptörlerini ekler: ana ağırlıklar
       donuktur, yalnızca küçük adaptör matrisleri öğrenir.
    4. Eğitir; kayıp yalnızca cevap tokenlarında hesaplanır.
    5. Adaptörü ve koşunun özetini outputs/adapters/<calisma>/ altına yazar.

Bayraklar:
    --config      configs/d1_gecturk.yaml gibi bir yapılandırma
    --model       Hugging Face kimliği (verilmezse yapılandırmadaki model.chosen)
    --smoke-test  8 örnek, 2 adım, nicemleme ve fp16 kapalı, CPU. Amaç skor değil,
                  kodun baştan sona hatasız çalıştığını Kaggle'a gitmeden görmek.

Kural (CLAUDE.md): bu betik --smoke-test ile yerelde denenmeden Kaggle'a gönderilmez;
Kapı A geçilmeden Kaggle'da çalıştırılmaz (Kapı A 2026-11-23'te geçildi).

Örnek:
    python scripts/train.py --config configs/d1_gecturk.yaml --smoke-test
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from trgec.config import adapter_run_name, load_config
from trgec.convert import read_jsonl, write_jsonl
from trgec.examples import IGNORE_INDEX, drop_over_length, encode, mix_records, pad_batch, render, stop_text, training_messages, use_plain_prompt
from trgec.modeling import restore_tied_output_layer

SMOKE_EXAMPLES = 8
SMOKE_STEPS = 2
SMOKE_BATCH = 4


def load_training_records(config: dict) -> list[dict]:
    """Yapılandırmadaki karışıma (mix) göre eğitim kayıtlarını okur: D1 yalnızca GECTurk, D2 ikisi."""
    processed = Path(config["paths"]["processed"])
    mix = config["mix"]
    gecturk = read_jsonl(processed / "gecturk_train_12k.jsonl") if mix.get("gecturk") else []
    oscar = read_jsonl(processed / "oscar_train_12k.jsonl") if mix.get("oscar") else []
    return mix_records(gecturk, oscar, mix.get("limit_per_source"), seed=config["seed"])


def build_features(tokenizer: Any, records: list[dict], max_len: int) -> tuple[list[dict], dict[str, Any]]:
    """Kayıtları token dizilerine çevirir; tavanı aşanları eler. (örnekler, özet) döndürür."""
    features = encode(tokenizer, [render(tokenizer, training_messages(record)) for record in records])
    lengths = [feature["length"] for feature in features]
    kept, dropped = drop_over_length(features, lengths, max_len)
    answer_tokens = sum(sum(1 for label in feature["labels"] if label != IGNORE_INDEX) for feature in kept)
    stats = {
        "records": len(records),
        "kept": len(kept),
        "dropped_over_max_len": len(dropped),
        "max_len": max_len,
        "tokens_total": sum(feature["length"] for feature in kept),
        "tokens_in_loss": answer_tokens,
        "longest": max(feature["length"] for feature in kept),
    }
    return kept, stats


def load_model(model_id: str, training: dict, smoke: bool):
    """Modeli yükler ve LoRA adaptörlerini ekler.

    Gerçek eğitimde (Kaggle, T4): ağırlıklar 4-bit NF4 ile nicemlenir, hesap fp16
    yapılır. Bu, 2 milyar parametrelik bir modeli yaklaşık 2 GB belleğe sığdırır.
    Duman testinde: nicemleme yok (bitsandbytes GPU ister), fp32, CPU.

    LoRA: her seçilen katmanın büyük ağırlık matrisi W donuk kalır; yanına r
    satırlı iki küçük matris (A ve B) eklenir ve katman W + (alpha / r) * B.A gibi
    davranır. Yalnızca A ve B öğrenir; bu, parametrelerin yaklaşık %1'idir.
    """
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM

    if smoke:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
        restore_tied_output_layer(model)
    else:
        from transformers import BitsAndBytesConfig

        quantization = None
        if training["load_in_4bit"]:
            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,  # T4 bf16 desteklemez
                bnb_4bit_use_double_quant=True,
            )
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization, dtype=torch.float16, device_map={"": 0})
        if restore_tied_output_layer(model):
            print("  çıktı katmanı boş yüklendi; gömme katmanına bağlandı (trgec.modeling)", flush=True)
        if quantization is not None:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training["gradient_checkpointing"])

    lora = LoraConfig(
        r=training["lora_r"],
        lora_alpha=training["lora_alpha"],
        lora_dropout=training["lora_dropout"],
        target_modules=training["lora_target_modules"],
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora)


def count_parameters(model) -> tuple[int, int]:
    """(öğrenen parametre sayısı, toplam parametre sayısı)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def make_monitor():
    """Kaybı ve hızı izleyen geri çağırma: kayıp NaN ya da sonsuz olursa eğitimi durdurur."""
    import torch
    from transformers import TrainerCallback

    class Monitor(TrainerCallback):
        def __init__(self) -> None:
            self.losses: list[tuple[int, float]] = []
            self.bad_loss_at: int | None = None
            self.started: float | None = None
            self.seconds = 0.0

        def on_train_begin(self, args, state, control, **kwargs):
            self.started = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

        def on_log(self, args, state, control, logs=None, **kwargs):
            loss = (logs or {}).get("loss")
            if loss is None:
                return
            self.losses.append((state.global_step, float(loss)))
            elapsed = time.time() - (self.started or time.time())
            print(f"  adım {state.global_step}/{state.max_steps} | kayıp {loss:.4f} | {elapsed:.0f} sn", flush=True)
            if not math.isfinite(loss):
                # fp16'da taşma kaybı NaN yapabilir; devam etmek GPU saatini boşa harcar.
                self.bad_loss_at = state.global_step
                control.should_training_stop = True

        def on_train_end(self, args, state, control, **kwargs):
            self.seconds = time.time() - (self.started or time.time())

    return Monitor()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", help="Hugging Face kimliği (varsayılan: model.chosen; duman testinde training.smoke_model)")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=SMOKE_STEPS, help="duman testinde adım sayısı; 30 verilirse kaybın düştüğü görülür")
    parser.add_argument("--batch-size", type=int, help="grup boyutu (varsayılan: training.batch_size)")
    parser.add_argument("--grad-accum", type=int, help="gradyan biriktirme adımı (varsayılan: training.grad_accum)")
    parser.add_argument("--resume", action="store_true", help="çıktı klasöründeki son kayıt noktasından devam et")
    parser.add_argument("--push-to-hub", metavar="DEPO", help="bitince adaptörü bu özel Hugging Face deposuna yükle (HF_TOKEN gerekir)")
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, Trainer, TrainingArguments

    config = load_config(args.config)
    training, smoke = config["training"], args.smoke_test
    model_id = args.model or (training["smoke_model"] if smoke else config["model"]["chosen"])
    if not model_id:
        raise SystemExit("Model belli değil: --model ver ya da Kapı B'de model.chosen doldurulsun.")
    if not smoke and not torch.cuda.is_available():
        raise SystemExit("GPU yok. Yerelde yalnızca --smoke-test çalıştırılır; gerçek eğitim Kaggle'da (CLAUDE.md).")
    max_len = training["max_len"] or training["max_len_ceiling"]
    run_name = adapter_run_name(config["run_name"], model_id, smoke)
    out_dir = Path(config["paths"]["outputs"]) / "adapters" / run_name
    torch.manual_seed(config["seed"])

    # 1-2. Veri
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if use_plain_prompt(tokenizer, model_id, config["model"].get("plain_prompt")):
        print("  sohbet şablonu kapatıldı: düz istem, cevap EOS ile biter (model.plain_prompt)", flush=True)
    records = load_training_records(config)
    if smoke:
        records = records[:SMOKE_EXAMPLES]
        write_jsonl(records, out_dir / "smoke_records.jsonl")  # predict.py bu örneklerle denenebilsin
    features, data_stats = build_features(tokenizer, records, max_len)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    stop = stop_text(tokenizer)
    print(f"{run_name}: {data_stats['kept']} örnek ({data_stats['dropped_over_max_len']} tanesi {max_len} tokenı aştığı için elendi)")
    print(f"  toplam {data_stats['tokens_total']} token; kayıp hesaplanan (cevap) {data_stats['tokens_in_loss']} token; bitiş işareti {stop!r}")
    first = features[0]
    print("  ilk örnekte kaybın hesaplandığı kısım:")
    print("    " + repr(tokenizer.decode([label for label in first["labels"] if label != IGNORE_INDEX])))

    # 3. Model
    model = load_model(model_id, training, smoke)
    trainable, total = count_parameters(model)
    print(f"  parametre: {total:,} toplam, {trainable:,} öğrenen (%{100 * trainable / total:.2f})")

    # 4. Eğitim
    batch_size = SMOKE_BATCH if smoke else (args.batch_size or training["batch_size"])
    grad_accum = 1 if smoke else (args.grad_accum or training["grad_accum"])
    print(f"  grup boyutu {batch_size} x biriktirme {grad_accum} = etkin {batch_size * grad_accum}")
    arguments = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=training["learning_rate"],
        num_train_epochs=training["epochs"],
        max_steps=args.smoke_steps if smoke else -1,
        warmup_steps=0 if smoke else training["warmup_ratio"],  # 1'den küçük değer toplam adımın oranı sayılır
        lr_scheduler_type="linear",
        weight_decay=training["weight_decay"],
        max_grad_norm=training["max_grad_norm"],
        optim="adamw_torch",
        fp16=training["fp16"] and not smoke,
        logging_steps=1 if smoke else training["logging_steps"],
        logging_nan_inf_filter=False,  # NaN kayıp gizlenmesin; Monitor görüp durdursun
        save_strategy="no" if smoke else "steps",
        save_steps=training["save_steps"],
        save_total_limit=2,
        train_sampling_strategy="group_by_length",  # benzer uzunluklar aynı grupta: daha az dolgu (rapor 03, bölüm 1.2)
        report_to="none",
        disable_tqdm=True,  # ilerleme çubuğu yerine satır satır kayıt: Kaggle günlüğünde okunaklı
        seed=config["seed"],
        use_cpu=smoke,
        dataloader_pin_memory=not smoke,
    )
    monitor = make_monitor()
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=Dataset.from_list(features),
        data_collator=lambda batch: {key: torch.tensor(value) for key, value in pad_batch(batch, pad_token_id).items()},
        callbacks=[monitor],
    )
    trainer.train(resume_from_checkpoint=args.resume or None)

    # 5. Kayıt
    steps = trainer.state.global_step
    peak_gb = round(torch.cuda.max_memory_allocated() / 1024**3, 2) if torch.cuda.is_available() else None
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    meta = {
        "run": run_name,
        "config": args.config,
        "model": model_id,
        "smoke_test": smoke,
        "data": data_stats,
        "stop_text": stop,
        "parameters_total": total,
        "parameters_trainable": trainable,
        "batch_size": batch_size,
        "grad_accum": arguments.gradient_accumulation_steps,
        "learning_rate": training["learning_rate"],
        "lora": {key: training[key] for key in ("lora_r", "lora_alpha", "lora_dropout", "lora_target_modules")},
        "load_in_4bit": bool(training["load_in_4bit"]) and not smoke,
        "fp16": arguments.fp16,
        "steps": steps,
        "seconds": round(monitor.seconds, 1),
        "seconds_per_step": round(monitor.seconds / steps, 3) if steps else None,
        "peak_gpu_memory_gb": peak_gb,
        "loss_first": round(monitor.losses[0][1], 4) if monitor.losses else None,
        "loss_last": round(monitor.losses[-1][1], 4) if monitor.losses else None,
        "losses": monitor.losses,
        "nan_loss_at_step": monitor.bad_loss_at,
        "torch": torch.__version__,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    if monitor.bad_loss_at is not None:
        raise SystemExit(f"Kayıp {monitor.bad_loss_at}. adımda NaN ya da sonsuz oldu; eğitim durduruldu. Özet: {out_dir / 'run_meta.json'}")
    print(f"\nbitti: {steps} adım, {monitor.seconds:.0f} sn ({meta['seconds_per_step']} sn/adım) | tepe GPU belleği: {peak_gb} GB")
    print(f"  kayıp: ilk {meta['loss_first']}, son {meta['loss_last']}")
    print(f"  adaptör: {out_dir}")
    if args.push_to_hub and not smoke:
        model.push_to_hub(args.push_to_hub, private=True)
        tokenizer.push_to_hub(args.push_to_hub, private=True)
        print(f"  Hugging Face'e yüklendi (özel depo): {args.push_to_hub}")


if __name__ == "__main__":
    main()
