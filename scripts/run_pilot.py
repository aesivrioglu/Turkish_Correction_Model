"""Pilot: aday modelleri sırayla kısa eğitimden geçirir ve doğrulama setlerinde tahmin üretir.

Görev tanımı: docs/gorevler/03_pilot_ve_model_secimi.md, adım 6.

Her aday için sırayla:
    1. scripts/train.py    pilot karışımıyla mini eğitim (her kaynaktan 1.000 örnek)
    2. scripts/predict.py  doğrulama setlerinde tahmin (pilot.eval_sets)
Her adım ayrı bir süreçte çalışır: bir aday bitince GPU belleği tamamen boşalır
ve bir adayın hatası diğerlerini durdurmaz. Hata veren aday özet dosyasına
yazılır, sıradaki adaya geçilir.

Skorlama burada yapılmaz: tahmin dosyaları indirilir ve yerelde
scripts/evaluate.py ile skorlanır (ERRANT-TR yerelde kurulu).

İkinci aşama pilot için üç seçenek:
    --modes label correct   tahmin iki talimatla da alınır ("düzelt ve etiketle" ile "yalnızca düzelt")
    --zero-shot             eğitim yapılmaz; temel modelin eğitimsiz halinden tahmin alınır (K21 c)
    --append                var olan özet dosyasına eklenir (aynı oturumda birden çok çağrı)
Özet dosyasında her tahmin koşusu (model x talimat) ayrı bir kayıttır.

Çıktı:  outputs/pilot_summary.json   her adayın durumu, süreleri, tepe belleği
        outputs/logs/<calisma>.log   her adayın tam günlüğü
        outputs/pilot_outputs.zip    tahminler, özetler ve günlükler (adaptör ağırlıkları hariç)

Yalnızca doğrulama setleri kullanılır; test setlerine dokunulmaz.

Örnek (yerelde, kodun çalıştığını görmek için):
    python scripts/run_pilot.py --config configs/pilot.yaml --models ufakai/ufakzeka-1 --smoke-test
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from trgec.config import adapter_run_name, load_config, prediction_run_name, zero_shot_run_name
from trgec.formats import MODE_CORRECT, MODE_LABEL
from trgec.pilot import halve_batch, is_out_of_memory

SMOKE_PREDICT_LIMIT = 4


def run_step(command: list[str], log_path: Path, env: dict[str, str]) -> tuple[bool, float, str]:
    """Bir betiği ayrı süreçte çalıştırır; çıktısını hem ekrana hem günlük dosyasına yazar.

    Döndürür: (başarılı mı, süre, çıktının son bölümü). Son bölüm, hatanın bellek
    taşması olup olmadığını anlamak için kullanılır.
    """
    start = time.time()
    tail: list[str] = []
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        log.write(f"\n$ {' '.join(command)}\n")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True, encoding="utf-8", errors="replace")
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            tail.append(line)
            del tail[:-200]
        code = process.wait()
    return code == 0, round(time.time() - start, 1), "".join(tail)


def train_with_fallback(base_command: list[str], batch_size: int, grad_accum: int, log_path: Path, env: dict[str, str]) -> dict:
    """Eğitimi çalıştırır; GPU belleği taşarsa grup boyutunu yarıya indirip yeniden dener.

    Uzunluğa göre gruplama en uzun grubu başa koyar, yani bellek yetmeyecekse
    eğitim ilk adımda durur; her deneme yalnızca bir dakika kadar sürer. Etkin
    grup boyutu (grup x biriktirme) hiç değişmez. Bellek dışı bir hata yeniden denenmez.
    """
    result: dict = {"train_ok": False, "train_seconds": 0.0, "attempts": []}
    setting: tuple[int, int] | None = (batch_size, grad_accum)
    while setting is not None:
        batch, accum = setting
        ok, seconds, tail = run_step(base_command + ["--batch-size", str(batch), "--grad-accum", str(accum)], log_path, env)
        out_of_memory = not ok and is_out_of_memory(tail)
        result["attempts"].append({"batch_size": batch, "grad_accum": accum, "ok": ok, "seconds": seconds, "out_of_memory": out_of_memory})
        result["train_seconds"] = round(result["train_seconds"] + seconds, 1)
        if ok:
            result.update(train_ok=True, batch_size=batch, grad_accum=accum)
            return result
        if not out_of_memory:
            return result
        setting = halve_batch(batch, accum)
        if setting:
            print(f"  GPU belleği yetmedi (grup {batch}); grup {setting[0]} x biriktirme {setting[1]} ile yeniden deneniyor", flush=True)
    return result


def clean_model_cache(model_id: str) -> None:
    """İndirilen model ağırlıklarını siler: sekiz adayın ağırlıkları diski doldurmasın."""
    from huggingface_hub import scan_cache_dir

    for repo in scan_cache_dir().repos:
        if repo.repo_id == model_id:
            hashes = [revision.commit_hash for revision in repo.revisions]
            scan_cache_dir().delete_revisions(*hashes).execute()
            print(f"  önbellek temizlendi: {model_id} ({repo.size_on_disk / 1024**3:.1f} GB)")


def pack_outputs(outputs: Path) -> Path:
    """İndirilecek dosyaları tek zip'e koyar. Adaptör ağırlıkları dahil edilmez (pilotta gerekmez, büyüktür)."""
    archive = outputs / "pilot_outputs.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        wanted = list((outputs / "predictions").rglob("*")) + list((outputs / "logs").rglob("*.log"))
        wanted += list((outputs / "adapters").glob("*/run_meta.json")) + [outputs / "pilot_summary.json"]
        for path in wanted:
            if path.is_file():
                bundle.write(path, path.relative_to(outputs).as_posix())
    return archive


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--models", nargs="*", help="çalıştırılacak adaylar (varsayılan: model.candidates)")
    parser.add_argument("--modes", nargs="+", default=[MODE_LABEL], choices=[MODE_LABEL, MODE_CORRECT], help="tahminde kullanılacak talimatlar (varsayılan: label)")
    parser.add_argument("--zero-shot", action="store_true", help="eğitim yapma; temel modelin eğitimsiz halinden tahmin al")
    parser.add_argument("--append", action="store_true", help="var olan pilot_summary.json dosyasına ekle (aynı adlı koşu yenisiyle değişir)")
    parser.add_argument("--smoke-test", action="store_true", help="yerel deneme: eğitim duman testi, her setten 4 cümle")
    parser.add_argument("--clean-cache", action="store_true", help="her adaydan sonra indirilen ağırlıkları sil (Kaggle'da disk için)")
    args = parser.parse_args()

    config = load_config(args.config)
    models = args.models or config["model"]["candidates"]
    eval_sets = config["pilot"]["eval_sets"]
    outputs = Path(config["paths"]["outputs"])
    (outputs / "logs").mkdir(parents=True, exist_ok=True)
    summary_path = outputs / "pilot_summary.json"

    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    # Kaggle iki T4 verir. Tek GPU görünür kılınır: 4-bit yüklenen model iki karta
    # otomatik bölünürse (DataParallel) eğitim bozulur.
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # bellek parçalanmasını azaltır

    summary: list[dict] = json.loads(summary_path.read_text(encoding="utf-8")) if args.append and summary_path.exists() else []
    new_entries: list[dict] = []
    for number, model_id in enumerate(models, start=1):
        print(f"\n===== [{number}/{len(models)}] {model_id}{' (eğitimsiz)' if args.zero_shot else ''} =====", flush=True)
        # Bu adayın bütün tahmin koşularında ortak olan alanlar (eğitim ölçümleri).
        shared: dict = {"model": model_id, "zero_shot": args.zero_shot, "adapter_run": None, "train_ok": False}

        if args.zero_shot:
            base_run = zero_shot_run_name(model_id, args.smoke_test)
            log_path = outputs / "logs" / f"{base_run}.log"
            model_args = ["--model", model_id]
            can_predict = True
        else:
            base_run = adapter_run_name(config["run_name"], model_id, args.smoke_test)
            log_path = outputs / "logs" / f"{base_run}.log"
            adapter_dir = outputs / "adapters" / base_run
            model_args = ["--adapter", str(adapter_dir)]
            shared["adapter_run"] = base_run
            train = [sys.executable, "scripts/train.py", "--config", args.config, "--model", model_id]
            if args.smoke_test:
                train.append("--smoke-test")
            shared.update(train_with_fallback(train, config["training"]["batch_size"], config["training"]["grad_accum"], log_path, env))
            can_predict = shared["train_ok"]
            if can_predict:
                meta = json.loads((adapter_dir / "run_meta.json").read_text(encoding="utf-8"))
                for key in ("steps", "seconds_per_step", "peak_gpu_memory_gb", "loss_first", "loss_last", "parameters_total", "parameters_trainable"):
                    shared[key] = meta.get(key)
                shared["dropped_over_max_len"] = meta["data"]["dropped_over_max_len"]
            else:
                print(f"  EĞİTİM BAŞARISIZ: {model_id}. Günlük: {log_path}", flush=True)

        # Eğitim başarısızsa tek kayıt yazılır; değilse her talimat için ayrı tahmin koşusu.
        for mode in args.modes if can_predict else args.modes[:1]:
            entry = {**shared, "run": prediction_run_name(base_run, mode), "mode": mode, "predict": {}}
            for name in eval_sets if can_predict else []:
                command = [sys.executable, "scripts/predict.py", "--config", args.config, "--testset", name, "--mode", mode, "--run-name", entry["run"], *model_args]
                if args.smoke_test:
                    command += ["--limit", str(SMOKE_PREDICT_LIMIT)]
                ok, seconds, _tail = run_step(command, log_path, env)
                entry["predict"][name] = {"ok": ok, "seconds": seconds}
            new_entries.append(entry)
            summary = [old for old in summary if old["run"] != entry["run"]] + [entry]
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        if args.clean_cache:
            clean_model_cache(model_id)

    archive = pack_outputs(outputs)
    print("\n===== özet =====")
    for entry in new_entries:
        predicted = sum(1 for result in entry["predict"].values() if result["ok"])
        training = "eğitimsiz" if entry["zero_shot"] else f"eğitim {'tamam' if entry['train_ok'] else 'BAŞARISIZ'} ({entry['train_seconds']} sn)"
        print(
            f"{entry['run']}: {training} | tahmin {predicted}/{len(eval_sets)} set | grup {entry.get('batch_size')} x {entry.get('grad_accum')} | "
            f"sn/adım {entry.get('seconds_per_step')} | tepe bellek {entry.get('peak_gpu_memory_gb')} GB"
        )
    print(f"\nindirilecek dosya: {archive}")
    if not all(entry["zero_shot"] or entry["train_ok"] for entry in new_entries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
