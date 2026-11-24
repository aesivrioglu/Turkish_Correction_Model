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

from trgec.config import adapter_run_name, load_config

SMOKE_PREDICT_LIMIT = 4


def run_step(command: list[str], log_path: Path, env: dict[str, str]) -> tuple[bool, float]:
    """Bir betiği ayrı süreçte çalıştırır; çıktısını hem ekrana hem günlük dosyasına yazar."""
    start = time.time()
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        log.write(f"\n$ {' '.join(command)}\n")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True, encoding="utf-8", errors="replace")
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        code = process.wait()
    return code == 0, round(time.time() - start, 1)


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

    summary: list[dict] = []
    for number, model_id in enumerate(models, start=1):
        run_name = adapter_run_name(config["run_name"], model_id, args.smoke_test)
        log_path = outputs / "logs" / f"{run_name}.log"
        adapter_dir = outputs / "adapters" / run_name
        entry: dict = {"model": model_id, "run": run_name, "train_ok": False, "predict": {}}
        print(f"\n===== [{number}/{len(models)}] {model_id} =====", flush=True)

        train = [sys.executable, "scripts/train.py", "--config", args.config, "--model", model_id]
        if args.smoke_test:
            train.append("--smoke-test")
        entry["train_ok"], entry["train_seconds"] = run_step(train, log_path, env)

        if entry["train_ok"]:
            meta = json.loads((adapter_dir / "run_meta.json").read_text(encoding="utf-8"))
            for key in ("steps", "seconds_per_step", "peak_gpu_memory_gb", "loss_first", "loss_last", "parameters_total", "parameters_trainable"):
                entry[key] = meta.get(key)
            entry["dropped_over_max_len"] = meta["data"]["dropped_over_max_len"]
            for name in eval_sets:
                command = [sys.executable, "scripts/predict.py", "--config", args.config, "--testset", name, "--adapter", str(adapter_dir)]
                if args.smoke_test:
                    command += ["--limit", str(SMOKE_PREDICT_LIMIT)]
                ok, seconds = run_step(command, log_path, env)
                entry["predict"][name] = {"ok": ok, "seconds": seconds}
        else:
            print(f"  EĞİTİM BAŞARISIZ: {model_id}. Günlük: {log_path}", flush=True)

        summary.append(entry)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        if args.clean_cache:
            clean_model_cache(model_id)

    archive = pack_outputs(outputs)
    print("\n===== özet =====")
    for entry in summary:
        predicted = sum(1 for result in entry["predict"].values() if result["ok"])
        print(
            f"{entry['model']}: eğitim {'tamam' if entry['train_ok'] else 'BAŞARISIZ'} ({entry['train_seconds']} sn) | "
            f"tahmin {predicted}/{len(eval_sets)} set | sn/adım {entry.get('seconds_per_step')} | tepe bellek {entry.get('peak_gpu_memory_gb')} GB"
        )
    print(f"\nindirilecek dosya: {archive}")
    if not all(entry["train_ok"] for entry in summary):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
