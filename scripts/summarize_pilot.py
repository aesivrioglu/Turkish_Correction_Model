"""Pilot özeti: adayların tahminlerini skorlar ve karşılaştırma tablosunu üretir.

Görev tanımı: docs/gorevler/03_pilot_ve_model_secimi.md, adım 6 ve 7.

Girdi:  Kaggle'dan indirilen pilot_outputs.zip'in açıldığı klasör
        (pilot_summary.json, predictions/<calisma>/<set>.jsonl, adapters/<calisma>/run_meta.json)
İş:     her aday ve her doğrulama seti için scripts/evaluate.py çalıştırılır
        (sonuç zaten varsa yeniden hesaplanmaz), sonra sayılar tek tabloda toplanır.
Çıktı:  outputs/pilot/pilot_table.md ve pilot_table.json

Yalnızca doğrulama setleri skorlanır. Tablo bir öneri değildir; seçimi kullanıcı yapar (Kapı B).

Örnek:
    python scripts/summarize_pilot.py --pilot-dir outputs/kaggle/pilot/unzipped
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from trgec.config import load_config
from trgec.pilot import STATUS_NAN, STATUS_NO_LEARNING, STATUS_OK, training_status


def percent(value: float | None, digits: int = 1) -> str:
    # Küçük ekleme: 19,45 gibi tam buçuklu değerler ikili gösterim yüzünden aşağı yuvarlanmasın.
    return "" if value is None else f"{100 * value + 1e-9:.{digits}f}".replace(".", ",")


def number(value: float | None, digits: int = 2) -> str:
    return "" if value is None else f"{value:.{digits}f}".replace(".", ",")


def ensure_evaluated(config_path: str, outputs: Path, run: str, testset: str) -> dict[str, Any] | None:
    """Bir adayın bir setteki skorunu döndürür; yoksa evaluate.py ile hesaplatır."""
    predictions = outputs / "predictions" / run / f"{testset}.jsonl"
    result_path = outputs / "eval" / run / f"{testset}.json"
    if not predictions.exists():
        return None
    if not result_path.exists() or result_path.stat().st_mtime < predictions.stat().st_mtime:
        command = [sys.executable, "scripts/evaluate.py", "--config", config_path, "--predictions", str(predictions), "--testset", testset, "--run-name", run]
        print(f"skorlanıyor: {run} / {testset}", flush=True)
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if completed.returncode != 0:
            print(completed.stdout[-600:], completed.stderr[-600:])
            return None
    return json.loads(result_path.read_text(encoding="utf-8"))


def collect(config: dict, config_path: str, pilot_dir: Path) -> list[dict[str, Any]]:
    """İndirilen dosyaları outputs/ altına kopyalar, skorlar ve aday başına bir satır üretir."""
    outputs = Path(config["paths"]["outputs"])
    summary = json.loads((pilot_dir / "pilot_summary.json").read_text(encoding="utf-8"))
    eval_sets = config["pilot"]["eval_sets"]
    rows = []
    for entry in summary:
        run = entry["run"]
        source = pilot_dir / "predictions" / run
        if source.exists():
            shutil.copytree(source, outputs / "predictions" / run, dirs_exist_ok=True)
        row: dict[str, Any] = {key: entry.get(key) for key in (
            "model", "run", "train_ok", "batch_size", "grad_accum", "steps", "seconds_per_step", "train_seconds",
            "peak_gpu_memory_gb", "loss_first", "loss_last", "parameters_total", "parameters_trainable", "dropped_over_max_len",
        )}
        row["oom_retries"] = sum(1 for attempt in entry.get("attempts", []) if attempt.get("out_of_memory"))
        # Eğitim yarıda durduysa (NaN) özet dosyasında sayı olmaz; adayın kendi run_meta.json dosyasına bakılır.
        meta_path = pilot_dir / "adapters" / run / "run_meta.json"
        run_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        row["nan_loss_at_step"] = run_meta.get("nan_loss_at_step")
        if row["batch_size"] is None and entry.get("attempts"):
            row["batch_size"], row["grad_accum"] = entry["attempts"][-1]["batch_size"], entry["attempts"][-1]["grad_accum"]
        row["status"] = training_status(entry["train_ok"], row["loss_first"], row["loss_last"], row["nan_loss_at_step"])
        row["sets"] = {}
        for testset in eval_sets:
            # Öğrenmemiş bir adayın tahminleri anlamsızdır; skorlanmaz.
            result = ensure_evaluated(config_path, outputs, run, testset) if row["status"] == STATUS_OK else None
            meta_path = outputs / "predictions" / run / f"{testset}.meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            if result is None:
                row["sets"][testset] = None
                continue
            scores = result["modes"][result["primary_mode"]]
            row["sets"][testset] = {
                "n": result["n"],
                "precision": scores["precision"],
                "recall": scores["recall"],
                "f_beta": scores["f_beta"],
                "f_beta_low": scores["f_beta_low"],
                "f_beta_high": scores["f_beta_high"],
                "malformed_rate": result["malformed_rate"],
                "overcorrection_rate": result["overcorrection_rate"],
                "clean_sentences": result["clean_sentences"],
                "sentences_per_second": meta.get("sentences_per_second"),
                "hit_length_limit": meta.get("summary", {}).get("hit_length_limit"),
            }
        rows.append(row)
    return rows


def render_table(rows: list[dict[str, Any]], eval_sets: list[str]) -> str:
    """Karşılaştırma tablolarını Markdown olarak üretir."""
    lines = ["## Eğitim (aynı 2.000 örnek, 1 tur, etkin grup 16)", ""]
    lines += ["| Aday | Parametre (4-bit sayım) | Grup x biriktirme | Adım | sn/adım | Eğitim (sn) | Tepe bellek (GB) | Kayıp ilk → son | Durum |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        params = f"{row['parameters_total'] / 1e9:.2f}B".replace(".", ",") if row["parameters_total"] else ""
        group = f"{row['batch_size']} x {row['grad_accum']}" if row["batch_size"] else ""
        loss = f"{number(row['loss_first'])} → {number(row['loss_last'])}" if row["loss_first"] is not None else ""
        status = {
            STATUS_OK: "tamam",
            STATUS_NAN: f"BAŞARISIZ: kayıp NaN (adım {row['nan_loss_at_step']})",
            STATUS_NO_LEARNING: "ÖĞRENMEDİ: kayıp düşmedi",
        }.get(row["status"], "BAŞARISIZ")
        lines.append(
            f"| `{row['model']}` | {params} | {group} | {row['steps'] or ''} | {number(row['seconds_per_step'], 3)} | "
            f"{number(row['train_seconds'], 0)} | {number(row['peak_gpu_memory_gb'])} | {loss} | {status} |"
        )
    for testset in eval_sets:
        lines += ["", f"## `{testset}` (ERRANT-TR)", ""]
        lines += ["| Aday | P | R | F0.5 | %95 güven aralığı | Bozuk çıktı (%) | Hatasız cümlede aşırı düzeltme (%) | Sınıra takılan | Cümle/sn |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        ranked = sorted(rows, key=lambda row: -(row["sets"].get(testset) or {}).get("f_beta", -1))
        for row in ranked:
            scores = row["sets"].get(testset)
            if not scores:
                lines.append(f"| `{row['model']}` | skorlanmadı | | | | | | | |")
                continue
            has_errors = scores["clean_sentences"] < scores["n"]
            prf = (percent(scores["precision"]), percent(scores["recall"]), percent(scores["f_beta"])) if has_errors else ("", "", "")
            interval = f"{percent(scores['f_beta_low'])}–{percent(scores['f_beta_high'])}" if has_errors else ""
            lines.append(
                f"| `{row['model']}` | {prf[0]} | {prf[1]} | {prf[2]} | {interval} | {percent(scores['malformed_rate'])} | "
                f"{percent(scores['overcorrection_rate']) if scores['clean_sentences'] else ''} | {scores['hit_length_limit']} | {number(scores['sentences_per_second'])} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pilot-dir", required=True, help="pilot_outputs.zip'in açıldığı klasör")
    parser.add_argument("--config", default="configs/pilot.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    rows = collect(config, args.config, Path(args.pilot_dir))
    table = render_table(rows, config["pilot"]["eval_sets"])
    out_dir = Path(config["paths"]["outputs"]) / "pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pilot_table.md").write_text(table, encoding="utf-8", newline="\n")
    (out_dir / "pilot_table.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(table)
    print(f"yazıldı: {out_dir / 'pilot_table.md'}")


if __name__ == "__main__":
    main()
