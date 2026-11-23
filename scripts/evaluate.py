"""Değerlendirme: bir tahmin dosyasını ERRANT-TR ile skorlar; cümle bazlı sayıları ve güven aralığını yazar.

Görev tanımı: docs/gorevler/02_degerlendirme_hatti.md

Girdi:  tahmin dosyası (JSONL: {"id", "hypothesis"}; varsa "well_formed") ve setin adı
Çıktı:  outputs/eval/<calisma>/<set>.json   P, R, F0.5, %95 güven aralığı, aşırı düzeltme oranı,
                                            cümle başına (tp, fp, fn)
        outputs/eval/<calisma>/<set>.hyp.m2 hipotezin M2 hali

Kural (K15): her sette, o setin yayımlanmış skorunda kullanılan araç kullanılır.
Şu an hazır olanlar: oscar_test, tweets, movie_reviews, oscar_dev, oscar_dev_clean (hepsi ERRANT-TR).
Film yorumlarında ERRANT-TR sonucu mT5'in 28,1'i ile kıyaslanır; etiketleyicinin 84,2'si ve
gecturk_dev için M2 scorer gerekir, o henüz bağlanmadı.

--baseline ile tahmin dosyası olmadan iki akıl sağlığı koşusu yapılır:
    source  hiçbir şeyi değiştirmeyen model (recall 0 olmalı)
    gold    kusursuz model (F0.5 = 1 olmalı)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

from trgec import errant_tr
from trgec.config import load_config
from trgec.convert import read_jsonl
from trgec.evaluation import align_predictions, clean_sentence_stats, summarize, tweet_protocol

# Set adı -> data/processed altındaki dosya ve (varsa) data/raw altındaki resmi M2 dosyası.
TESTSETS = {
    "oscar_test": {"file": "oscar_test.jsonl", "official_m2": "oscar_gec_test/oscar_gec_m2_W_MAPPED_LABELS.txt"},
    "tweets": {"file": "tweets.jsonl", "official_m2": None},
    "movie_reviews": {"file": "movie_reviews.jsonl", "official_m2": None},
    "oscar_dev": {"file": "oscar_dev.jsonl", "official_m2": None},
    "oscar_dev_clean": {"file": "oscar_dev_clean.jsonl", "official_m2": None},
}
NOT_READY = ("gecturk_dev",)
PROGRESS_EVERY = 300


def to_m2(annotator: Any, sources: Sequence[str], texts: Sequence[str], label: str) -> tuple[str, list[int]]:
    """Cümle çiftlerini tek tek M2'ye çevirir; (M2 metni, hata veren satır numaraları) döndürür.

    ERRANT-TR bir cümlede hata verirse (ör. boş ya da bozuk model çıktısı) o cümle
    "değişiklik yok" sayılır: bütün altın düzeltmeler kaçırılmış olur, yanlış
    pozitif eklenmez. Böyle satırların sayısı sonuçla birlikte raporlanır.
    """
    blocks, failed, start = [], [], time.time()
    for number, (source, text) in enumerate(zip(sources, texts), start=1):
        try:
            if not text.strip():
                raise ValueError("boş hipotez")
            block = errant_tr.parallel_to_m2(annotator, [source], [text])
        except Exception:  # noqa: BLE001  (üçüncü taraf araç her türlü hata verebilir)
            failed.append(number)
            block = errant_tr.parallel_to_m2(annotator, [source], [source])
        blocks.append(block.strip("\n"))
        if number % PROGRESS_EVERY == 0:
            print(f"  {label}: {number}/{len(sources)} cümle, {time.time() - start:.0f} sn", flush=True)
    return errant_tr.map_labels("\n\n".join(blocks) + "\n\n"), failed


def reference_m2(config: dict, testset: str, annotator: Any, sources: Sequence[str], golds: Sequence[str]) -> tuple[str, str]:
    """Referans M2'yi döndürür: (M2 metni, nereden geldiği).

    Resmi M2 dosyası olan sette o dosya kullanılır. Olmayan sette referans,
    kaynak ve altın cümlelerden ERRANT-TR ile üretilir ve outputs/eval/_ref
    altında saklanır; dosya adı altın cümlelerin özetini taşır, böylece
    dönüşüm ayarı değişince eski dosya kullanılmaz.
    """
    official = TESTSETS[testset]["official_m2"]
    if official:
        path = Path(config["paths"]["raw"]) / official
        return path.open(encoding="utf-8", newline="").read(), str(path)
    digest = hashlib.sha256("\n".join(list(sources) + list(golds)).encode("utf-8")).hexdigest()[:12]
    path = Path(config["paths"]["outputs"]) / "eval" / "_ref" / f"{testset}-{digest}.m2"
    if path.exists():
        return path.read_text(encoding="utf-8"), f"{path} (önceden üretilmiş)"
    m2, failed = to_m2(annotator, sources, golds, "referans")
    if failed:
        raise RuntimeError(f"Referans üretilirken {len(failed)} cümlede hata: {failed[:10]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(m2, encoding="utf-8", newline="\n")
    return m2, f"{path} (şimdi üretildi)"


def evaluate(config: dict, testset: str, predictions: list[dict], run_name: str) -> dict:
    records = read_jsonl(Path(config["paths"]["processed"]) / TESTSETS[testset]["file"])
    aligned = align_predictions(records, predictions)
    sources = [record["source"] for record in records]
    golds = [record["target"] for record in records]
    hypotheses = [prediction["hypothesis"] for prediction in aligned]

    protocol: dict[str, Any] = {}
    if testset == "tweets":
        protocol = dict(config["eval"]["tweets"])
        golds, hypotheses = tweet_protocol(golds, hypotheses, protocol["capitalize"], protocol["keep_punct"])

    errant_dir = config["eval"]["errant_tr_dir"]
    annotator = errant_tr.load_annotator(errant_dir)
    ref_m2, ref_origin = reference_m2(config, testset, annotator, sources, golds)
    hyp_m2, failed = to_m2(annotator, sources, hypotheses, "hipotez")

    result: dict[str, Any] = {
        "testset": testset,
        "run": run_name,
        "n": len(records),
        "scorer": f"ERRANT-TR @ {errant_tr.PINNED_COMMIT[:7]}",
        "reference": ref_origin,
        "protocol": protocol,
        "beta": config["eval"]["beta"],
        "primary_mode": config["eval"]["errant_tr_mode"],  # raporlanan ana skorun modu (K15)
        "scorer_failures": len(failed),
        "modes": {},
        "sentence_counts": {},
        "ids": [record["id"] for record in records],
    }
    for mode in (errant_tr.MODE_SPAN, errant_tr.MODE_SPAN_AND_TYPE):
        counts = errant_tr.sentence_counts(hyp_m2, ref_m2, mode=mode, beta=config["eval"]["beta"], errant_dir=errant_dir)
        result["modes"][mode] = summarize(counts, beta=config["eval"]["beta"], n_boot=config["eval"]["n_boot"], seed=config["seed"])
        result["sentence_counts"][mode] = counts
    result.update(clean_sentence_stats(sources, golds, hypotheses))
    flags = [prediction["well_formed"] for prediction in aligned if "well_formed" in prediction]
    result["malformed_rate"] = round(1 - sum(flags) / len(flags), 4) if flags else None

    out_dir = Path(config["paths"]["outputs"]) / "eval" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{testset}.hyp.m2").write_text(hyp_m2, encoding="utf-8", newline="\n")
    (out_dir / f"{testset}.json").write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return result


def baseline_predictions(config: dict, testset: str, kind: str) -> list[dict]:
    """Akıl sağlığı koşuları için tahmin üretir: kaynak cümle ya da altın cümle."""
    records = read_jsonl(Path(config["paths"]["processed"]) / TESTSETS[testset]["file"])
    if kind == "source":
        return [{"id": r["id"], "hypothesis": r["source"]} for r in records]
    golds = [r["target"] for r in records]
    if testset == "tweets":  # kusursuz model, protokolün beklediği gibi büyük harfle başlar
        golds, _ = tweet_protocol(golds, [], config["eval"]["tweets"]["capitalize"], config["eval"]["tweets"]["keep_punct"])
    return [{"id": r["id"], "hypothesis": gold} for r, gold in zip(records, golds)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions", help="tahmin dosyası (JSONL)")
    source.add_argument("--baseline", choices=["source", "gold"], help="tahmin dosyası yerine akıl sağlığı koşusu")
    parser.add_argument("--testset", required=True, choices=list(TESTSETS) + list(NOT_READY))
    parser.add_argument("--run-name", help="çıktı klasörünün adı (varsayılan: tahmin dosyasının klasörü)")
    parser.add_argument(
        "--hypothesis-field",
        default="hypothesis",
        help="tahmin dosyasında hipotezin okunacağı alan (ör. raw_output); çıkarımı yinelemeden başka bir çözme biçimini skorlamak için",
    )
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    if args.testset in NOT_READY:
        raise NotImplementedError(f"'{args.testset}' M2 scorer ile skorlanır (K15); M2 scorer henüz bağlanmadı.")
    config = load_config(args.config)

    if args.baseline:
        predictions = baseline_predictions(config, args.testset, args.baseline)
        run_name = args.run_name or f"baseline_{args.baseline}"
    else:
        predictions = read_jsonl(args.predictions)
        run_name = args.run_name or Path(args.predictions).resolve().parent.name
        if args.hypothesis_field != "hypothesis":
            predictions = [{**row, "hypothesis": row[args.hypothesis_field]} for row in predictions]
            run_name = args.run_name or f"{run_name}_{args.hypothesis_field}"

    result = evaluate(config, args.testset, predictions, run_name)
    print(f"\n{args.testset} | {run_name} | {result['n']} cümle | referans: {result['reference']}")
    for mode, summary in result["modes"].items():
        mark = "*" if mode == result["primary_mode"] else " "
        print(
            f" {mark}{mode:3s}: TP {summary['tp']} FP {summary['fp']} FN {summary['fn']} | "
            f"P {summary['precision']:.4f} R {summary['recall']:.4f} F{result['beta']} {summary['f_beta']:.4f} "
            f"[{summary['f_beta_low']:.4f}, {summary['f_beta_high']:.4f}]"
        )
    print(f"  (* ana skorun modu: {result['primary_mode']})")
    print(
        f"  hatasız cümle: {result['clean_sentences']} | aşırı düzeltme oranı: {result['overcorrection_rate']} | "
        f"bozuk çıktı oranı: {result['malformed_rate']} | skorlayıcının hata verdiği cümle: {result['scorer_failures']}"
    )


if __name__ == "__main__":
    main()
