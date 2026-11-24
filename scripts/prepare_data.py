"""Veri hazırlığı: ortak biçime çevirme, sızıntı taraması, örnekleme.

Görev tanımı: docs/gorevler/01_veri_hazirligi.md

Çıktı sözleşmesi (data/processed/<ad>.jsonl, satır başına bir örnek):
    {"id": str, "source": str, "target": str, "rules": [str, ...],
     "origin": "gecturk" | "oscar" | "tweets", "split": str, "extra": {...}}

Adımlar:
    convert  OSCAR testini, tweetleri ve GECTurk'ün dört M2 dosyasını (train,
             dev, test, film yorumları) ortak biçime çevirir.
    sample   GECTurk train'den katmanlı 12 binlik alt kümeyi; OSCAR eğitim
             verisinden oscar_dev, oscar_dev_clean ve eğitim alt kümesini seçer.
             Yinelemeler ve test ya da doğrulama setleriyle çakışanlar elenir.
             OSCAR akışından seçilen havuzlar data/processed/pools altında
             saklanır; oran gibi ayarlar değişince veri yeniden okunmaz.
    leakage  Üretilmiş dosyalar arasındaki bütün eğitim-test çiftlerini yeniden
             tarar ve sayıları raporlar; çakışma varsa hata verir.
İşin mantığı src/trgec altında; bu betik yalnızca dosya okur, yazar ve özet basar.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterator

from trgec.config import load_config
from trgec.convert import (
    convert_gecturk,
    convert_oscar_test,
    convert_tweets,
    oscar_clean_record,
    oscar_row_problem,
    oscar_train_record,
    read_jsonl,
    write_jsonl,
)
from trgec.leakage import dedupe_records, match_keys, split_by_overlap
from trgec.sampling import Reservoir, mix_with_clean, rule_counts, stratified_sample

# data/raw altındaki dosya adları (kaynak ve sürümler: docs/raporlar/01_tokenizasyon.md)
OSCAR_TEST_DIR = "oscar_gec_test"
OSCAR_TEST_INP = "inp.txt"
OSCAR_TEST_REF = "ref.txt"
OSCAR_TEST_M2 = "oscar_gec_m2_W_MAPPED_LABELS.txt"
TWEETS_FILE = "tweets/data.txt"
GECTURK_DIR = "gecturk"
# Çıktı adı -> (ayrım adı, M2 dosyası, hedef dosyası). Film yorumlarının hedef dosyası yok.
GECTURK_FILES = {
    "gecturk_train": ("train", "gec_tr/gec_tr_source_annot_train.txt", "gec_tr/gec_tr_target_train.txt"),
    "gecturk_dev": ("dev", "gec_tr/gec_tr_source_annot_val.txt", "gec_tr/gec_tr_target_val.txt"),
    "gecturk_test": ("test", "gec_tr/gec_tr_source_annot_test.txt", "gec_tr/gec_tr_target_test.txt"),
    "movie_reviews": ("movie_reviews", "movie_reviews_test_sentences.txt", None),
}
# Sızıntı taramasında eğitim tarafı ve karşısındaki setler (görev 01, adım 4).
LEAKAGE_PAIRS = {
    "gecturk_train_12k": ("gecturk_dev", "gecturk_test", "movie_reviews", "oscar_test", "tweets"),
    "oscar_train_12k": ("oscar_dev", "oscar_dev_clean", "movie_reviews", "oscar_test", "tweets", "gecturk_dev", "gecturk_test"),
    "oscar_dev": ("movie_reviews", "oscar_test", "tweets"),
    "oscar_dev_clean": ("movie_reviews", "oscar_test", "tweets"),
    "gecturk_dev": ("movie_reviews", "oscar_test", "tweets"),
    "gecturk_dev_2k": ("gecturk_train_12k", "movie_reviews", "oscar_test", "tweets"),
}
OSCAR_TRAIN_FILE = "train.csv"

NOT_WRITTEN = ("download",)
PROGRESS_EVERY = 250_000


def _read(path: Path) -> str:
    """Dosyayı satır sonlarını değiştirmeden okur (newline='' evrensel çeviriyi kapatır)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} yok. Kaynaklar: docs/raporlar/01_tokenizasyon.md")
    with path.open(encoding="utf-8", newline="") as handle:
        return handle.read()


def _summary(name: str, records: list[dict], out: Path) -> None:
    unchanged = sum(1 for r in records if r["source"] == r["target"])
    print(f"{name}: {len(records)} örnek, {unchanged} tanesinde kaynak ve hedef aynı -> {out}")


def convert(config: dict) -> None:
    """Ham dosyaları ortak JSONL biçimine çevirir. Ham dosyalara dokunmaz."""
    raw = Path(config["paths"]["raw"])
    processed = Path(config["paths"]["processed"])

    oscar_dir = raw / OSCAR_TEST_DIR
    records = convert_oscar_test(
        _read(oscar_dir / OSCAR_TEST_INP),
        _read(oscar_dir / OSCAR_TEST_REF),
        _read(oscar_dir / OSCAR_TEST_M2),
    )
    out = processed / "oscar_test.jsonl"
    write_jsonl(records, out)
    _summary("OSCAR GEC test", records, out)

    records = convert_tweets(_read(raw / TWEETS_FILE))
    out = processed / "tweets.jsonl"
    write_jsonl(records, out)
    _summary("#Turki$hTweets", records, out)

    for name, (split, m2_file, target_file) in GECTURK_FILES.items():
        target_text = _read(raw / GECTURK_DIR / target_file) if target_file else None
        records = convert_gecturk(_read(raw / GECTURK_DIR / m2_file), split, target_text)
        out = processed / f"{name}.jsonl"
        write_jsonl(records, out)
        _summary(f"GECTurk {split}", records, out)

    # GECTurk dev doğrulama setimizdir. Üç gerçek test setinden biriyle aynı olan cümle
    # doğrulama tarafından çıkarılır; test setleri değiştirilmez (görev 01, adım 4).
    out = processed / "gecturk_dev.jsonl"
    dev = read_jsonl(out)
    for name in LEAKAGE_PAIRS["gecturk_dev"]:
        dev, leaked = split_by_overlap(dev, _keys_of(config, name))
        if leaked:
            print(f"GECTurk dev: {name} ile çakışan {len(leaked)} cümle çıkarıldı ({', '.join(r['id'] for r in leaked)})")
    write_jsonl(dev, out)
    _summary("GECTurk dev (çakışanlar çıkarıldıktan sonra)", dev, out)


def _oscar_train_rows(config: dict) -> Iterator[tuple[int, object, object]]:
    """OSCAR eğitim verisini akış halinde okur: (satır numarası, cümle, düzeltme).

    Veri diske kaydedilmez. Satır numarası 1'den başlar ve train.csv içindeki
    sırayı gösterir; sabit sürümle (train_revision) birlikte satırı tanımlar.
    """
    from datasets import load_dataset  # yalnızca bu adımda gerekir; içe aktarması yavaştır

    oscar = config["data"]["oscar_gec"]
    stream = load_dataset(
        oscar["hf_id_train"],
        data_files=OSCAR_TRAIN_FILE,
        split="train",
        streaming=True,
        revision=oscar["train_revision"],
    )
    for row_number, row in enumerate(stream, start=1):
        yield row_number, row.get("sentence"), row.get("correction")


def _keys_of(config: dict, name: str) -> set[str]:
    """Bir setin kaynak ve hedef cümlelerinin eşleştirme anahtarları."""
    path = Path(config["paths"]["processed"]) / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} yok. Önce '--step convert' (ya da 'sample') çalıştır.")
    records = read_jsonl(path)
    return match_keys([r["source"] for r in records] + [r["target"] for r in records])


def _test_side_keys(config: dict) -> dict[str, set[str]]:
    """OSCAR havuzlarının taranacağı test setlerinin anahtarları (kaynak ve hedef)."""
    return {name: _keys_of(config, name) for name in ("oscar_test", "tweets", "movie_reviews")}


def _quantile(sorted_values: list[int], q: float) -> int:
    return sorted_values[min(len(sorted_values) - 1, math.ceil(q * len(sorted_values)) - 1)]


def _describe(records: list[dict], max_words: int) -> dict:
    words = sorted(len(r["source"].split()) for r in records)
    clean = sum(1 for r in records if r["source"] == r["target"])
    return {
        "n": len(records),
        "clean": clean,
        "clean_ratio": round(clean / len(records), 4),
        "words_mean": round(sum(words) / len(words), 1),
        "words_p50": _quantile(words, 0.50),
        "words_p95": _quantile(words, 0.95),
        "words_max": words[-1],
        "target_over_limit": sum(1 for r in records if len(r["target"].split()) > max_words),
    }


def _pool_signature(config: dict) -> dict:
    """Havuzların hangi ayarlarla üretildiği. Değişirse veri yeniden okunur."""
    cfg = config["sampling"]
    return {
        "source": config["data"]["oscar_gec"]["hf_id_train"],
        "revision": config["data"]["oscar_gec"]["train_revision"],
        "seed": config["seed"],
        "max_words": cfg["oscar_max_words"],
        "error_pool_k": cfg["oscar_dev_size"] + cfg["oscar_n"] + cfg["oscar_reserve"],
        "clean_pool_k": cfg["oscar_clean_pool"],
    }


def _build_oscar_pools(config: dict) -> tuple[list[dict], list[dict], dict]:
    """OSCAR eğitim verisini bir kez okur ve iki havuz üretir.

    Hatalı havuz: uygun satırlardan rastgele örnek (kaynak -> hedef).
    Hatasız havuz: aynı akıştan, farklı tohumla bağımsız ikinci bir rastgele
    örnek; bu satırların yalnızca düzeltilmiş hali kullanılır (hedef -> hedef).
    İki havuzdan da yinelemeler ve test setleriyle çakışanlar atılır; hatasız
    havuzdan ayrıca hatalı havuzda geçen satırlar ve cümleler atılır.
    """
    signature = _pool_signature(config)
    max_words, seed = signature["max_words"], signature["seed"]
    counts: Counter = Counter()
    errors, clean = Reservoir(signature["error_pool_k"], seed), Reservoir(signature["clean_pool_k"], seed + 1)

    print(f"OSCAR eğitim verisi okunuyor ({signature['source']})...", flush=True)
    for row_number, sentence, correction in _oscar_train_rows(config):
        counts["rows"] += 1
        if counts["rows"] % PROGRESS_EVERY == 0:
            print(f"  {counts['rows']:,} satır okundu, {counts['eligible']:,} uygun", flush=True)
        problem = oscar_row_problem(sentence, correction, max_words)
        if problem:
            counts[problem] += 1
            continue
        counts["eligible"] += 1
        if sentence.strip() == correction.strip():
            counts["eligible_clean"] += 1
        errors.offer((row_number, sentence, correction))
        clean.offer((row_number, sentence, correction))

    test_keys = _test_side_keys(config)
    meta: dict = {"signature": signature, "stream": dict(counts)}

    error_pool = [oscar_train_record(row, sentence, correction, "train") for row, sentence, correction in errors.result()]
    error_pool, meta["error_duplicates_dropped"] = dedupe_records(error_pool)
    meta["error_leaked_dropped"] = {}
    for name, keys in test_keys.items():
        error_pool, leaked = split_by_overlap(error_pool, keys)
        meta["error_leaked_dropped"][name] = len(leaked)

    error_rows = {record["extra"]["row"] for record in error_pool}
    error_keys = match_keys([r["source"] for r in error_pool] + [r["target"] for r in error_pool])
    clean_pool = [
        oscar_clean_record(row, correction, "train")
        for row, _sentence, correction in clean.result()
        if row not in error_rows
    ]
    meta["clean_rows_also_in_error_pool"] = signature["clean_pool_k"] - len(clean_pool)
    clean_pool, meta["clean_duplicates_dropped"] = dedupe_records(clean_pool)
    clean_pool, same_as_error = split_by_overlap(clean_pool, error_keys)
    meta["clean_same_text_as_error_pool"] = len(same_as_error)
    meta["clean_leaked_dropped"] = {}
    for name, keys in test_keys.items():
        clean_pool, leaked = split_by_overlap(clean_pool, keys)
        meta["clean_leaked_dropped"][name] = len(leaked)
    meta["error_pool"], meta["clean_pool"] = len(error_pool), len(clean_pool)
    return error_pool, clean_pool, meta


def _load_or_build_oscar_pools(config: dict, refresh: bool) -> tuple[list[dict], list[dict], dict]:
    """Havuzlar aynı ayarlarla daha önce üretildiyse diskten okur, yoksa veriyi yeniden okur."""
    pools = Path(config["paths"]["processed"]) / "pools"
    error_path, clean_path, meta_path = pools / "oscar_error_pool.jsonl", pools / "oscar_clean_pool.jsonl", pools / "oscar_pool_meta.json"
    if not refresh and all(path.exists() for path in (error_path, clean_path, meta_path)):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("signature") == _pool_signature(config):
            print(f"Havuzlar diskten okundu: {pools}")
            return read_jsonl(error_path), read_jsonl(clean_path), meta
    error_pool, clean_pool, meta = _build_oscar_pools(config)
    write_jsonl(error_pool, error_path)
    write_jsonl(clean_pool, clean_path)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return error_pool, clean_pool, meta


def _write_subset(config: dict, name: str, subset: list[dict]) -> Path:
    """Alt kümeyi ve kimlik listesini yazar; kimlik listesi alt kümeyi yeniden üretilebilir kılar."""
    processed = Path(config["paths"]["processed"])
    out = processed / f"{name}.jsonl"
    write_jsonl(subset, out)
    ids = processed / "splits" / f"{name}_ids.txt"
    ids.parent.mkdir(parents=True, exist_ok=True)
    ids.write_text("\n".join(r["id"] for r in subset) + "\n", encoding="utf-8", newline="\n")
    return out


def sample_gecturk(config: dict) -> None:
    """GECTurk train'den katmanlı alt kümeyi üretir (D1 ve D2'de aynı alt küme kullanılır).

    Sıra: (1) kuralı olup cümlesi değişmeyen satırlar atılır, (2) aynı kaynak
    cümleyi taşıyan yinelemeler atılır, (3) GECTurk dev, GECTurk test, film
    yorumları, OSCAR testi ve tweetlerle çakışanlar atılır, (4) kalan havuzdan
    kural başına alt sınırlı örnek seçilir (K10).
    """
    cfg, seed = config["sampling"], config["seed"]
    processed = Path(config["paths"]["processed"])
    train = read_jsonl(processed / "gecturk_train.jsonl")
    stats: dict = {"seed": seed, "train": len(train), "leaked_dropped": {}}

    pool = [r for r in train if not (r["rules"] and r["source"] == r["target"])]
    stats["noop_edit_dropped"] = len(train) - len(pool)
    pool, stats["duplicates_dropped"] = dedupe_records(pool)
    for name in LEAKAGE_PAIRS["gecturk_train_12k"]:
        pool, leaked = split_by_overlap(pool, _keys_of(config, name))
        stats["leaked_dropped"][name] = len(leaked)
    stats["pool"] = len(pool)

    subset = stratified_sample(pool, cfg["gecturk_n"], cfg["clean_ratio"], cfg["min_per_rule"], seed=seed)
    out = _write_subset(config, "gecturk_train_12k", subset)
    stats["gecturk_train_12k"] = _describe(subset, cfg["oscar_max_words"])
    full, in_pool, chosen = rule_counts(train), rule_counts(pool), rule_counts(subset)
    stats["rules"] = {
        rule: {"train": full[rule], "pool": in_pool[rule], "subset": chosen[rule]}
        for rule in sorted(full, key=lambda r: int(r.split("_")[1]))
    }
    stats_path = processed / "splits" / "gecturk_sample_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    _summary("gecturk_train_12k", subset, out)
    print(json.dumps({k: v for k, v in stats.items() if k != "rules"}, ensure_ascii=False, indent=2))
    print("kural: train / havuz / alt küme")
    for rule, row in stats["rules"].items():
        print(f"  {rule:8s} {row['train']:6d} {row['pool']:6d} {row['subset']:6d}")

    # Doğrulama alt kümesi: GECTurk dev'in tamamı (20.768 cümle) her adayda çıkarım için çok
    # uzundur. Aynı katmanlı yöntemle sabit bir alt küme seçilir; bütün adaylar aynı cümlelerle ölçülür.
    dev = read_jsonl(processed / "gecturk_dev.jsonl")
    dev_subset = stratified_sample(dev, cfg["gecturk_dev_n"], cfg["clean_ratio"], cfg["gecturk_dev_min_per_rule"], seed=seed)
    name = f"gecturk_dev_{cfg['gecturk_dev_n'] // 1000}k"
    out = _write_subset(config, name, dev_subset)
    _summary(name, dev_subset, out)
    dev_rules = rule_counts(dev_subset)
    print("  kural başına cümle:", ", ".join(f"{rule.split('_')[1]}:{dev_rules[rule]}" for rule in sorted(dev_rules, key=lambda r: int(r.split("_")[1]))))


def leakage(config: dict) -> None:
    """Üretilmiş dosyalarda bütün eğitim-test çiftlerini yeniden tarar; çakışma varsa durur.

    Eleme örnekleme adımında yapılır; bu adım sonucu bağımsız olarak doğrular
    ve sayıları data/processed/splits/leakage_report.json dosyasına yazar.
    """
    processed = Path(config["paths"]["processed"])
    report: dict[str, dict[str, int]] = {}
    for train_name, others in LEAKAGE_PAIRS.items():
        records = read_jsonl(processed / f"{train_name}.jsonl")
        report[train_name] = {}
        for other in others:
            _kept, leaked = split_by_overlap(records, _keys_of(config, other))
            report[train_name][other] = len(leaked)
            print(f"  {train_name:18s} ile {other:16s}: {len(leaked)} çakışma ({len(records)} satır tarandı)")
    path = processed / "splits" / "leakage_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    total = sum(count for row in report.values() for count in row.values())
    if total:
        raise SystemExit(f"Sızıntı taraması: toplam {total} çakışma var. Ayrıntı: {path}")
    print(f"Sızıntı taraması: çakışma yok -> {path}")


def sample(config: dict, refresh: bool = False) -> None:
    """OSCAR'dan doğrulama dilimlerini ve eğitim alt kümesini üretir.

    oscar_dev:        hatalı havuzun ilk 500 satırı (kaynak -> hedef).
    oscar_dev_clean:  hatasız havuzun ilk satırları (hedef -> hedef); organik
                      metinde aşırı düzeltmeyi doğrulama aşamasında ölçmek için.
    oscar_train_12k:  kalan hatalı ve hatasız satırların oscar_clean_ratio
                      oranındaki karışımı (K27). Toplam oscar_n olarak kalır.
    Üç dosya birbirinden ayrıktır; yedek satırlar kullanılmaz.
    """
    cfg = config["sampling"]
    n_train, n_dev, n_dev_clean = cfg["oscar_n"], cfg["oscar_dev_size"], cfg["oscar_dev_clean_size"]
    clean_ratio, max_words, seed = cfg["oscar_clean_ratio"], cfg["oscar_max_words"], config["seed"]
    processed = Path(config["paths"]["processed"])

    error_pool, clean_pool, meta = _load_or_build_oscar_pools(config, refresh)
    if len(error_pool) < n_dev + n_train:
        raise RuntimeError(f"Hatalı havuzda {len(error_pool)} örnek var; {n_dev + n_train} gerekli. oscar_reserve artırılmalı.")

    dev = [oscar_train_record(r["extra"]["row"], r["source"], r["target"], "dev") for r in error_pool[:n_dev]]
    dev_clean = [oscar_clean_record(r["extra"]["row"], r["target"], "dev") for r in clean_pool[:n_dev_clean]]
    train = mix_with_clean(error_pool[n_dev:], clean_pool[n_dev_clean:], n_train, clean_ratio, seed=seed)

    stats = dict(meta)
    stats["clean_ratio"] = clean_ratio
    for name, subset in (("oscar_dev", dev), ("oscar_dev_clean", dev_clean), ("oscar_train_12k", train)):
        out = _write_subset(config, name, subset)
        stats[name] = _describe(subset, max_words)
        _summary(name, subset, out)
    stats_path = processed / "splits" / "oscar_sample_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--step", choices=["download", "convert", "leakage", "sample", "all"], default="all")
    parser.add_argument("--refresh-pools", action="store_true", help="OSCAR havuzlarını diskten okumak yerine veriyi yeniden oku")
    args = parser.parse_args()
    config = load_config(args.config)

    if args.step in NOT_WRITTEN:
        raise NotImplementedError(f"'{args.step}' adımı henüz yazılmadı. Bkz. docs/gorevler/01_veri_hazirligi.md")
    if args.step in ("convert", "all"):
        convert(config)
    if args.step in ("sample", "all"):
        sample_gecturk(config)
        sample(config, refresh=args.refresh_pools)
    if args.step in ("leakage", "all"):
        leakage(config)
    if args.step == "all":
        print("Henüz yazılmamış adımlar atlandı: " + ", ".join(NOT_WRITTEN))


if __name__ == "__main__":
    main()
