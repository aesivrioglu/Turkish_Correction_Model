"""ERRANT-TR kurulumunu doğrular: OSCAR GEC testinin resmi M2 dosyasını yeniden üretir.

Altın kaynak (inp.txt) ve altın düzeltmeden (ref.txt) ERRANT-TR ile M2 üretir ve
yayımlanmış M2 dosyasıyla blok blok karşılaştırır. Kurulum doğruysa bütün
bloklar aynı çıkar ve F0.5 = 1,0 olur. Model değerlendirmesi değildir.

Kullanım:
    python scripts/check_errant_tr.py              # 2.408 satır, yaklaşık 14 dakika
    python scripts/check_errant_tr.py --limit 60   # hızlı deneme, yaklaşık 15 saniye
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from trgec import errant_tr
from trgec.config import load_config
from trgec.convert import split_lines
from trgec.metrics import prf

OSCAR_TEST_DIR = "oscar_gec_test"
OSCAR_TEST_M2 = "oscar_gec_m2_W_MAPPED_LABELS.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--limit", type=int, default=None, help="yalnızca ilk N satır")
    args = parser.parse_args()
    config = load_config(args.config)

    raw = Path(config["paths"]["raw"]) / OSCAR_TEST_DIR
    read = lambda name: (raw / name).open(encoding="utf-8", newline="").read()  # noqa: E731
    sources, golds = split_lines(read("inp.txt")), split_lines(read("ref.txt"))
    official = errant_tr.split_blocks(read(OSCAR_TEST_M2))
    limit = args.limit or len(sources)
    sources, golds, official = sources[:limit], golds[:limit], official[:limit]

    errant_dir = config["eval"]["errant_tr_dir"]
    annotator = errant_tr.load_annotator(errant_dir)
    start = time.time()
    blocks = []
    for number, (source, gold) in enumerate(zip(sources, golds), start=1):
        m2 = errant_tr.map_labels(errant_tr.parallel_to_m2(annotator, [source], [gold]))
        blocks.append(errant_tr.split_blocks(m2)[0])
        if number % 300 == 0:
            print(f"  {number} satır, {time.time() - start:.0f} sn", flush=True)

    out = Path(config["paths"]["outputs"]) / "errant_check" / "oscar_ref_regenerated.m2"
    out.parent.mkdir(parents=True, exist_ok=True)
    regenerated = "\n\n".join(blocks) + "\n\n"
    out.write_text(regenerated, encoding="utf-8", newline="\n")

    same = sum(a == b for a, b in zip(blocks, official))
    print(f"satır: {limit} | süre: {time.time() - start:.0f} sn | birebir aynı blok: {same}")
    reference = "\n\n".join(official) + "\n\n"
    for mode in (errant_tr.MODE_SPAN, errant_tr.MODE_SPAN_AND_TYPE):
        tp, fp, fn = errant_tr.total_counts(errant_tr.sentence_counts(regenerated, reference, mode=mode, errant_dir=errant_dir))
        precision, recall, f_score = prf(tp, fp, fn)
        print(f"{mode}: TP {tp} FP {fp} FN {fn} | P {precision:.4f} R {recall:.4f} F0.5 {f_score:.4f}")
    if same != limit:
        raise SystemExit(f"UYARI: {limit - same} blok resmi dosyadan farklı. Çıktı: {out}")
    print("Kurulum doğrulandı.")


if __name__ == "__main__":
    main()
