"""Değerlendirme: resmi skorlayıcıyı çalıştırır, cümle bazlı sayıları ve güven aralıklarını üretir.

Bu betik henüz yazılmadı. Görev tanımı: docs/gorevler/02_degerlendirme_hatti.md

Girdi:  tahmin dosyası (JSONL: {"id", "source", "hypothesis"}) ve test setinin adı
Çıktı:  outputs/eval/<calisma>/<set>.json  (P, R, F0.5, güven aralığı, aşırı düzeltme oranı)

Kural (K15): Her test setinde, o setin yayımlanmış skorunda kullanılan araç kullanılır.
Araçlar configs/base.yaml içinde eval.scorers altında: film yorumlarında M2 scorer ve
ERRANT-TR, OSCAR test ve tweetlerde ERRANT-TR.
Hazır yardımcılar: trgec.bootstrap, trgec.metrics, trgec.guard
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--testset", required=True, choices=["movie_reviews", "oscar_test", "tweets", "gecturk_dev", "oscar_dev"])
    parser.add_argument("--config", default="configs/base.yaml")
    parser.parse_args()
    raise NotImplementedError("Bkz. docs/gorevler/02_degerlendirme_hatti.md")


if __name__ == "__main__":
    main()
