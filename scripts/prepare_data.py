"""Veri hazırlığı: indirme, ortak biçime çevirme, sızıntı taraması, örnekleme.

Bu betik henüz yazılmadı. Görev tanımı: docs/gorevler/01_veri_hazirligi.md

Çıktı sözleşmesi (data/processed/<ad>.jsonl, satır başına bir örnek):
    {"id": str, "source": str, "target": str, "rules": [str, ...],
     "origin": "gecturk" | "oscar" | "tweets", "split": str}

Hazır yardımcılar: trgec.sampling.stratified_sample, trgec.leakage.find_overlap,
trgec.text.normalize_for_match
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--step", choices=["download", "convert", "leakage", "sample", "all"], default="all")
    parser.parse_args()
    raise NotImplementedError("Bkz. docs/gorevler/01_veri_hazirligi.md")


if __name__ == "__main__":
    main()
