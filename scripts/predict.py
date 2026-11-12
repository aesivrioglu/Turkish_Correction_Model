"""Çıkarım: bir modelin test ya da doğrulama setindeki tahminlerini JSONL olarak yazar.

Bu betik henüz yazılmadı. Görev tanımı: docs/gorevler/03_pilot_ve_model_secimi.md

Çıktı: outputs/predictions/<calisma>/<set>.jsonl
    {"id", "source", "raw_output", "hypothesis", "edits", "well_formed", "guard_triggered"}

Hazır yardımcılar: trgec.formats.parse_output, trgec.guard.apply_guard
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--testset", required=True)
    parser.add_argument("--adapter", help="LoRA adaptörünün yolu ya da Hub kimliği")
    parser.parse_args()
    raise NotImplementedError("Bkz. docs/gorevler/03_pilot_ve_model_secimi.md")


if __name__ == "__main__":
    main()
