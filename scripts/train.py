"""QLoRA eğitimi. Yerelde CPU üzerinde duman testi, Kaggle'da gerçek eğitim.

Bu betik henüz yazılmadı. Görev tanımı: docs/gorevler/03_pilot_ve_model_secimi.md

Zorunlu bayraklar:
    --config      configs/d1_gecturk.yaml gibi bir yapılandırma
    --smoke-test  8 örnek, 2 adım, 4-bit nicemleme kapalı (yerelde hata yakalamak için)

Hazır yardımcılar: trgec.formats.build_messages, trgec.lengths.suggest_max_len
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke-test", action="store_true")
    parser.parse_args()
    raise NotImplementedError("Bkz. docs/gorevler/03_pilot_ve_model_secimi.md")


if __name__ == "__main__":
    main()
