"""Elle kontrol: bir setten rastgele örnekleri (kaynak, hedef, kurallar) ekrana yazdırır.

Görev tanımı: docs/gorevler/01_veri_hazirligi.md, adım 7.

Biçim hatası burada yakalanmazsa sonraki bütün skorlar bozulur; bu yüzden her
set gözle kontrol edilir. Kaynak ile hedef arasında değişen kelimeler ayrıca
listelenir ki uzun cümlelerde fark aranmasın.

Örnek:
    python scripts/show_examples.py --set gecturk_train_12k
    python scripts/show_examples.py --set oscar_train_12k --n 20 --only clean
"""

from __future__ import annotations

import argparse
import difflib
import random
import sys
from pathlib import Path

from trgec.config import load_config
from trgec.convert import read_jsonl


def changed_words(source: str, target: str) -> list[str]:
    """Kaynak ve hedef arasında değişen kelime öbeklerini 'yanlış => doğru' olarak döndürür."""
    source_words, target_words = source.split(), target.split()
    matcher = difflib.SequenceMatcher(a=source_words, b=target_words, autojunk=False)
    changes = []
    for tag, a_start, a_end, b_start, b_end in matcher.get_opcodes():
        if tag != "equal":
            before = " ".join(source_words[a_start:a_end]) or "∅"
            after = " ".join(target_words[b_start:b_end]) or "∅"
            changes.append(f"{before} => {after}")
    return changes


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows konsolunda Türkçe harfler bozulmasın
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", required=True, dest="name", help="data/processed altındaki setin adı (ör. gecturk_train_12k)")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", choices=["all", "clean", "errors"], default="all", help="yalnızca hatasız ya da yalnızca hatalı örnekler")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    path = Path(config["paths"]["processed"]) / f"{args.name}.jsonl"
    records = read_jsonl(path)
    total = len(records)
    if args.only == "clean":
        records = [r for r in records if r["source"] == r["target"]]
    elif args.only == "errors":
        records = [r for r in records if r["source"] != r["target"]]
    chosen = random.Random(args.seed).sample(records, min(args.n, len(records)))

    print(f"{args.name}: {total} örnek; gösterilen {len(chosen)} (süzgeç: {args.only}, tohum {args.seed})\n")
    for number, record in enumerate(chosen, start=1):
        print(f"[{number}] {record['id']}")
        print(f"  kaynak : {record['source']}")
        if record["source"] == record["target"]:
            print("  hedef  : (aynı, hatasız örnek)")
        else:
            print(f"  hedef  : {record['target']}")
            print(f"  değişen: {' ; '.join(changed_words(record['source'], record['target']))}")
        if record["rules"]:
            print(f"  kurallar: {', '.join(record['rules'])}")
        print()


if __name__ == "__main__":
    main()
