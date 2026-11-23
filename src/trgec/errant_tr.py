"""ERRANT-TR sarmalayıcısı: paralel cümleler -> M2 -> cümle başına (tp, fp, fn).

ERRANT-TR (Uz ve Eryiğit 2023, github.com/harunuz/erranttr, MIT) paketlenmemiştir
ve process.py sabit bir dosya adı okur. Bu modül aynı adımları dosya adı
dayatmadan çalıştırır. Depo external/erranttr altına, PINNED_COMMIT sürümüyle
klonlanır (kurulum: docs/KURULUM.md). Depoya girmez.

Ersoy ve Yıldız (2024) OSCAR testi ve tweet skorlarını bu araçla üretti (K15).
Onların yolu üç adımdır; hepsi burada karşılığıyla durur:

    process.py           -> parallel_to_m2
    map_labels.py        -> map_labels
    commands/compare_m2  -> sentence_counts

Yalnızca sentence_counts ve map_labels Zemberek olmadan çalışır; M2 üretmek
(parallel_to_m2) için zemberek-python ve rapidfuzz kurulu olmalıdır.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
import typing
from pathlib import Path
from typing import Any, Iterable, Sequence

DEFAULT_DIR = Path("external/erranttr")
PINNED_COMMIT = "56cf0a3d8b1e98c9a3500ac972cbcf20dc1fd4db"
RESOURCE_FILES = ("tr_TR.aff", "TS_Corpus_Turkish_Word_List.txt")
MODE_SPAN = "cs"  # yalnızca konum ve düzeltme eşleşmeli (ERRANT'ın varsayılanı)
MODE_SPAN_AND_TYPE = "cse"  # hata türü de eşleşmeli
NOOP_EDIT = "A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0"

# map_labels.py içindeki tabloyla birebir aynı (sıra dahil).
MAPPED_LABELS = {
    "DA": "SPELL",
    "ACCENT": "SPELL",
    "NOUN:CASE": "NOUN:INFL",
    "NOUN:NUM:SURF": "NOUN:INFL",
    "ADJ:CASE": "ADJ",
    "VERB:INFL:TENSE": "VERB:TENSE",
    "NOUN:PHRASE": "OTHER",
    "NOUN:STC:NEG": "VERB",
    "NOUN-VERB:INFL:CASE": "NOUN:INFL",
    "PRON:CASE": "NOUN:INFL",
    "PRON:INFL": "NOUN:INFL",
    "PRON:POSS": "NOUN:POSS",
}


def _patch_typing_io() -> None:
    """Python 3.13 uyumluluğu: antlr4-python3-runtime 4.8 'typing.io' modülünü ister.

    Bu modül Python 3.13'te kaldırıldı. zemberek-python 4.8'e sabitlendiği için
    sürüm yükseltilemez; aynı adları taşıyan boş bir modül tanıtmak yeterlidir.
    """
    if "typing.io" in sys.modules:
        return
    shim = types.ModuleType("typing.io")
    shim.TextIO, shim.BinaryIO, shim.IO = typing.TextIO, typing.BinaryIO, typing.IO
    sys.modules["typing.io"] = shim


def check_install(errant_dir: str | Path = DEFAULT_DIR) -> Path:
    """Deponun ve iki kaynak dosyanın yerinde olduğunu doğrular; depo yolunu döndürür."""
    root = Path(errant_dir)
    if not (root / "erranttr" / "process.py").exists():
        raise FileNotFoundError(f"ERRANT-TR bulunamadı: {root}. Kurulum: docs/KURULUM.md")
    for name in RESOURCE_FILES:
        if not (root / "erranttr" / "tr" / "resources" / name).exists():
            raise FileNotFoundError(f"ERRANT-TR kaynak dosyası eksik: {name}. Kurulum: docs/KURULUM.md")
    return root


def load_annotator(errant_dir: str | Path = DEFAULT_DIR) -> Any:
    """ERRANT-TR açıklayıcısını yükler (Zemberek sözlüğü yüklendiği için birkaç saniye sürer)."""
    root = check_install(errant_dir)
    _patch_typing_io()
    if str(root.resolve()) not in sys.path:
        sys.path.insert(0, str(root.resolve()))
    import erranttr  # noqa: PLC0415  (yol yukarıda eklendi)

    return erranttr.load("tr")


def _surface(tokens: Iterable[Any]) -> str:
    return " ".join(token.word_analysis.inp for token in tokens)


def parallel_to_m2(annotator: Any, sources: Sequence[str], corrections: Sequence[str]) -> str:
    """Kaynak ve düzeltilmiş cümle çiftlerinden M2 metni üretir (process.py ile aynı mantık).

    Her çift için bir blok yazılır: Zemberek'in tokenize ettiği kaynak ('S ...'),
    ardından her düzenleme için bir 'A ...' satırı. Kaynak ve düzeltme aynıysa
    tek bir 'noop' satırı yazılır. process.py boş kaynak satırını atlar; burada
    hata verilir, çünkü atlanan satır referans M2 ile hizayı kaydırır.
    """
    if len(sources) != len(corrections):
        raise ValueError(f"{len(sources)} kaynak, {len(corrections)} düzeltme var")
    blocks = []
    for number, (source, correction) in enumerate(zip(sources, corrections), start=1):
        source, correction = source.strip(), correction.strip()
        if not source:
            raise ValueError(f"{number}. kaynak cümle boş")
        orig = annotator.parse(source)
        lines = ["S " + _surface(orig)]
        if source == correction:
            lines.append(NOOP_EDIT)
        else:
            cor = annotator.parse(correction)
            lines.extend(edit.to_m2(0) for edit in annotator.annotate(orig, cor))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n\n"


def map_labels(m2_text: str) -> str:
    """ERRANT-TR'ye özgü hata türlerini ERRANT karşılıklarına çevirir (map_labels.py ile aynı).

    Özgün betik türleri alan alan değil, bütün dosyada düz metin değiştirmeyle
    eşler. Bu yüzden cümlenin ve düzeltmenin içindeki büyük harfli 'DA' gibi
    parçalar da değişir ('DAMACANA' -> 'SPELLMACANA'). Yayımlanmış OSCAR GEC
    referans dosyası bu betikten geçmiştir (13 'S' satırında iz var). Düzeltme
    metinleri referansla eşleşsin diye hipotez M2'si de aynı işlemden geçmelidir;
    davranış bilerek aynen korunur.
    """
    for old, new in MAPPED_LABELS.items():
        m2_text = m2_text.replace(old, new)
    return m2_text


def split_blocks(m2_text: str) -> list[str]:
    """M2 metnini bloklara böler (compare_m2.py ile aynı kural)."""
    return m2_text.replace("\r\n", "\n").strip().split("\n\n")


def _load_compare_module(errant_dir: str | Path) -> Any:
    """compare_m2.py dosyasını paket olarak değil dosya olarak yükler.

    'import erranttr' Zemberek'i de yükler; skorlama için buna gerek yoktur.
    """
    path = Path(errant_dir) / "erranttr" / "commands" / "compare_m2.py"
    if not path.exists():
        raise FileNotFoundError(f"ERRANT-TR bulunamadı: {path}. Kurulum: docs/KURULUM.md")
    spec = importlib.util.spec_from_file_location("erranttr_compare_m2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sentence_counts(
    hyp_m2: str,
    ref_m2: str,
    mode: str = MODE_SPAN,
    beta: float = 0.5,
    errant_dir: str | Path = DEFAULT_DIR,
) -> list[tuple[int, int, int]]:
    """Hipotez ve referans M2 metinlerinden cümle başına (tp, fp, fn) üretir.

    Sayım ERRANT-TR'nin compare_m2.py işlevleriyle yapılır; bu işlev yalnızca
    onun ana döngüsünü tekrarlar ve toplamı basmak yerine cümle bazlı sayıları
    döndürür (bootstrap için gerekli, K18). mode: 'cs' ya da 'cse'.
    """
    if mode not in (MODE_SPAN, MODE_SPAN_AND_TYPE):
        raise ValueError(f"Bilinmeyen mod: {mode}")
    compare = _load_compare_module(errant_dir)
    args = argparse.Namespace(
        beta=beta, verbose=False, dt=False, ds=False,
        cs=mode == MODE_SPAN, cse=mode == MODE_SPAN_AND_TYPE,
        single=False, multi=False, filt=[], cat=None,
    )
    hyp_blocks, ref_blocks = split_blocks(hyp_m2), split_blocks(ref_m2)
    if len(hyp_blocks) != len(ref_blocks):
        raise ValueError(f"Hipotezde {len(hyp_blocks)}, referansta {len(ref_blocks)} blok var")
    best = {"tp": 0, "fp": 0, "fn": 0}
    counts = []
    for sent_id, (hyp, ref) in enumerate(zip(hyp_blocks, ref_blocks)):
        hyp_dict = compare.process_edits(compare.simplify_edits(hyp), args)
        ref_dict = compare.process_edits(compare.simplify_edits(ref), args)
        count, _categories = compare.evaluate_edits(hyp_dict, ref_dict, best, sent_id, "", args)
        for key in best:
            best[key] += count[key]
        counts.append((count["tp"], count["fp"], count["fn"]))
    return counts


def total_counts(counts: Iterable[tuple[int, int, int]]) -> tuple[int, int, int]:
    """Cümle bazlı sayıları toplar."""
    tp = fp = fn = 0
    for a, b, c in counts:
        tp, fp, fn = tp + a, fp + b, fn + c
    return tp, fp, fn
