"""ERRANT-TR sarmalayıcısının testleri.

Skorlama testleri external/erranttr deposunu, M2 üretme testleri ayrıca
zemberek-python'ı ister. Yoksa testler atlanır (kurulum: docs/KURULUM.md).
"""

import pytest

from trgec import errant_tr
from trgec.metrics import prf

HAS_REPO = (errant_tr.DEFAULT_DIR / "erranttr" / "commands" / "compare_m2.py").exists()
needs_repo = pytest.mark.skipif(not HAS_REPO, reason="external/erranttr yok")

REF = (
    "S Bende geldim ama o gelmedi\n"
    "A 0 1|||R:ORTH|||Ben de|||REQUIRED|||-NONE-|||0\n"
    "A 5 5|||M:PUNC|||.|||REQUIRED|||-NONE-|||0\n"
    "\n"
    "S Hata yok .\n"
    "A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0\n"
    "\n"
)
SOURCE_ONLY = (
    "S Bende geldim ama o gelmedi\n"
    "A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0\n"
    "\n"
    "S Hata yok .\n"
    "A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0\n"
    "\n"
)
OTHER_TYPE = REF.replace("R:ORTH", "R:SPELL")
OVERCORRECTED = SOURCE_ONLY.replace(
    "S Hata yok .\nA -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0",
    "S Hata yok .\nA 0 1|||R:SPELL|||Hatta|||REQUIRED|||-NONE-|||0",
)


def test_map_labels_matches_original_script_including_its_side_effect():
    text = "S DAMACANA aldım\nA 0 1|||R:DA|||Damacana|||REQUIRED|||-NONE-|||0\nA 1 2|||R:NOUN:CASE|||x|||REQUIRED|||-NONE-|||0\n"
    mapped = errant_tr.map_labels(text)
    assert "|||R:SPELL|||" in mapped and "|||R:NOUN:INFL|||" in mapped
    # Özgün betik cümle metnini de değiştirir; referans dosyası böyle üretildiği için aynen korunur.
    assert mapped.startswith("S SPELLMACANA aldım")


def test_split_blocks_handles_crlf_and_trailing_blank_lines():
    assert len(errant_tr.split_blocks(REF)) == 2
    assert len(errant_tr.split_blocks(REF.replace("\n", "\r\n"))) == 2


@needs_repo
def test_gold_as_hypothesis_scores_one():
    counts = errant_tr.sentence_counts(REF, REF)
    assert counts == [(2, 0, 0), (0, 0, 0)]
    assert prf(*errant_tr.total_counts(counts)) == (1.0, 1.0, 1.0)


@needs_repo
def test_source_as_hypothesis_has_zero_recall():
    counts = errant_tr.sentence_counts(SOURCE_ONLY, REF)
    assert counts == [(0, 0, 2), (0, 0, 0)]
    precision, recall, f_score = prf(*errant_tr.total_counts(counts))
    assert recall == 0.0 and f_score == 0.0


@needs_repo
def test_overcorrection_counts_as_false_positive():
    assert errant_tr.sentence_counts(OVERCORRECTED, REF) == [(0, 0, 2), (0, 1, 0)]


@needs_repo
def test_type_only_matters_in_cse_mode():
    assert errant_tr.total_counts(errant_tr.sentence_counts(OTHER_TYPE, REF, mode="cs")) == (2, 0, 0)
    assert errant_tr.total_counts(errant_tr.sentence_counts(OTHER_TYPE, REF, mode="cse")) == (1, 1, 1)


@needs_repo
def test_sentence_counts_rejects_bad_input():
    with pytest.raises(ValueError):
        errant_tr.sentence_counts(REF, REF + REF)
    with pytest.raises(ValueError):
        errant_tr.sentence_counts(REF, REF, mode="xx")


@pytest.fixture(scope="module")
def annotator():
    errant_tr._patch_typing_io()  # zemberek'in içe aktarılabilmesi için (Python 3.13)
    pytest.importorskip("zemberek")
    pytest.importorskip("rapidfuzz")
    try:
        return errant_tr.load_annotator()
    except FileNotFoundError as error:
        pytest.skip(str(error))


def test_parallel_to_m2_end_to_end(annotator):
    sources = ["Bende dün akşam sinemaya gittim.", "Hata yok."]
    gold = ["Ben de dün akşam sinemaya gittim.", "Hata yok."]
    ref = errant_tr.map_labels(errant_tr.parallel_to_m2(annotator, sources, gold))
    blocks = errant_tr.split_blocks(ref)
    assert blocks[0].split("\n")[0] == "S Bende dün akşam sinemaya gittim ."
    assert "|||Ben de|||" in blocks[0]
    assert blocks[1].split("\n")[1] == errant_tr.NOOP_EDIT

    # Akıl sağlığı: altın düzeltme hipotez olunca F0.5 = 1; kaynak cümle hipotez olunca recall = 0.
    assert prf(*errant_tr.total_counts(errant_tr.sentence_counts(ref, ref)))[2] == 1.0
    unchanged = errant_tr.map_labels(errant_tr.parallel_to_m2(annotator, sources, sources))
    tp, fp, fn = errant_tr.total_counts(errant_tr.sentence_counts(unchanged, ref))
    assert (tp, fp) == (0, 0) and fn >= 1


def test_parallel_to_m2_rejects_misaligned_or_empty(annotator):
    with pytest.raises(ValueError):
        errant_tr.parallel_to_m2(annotator, ["a"], ["a", "b"])
    with pytest.raises(ValueError):
        errant_tr.parallel_to_m2(annotator, ["  "], ["a"])
