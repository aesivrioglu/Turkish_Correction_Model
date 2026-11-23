import json

import pytest

from trgec import convert
from trgec.convert import (
    convert_oscar_test,
    convert_tweets,
    m2_edit_types,
    make_record,
    read_jsonl,
    split_lines,
    write_jsonl,
)

# Sahte veri: gerçek dosyaların biçimini taklit eder, içerik uydurmadır.
M2 = (
    "S Bende geldim .\n"
    "A 0 1|||R:SPELL|||Ben de|||REQUIRED|||-NONE-|||0\n"
    "A 2 2|||M:PUNC|||,|||REQUIRED|||-NONE-|||0\n"
    "\n"
    "S Hata yok .\n"
    "A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0\n"
)
TWEET_HEADER = "\ufeffRAW DATA\tPREPROCESSED DATA\tCORRECTED DATA\t\tANNOTATED DATA\r\n"


def tweet_file(*rows):
    return TWEET_HEADER + "".join("\t".join(row) + "\r\n" for row in rows)


def test_make_record_schema_and_id():
    record = make_record(7, "oscar", "test", "kaynak", "hedef")
    assert record == {
        "id": "oscar-test-000007",
        "source": "kaynak",
        "target": "hedef",
        "rules": [],
        "origin": "oscar",
        "split": "test",
        "extra": {},
    }


def test_split_lines_only_breaks_on_newline():
    # U+2028 (satır ayırıcı) cümlenin içinde kalmalı; str.splitlines() burada bölerdi.
    text = "bir\u2028iki\r\nüç\n"
    assert split_lines(text) == ["bir\u2028iki", "üç"]
    assert len(text.splitlines()) == 3


def test_m2_edit_types_skips_noop():
    assert m2_edit_types(M2) == [["R:SPELL", "M:PUNC"], []]


def test_m2_edit_types_rejects_broken_block():
    with pytest.raises(ValueError):
        m2_edit_types("A 0 1|||R:SPELL|||x|||REQUIRED|||-NONE-|||0\n")


def test_oscar_keeps_natural_text_untouched():
    inp = "Bende geldim.Sonra  gittim\nHata yok.\n"
    ref = "Ben de geldim. Sonra gittim.\nHata yok.\n"
    records = convert_oscar_test(inp, ref, M2)
    assert [r["id"] for r in records] == ["oscar-test-000001", "oscar-test-000002"]
    # Doğal metin: noktalama bitişik, çift boşluk yerinde, tokenizasyon yok (K24).
    assert records[0]["source"] == "Bende geldim.Sonra  gittim"
    assert records[0]["target"] == "Ben de geldim. Sonra gittim."
    assert records[0]["rules"] == []
    assert records[0]["extra"] == {"m2_types": ["R:SPELL", "M:PUNC"]}
    assert records[1]["extra"] == {"m2_types": []}
    assert all(r["origin"] == "oscar" and r["split"] == "test" for r in records)


def test_oscar_without_m2_has_empty_extra():
    assert convert_oscar_test("a\n", "b\n")[0]["extra"] == {}


def test_oscar_rejects_misaligned_files():
    with pytest.raises(ValueError):
        convert_oscar_test("bir\niki\n", "bir\n")
    with pytest.raises(ValueError):
        convert_oscar_test("bir\n", "bir\n", M2)


def test_tweets_bar_becomes_space_and_labels_are_kept():
    text = tweet_file(
        ("Tabiki geldim :)", "tabiki geldim", "tabii|ki geldim", "OOV-ill_formed-sep IV"),
        ("bir çok kişi", "bir|çok kişi ", "birçok  kişi", "OOV-ill_formed-joint IV"),
    )
    first, second = convert_tweets(text)
    assert first["id"] == "tweets-test-000001"
    assert first["source"] == "tabiki geldim"
    assert first["target"] == "tabii ki geldim"
    assert first["rules"] == []
    assert first["extra"] == {
        "raw": "Tabiki geldim :)",
        "labels": ["OOV-ill_formed-sep", "IV"],
        "source_aligned": "tabiki geldim",
        "target_aligned": "tabii|ki geldim",
    }
    # '|' boşluğa döner; dosyadan gelen sondaki boşluk ve çift boşluk temizlenir.
    assert second["source"] == "bir çok kişi"
    assert second["target"] == "birçok kişi"
    assert second["extra"]["source_aligned"] == "bir|çok kişi "


def test_tweets_keep_combining_dot_and_case():
    # Veri setindeki str.lower() izi (i + U+0307) ölçütün kaynağıdır, değiştirilmez.
    text = tweet_file(("Yedi̇m", "yedi\u0307m", "Yedim", "OOV-ill_formed-accent"))
    record = convert_tweets(text)[0]
    assert record["source"] == "yedi\u0307m"
    assert record["target"] == "Yedim"


def test_tweets_reject_bad_rows():
    with pytest.raises(ValueError):
        convert_tweets("sütun başlığı yok\n")
    with pytest.raises(ValueError):
        convert_tweets(tweet_file(("ham", "iki kelime", "iki kelime")))
    with pytest.raises(ValueError):
        convert_tweets(tweet_file(("ham", "iki kelime", "iki kelime", "IV")))


def test_jsonl_roundtrip_keeps_turkish_letters(tmp_path):
    records = [make_record(1, "tweets", "test", "çğıöşü İ", "ÇĞIÖŞÜ i", extra={"labels": ["IV", "IV"]})]
    path = tmp_path / "alt" / "ornek.jsonl"
    assert write_jsonl(records, path) == 1
    assert "çğıöşü İ" in path.read_text(encoding="utf-8")  # \u kaçışı yok, dosya okunabilir
    assert read_jsonl(path) == records
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["id"] == "tweets-test-000001"


def test_oscar_row_problem_reasons():
    from trgec.convert import oscar_row_problem

    assert oscar_row_problem("bir iki üç", "Bir iki üç.", max_words=3) is None
    assert oscar_row_problem("bir iki üç dört", "Bir iki üç dört.", max_words=3) == "too_long"
    assert oscar_row_problem("  ", "x", max_words=3) == "empty"
    assert oscar_row_problem(None, "x", max_words=3) == "empty"
    assert oscar_row_problem(float("nan"), "x", max_words=3) == "empty"
    assert oscar_row_problem("bir\niki", "bir iki", max_words=3) == "multiline"
    # Baş ve sondaki satır sonu sorun değil; silinir.
    assert oscar_row_problem("bir iki\n", "Bir iki.", max_words=3) is None


def test_oscar_train_record_keeps_row_number_and_inner_text():
    from trgec.convert import oscar_train_record

    record = oscar_train_record(1234567, "  geldim,gittim  ", "Geldim, gittim. ", "train")
    assert record["id"] == "oscar-train-1234567"
    assert record["source"] == "geldim,gittim"
    assert record["target"] == "Geldim, gittim."
    assert record["rules"] == [] and record["origin"] == "oscar" and record["split"] == "train"
    assert record["extra"] == {"row": 1234567}
    assert oscar_train_record(42, "a", "b", "dev")["id"] == "oscar-dev-000042"


def test_oscar_clean_record_uses_correction_on_both_sides():
    from trgec.convert import oscar_clean_record

    record = oscar_clean_record(77, " Geldim, gittim. ", "train")
    assert record["id"] == "oscar-train-000077-c"
    assert record["source"] == record["target"] == "Geldim, gittim."
    assert record["extra"] == {"row": 77, "clean_from_target": True}


# --- GECTurk (M2) ---

GECTURK_M2 = (
    "S Dün saha da maç vardı.\n"
    "A 1 3|||rule_6|||sahada|||REQUIRED|||-NONE-|||0\n"
    "\n"
    "S Hava çok güzel.\n"
    "\n"
    "S herkez geldi ama o gelmedi de bende gittim.\n"
    "A 0 1|||rule_23|||Herkes|||REQUIRED|||-NONE-|||0\n"
    "A 6 7|||rule_1|||ben de|||REQUIRED|||-NONE-|||0\n"
    "A 0 1|||rule_23|||Herkes|||REQUIRED|||-NONE-|||0\n"
)


def test_parse_m2_reads_blocks_and_edits():
    blocks = convert.parse_m2(GECTURK_M2)
    assert [source for source, _ in blocks] == ["Dün saha da maç vardı.", "Hava çok güzel.", "herkez geldi ama o gelmedi de bende gittim."]
    assert blocks[0][1] == [{"start": 1, "end": 3, "rule": "rule_6", "correction": "sahada"}]
    assert blocks[1][1] == []
    assert convert.parse_m2("") == []
    # noop satırı düzenleme sayılmaz.
    assert convert.parse_m2("S Tamam.\nA -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0\n")[0][1] == []
    with pytest.raises(ValueError):
        convert.parse_m2("Bu bir M2 bloğu değil\n")
    with pytest.raises(ValueError):
        convert.parse_m2("S Cümle.\nA bozuk satır\n")


def test_apply_m2_edits_handles_length_changes():
    edits = [
        {"start": 1, "end": 3, "correction": "sahada"},  # iki kelime -> bir kelime
        {"start": 4, "end": 5, "correction": "var dı"},  # bir kelime -> iki kelime
    ]
    assert convert.apply_m2_edits("Dün saha da maç vardı.", edits) == "Dün sahada maç var dı"
    assert convert.apply_m2_edits("Hava çok güzel.", []) == "Hava çok güzel."
    with pytest.raises(ValueError):  # cümlenin dışına taşıyor
        convert.apply_m2_edits("iki kelime", [{"start": 1, "end": 5, "correction": "x"}])
    with pytest.raises(ValueError):  # örtüşüyor
        convert.apply_m2_edits("a b c", [{"start": 0, "end": 2, "correction": "x"}, {"start": 1, "end": 3, "correction": "y"}])


def test_resolve_overlapping_edits_keeps_widest_then_latest():
    nested = [
        {"start": 0, "end": 1, "rule": "rule_23", "correction": "Halbu"},
        {"start": 0, "end": 2, "rule": "rule_8", "correction": "Halbuki"},
        {"start": 4, "end": 5, "rule": "rule_1", "correction": "ben de"},
    ]
    assert [e["correction"] for e in convert.resolve_overlapping_edits(nested)] == ["Halbuki", "ben de"]
    two_steps = [
        {"start": 3, "end": 4, "rule": "rule_2", "correction": "izlenilmesede"},
        {"start": 3, "end": 4, "rule": "rule_1", "correction": "izlenilmese de"},
    ]
    assert [e["correction"] for e in convert.resolve_overlapping_edits(two_steps)] == ["izlenilmese de"]


def test_convert_gecturk_with_target_file():
    targets = "Dün sahada maç vardı.\nHava çok güzel.\nHerkes geldi ama o gelmedi de ben de gittim.\n"
    records = convert.convert_gecturk(GECTURK_M2, "train", targets)
    assert [r["id"] for r in records] == ["gecturk-train-000001", "gecturk-train-000002", "gecturk-train-000003"]
    assert records[0]["target"] == "Dün sahada maç vardı."
    assert records[0]["rules"] == ["rule_6"]
    assert records[0]["extra"]["edits"] == [{"start": 1, "end": 3, "source": "saha da", "target": "sahada", "rule": "rule_6"}]
    assert records[1]["rules"] == [] and records[1]["source"] == records[1]["target"]
    # Kurallar ilk geçiş sırasıyla ve yinelemesiz; düzenlemelerin hepsi extra içinde.
    assert records[2]["rules"] == ["rule_23", "rule_1"]
    assert len(records[2]["extra"]["edits"]) == 3
    with pytest.raises(ValueError):
        convert.convert_gecturk(GECTURK_M2, "train", "tek satır\n")


def test_convert_gecturk_without_target_applies_edits():
    records = convert.convert_gecturk(GECTURK_M2, "test")
    assert records[0]["target"] == "Dün sahada maç vardı."
    assert "overlap_resolved" not in records[0]["extra"]
    # Üçüncü cümlede aynı düzenleme iki kez yazılmış: biri uygulanır, cümle işaretlenir.
    assert records[2]["target"] == "Herkes geldi ama o gelmedi de ben de gittim."
    assert records[2]["extra"]["overlap_resolved"] is True
