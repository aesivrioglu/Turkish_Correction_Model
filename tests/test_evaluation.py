import pytest

from trgec.evaluation import (
    align_predictions,
    capitalize_first,
    clean_sentence_stats,
    strip_punctuation,
    summarize,
    tweet_protocol,
)


def test_capitalize_first_modes():
    assert capitalize_first("iyi akşamlar Adana", "turkish") == "İyi akşamlar Adana"
    assert capitalize_first("iyi akşamlar Adana", "ascii") == "Iyi akşamlar Adana"
    assert capitalize_first("iyi akşamlar Adana", "python_capitalize") == "Iyi akşamlar adana"
    assert capitalize_first("ısrar etme", "turkish") == "Israr etme"
    assert capitalize_first("iyi", "none") == "iyi"
    # Harf olmayan ilk karakter ve boş metin değişmez.
    assert capitalize_first("#gündem bugün", "turkish") == "#gündem bugün"
    assert capitalize_first("3 kişi geldi", "turkish") == "3 kişi geldi"
    assert capitalize_first("", "turkish") == ""
    with pytest.raises(ValueError):
        capitalize_first("a", "bilinmeyen")


def test_strip_punctuation_keeps_listed_characters():
    text = "Adana'da hava güzel, değil mi? #gündem :)"
    assert strip_punctuation(text, keep="'#") == "Adana'da hava güzel değil mi #gündem"
    assert strip_punctuation(text, keep="") == "Adana da hava güzel değil mi gündem"
    # Bitişik noktalama kelimeleri birleştirmez.
    assert strip_punctuation("geldim,gittim.") == "geldim gittim"
    # Simgeler (emoji, $) noktalama sayılmaz.
    assert strip_punctuation("#Turki$hTweets 🙂", keep="#") == "#Turki$hTweets 🙂"


def test_tweet_protocol_transforms_gold_and_hypothesis_separately():
    golds, hyps = tweet_protocol(["iyi akşamlar"], ["İyi akşamlar!"])
    assert golds == ["İyi akşamlar"]
    assert hyps == ["İyi akşamlar"]


def test_align_predictions_orders_and_validates():
    records = [{"id": "a"}, {"id": "b"}]
    predictions = [{"id": "b", "hypothesis": "2"}, {"id": "a", "hypothesis": "1"}]
    assert [p["hypothesis"] for p in align_predictions(records, predictions)] == ["1", "2"]
    with pytest.raises(ValueError):
        align_predictions(records, predictions[:1])
    with pytest.raises(ValueError):
        align_predictions(records, predictions + [{"id": "c", "hypothesis": "3"}])
    with pytest.raises(ValueError):
        align_predictions(records, predictions + [{"id": "a", "hypothesis": "x"}])


def test_summarize_totals_and_interval():
    summary = summarize([(2, 0, 0), (0, 1, 1), (1, 0, 1)], n_boot=200)
    assert (summary["tp"], summary["fp"], summary["fn"]) == (3, 1, 2)
    assert summary["precision"] == 0.75 and summary["recall"] == 0.6
    assert summary["f_beta_low"] <= summary["f_beta"] <= summary["f_beta_high"]


def test_clean_sentence_stats():
    sources = ["Hata yok.", "Bende geldim.", "Bu da doğru."]
    golds = ["Hata yok.", "Ben de geldim.", "Bu da doğru."]
    hyps = ["Hata yok.", "Ben de geldim.", "Bu de doğru."]
    assert clean_sentence_stats(sources, golds, hyps) == {"clean_sentences": 2, "overcorrection_rate": 0.5}
    assert clean_sentence_stats(["a"], ["b"], ["b"]) == {"clean_sentences": 0, "overcorrection_rate": None}
