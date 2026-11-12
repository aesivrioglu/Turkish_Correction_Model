from trgec.text import normalize_for_match, tr_lower, tr_upper


def test_tr_lower_handles_dotted_and_dotless_i():
    assert tr_lower("IŞIK") == "ışık"
    assert tr_lower("İSTANBUL") == "istanbul"


def test_tr_upper_handles_dotted_and_dotless_i():
    assert tr_upper("ışık") == "IŞIK"
    assert tr_upper("istanbul") == "İSTANBUL"


def test_normalize_ignores_case_punctuation_and_spacing():
    assert normalize_for_match("  Film, ÇOK  güzeldi! ") == normalize_for_match("film çok güzeldi")


def test_normalize_drops_combining_dot_from_python_lower():
    # "İ".lower() Python'da "i" + U+0307 verir; #Turki$hTweets kaynak sütununda 47 kez geçiyor.
    # Birleşen nokta kelimeyi bölmemeli, yoksa sızıntı taraması eşleşmeyi kaçırır.
    assert normalize_for_match("yedi̇m") == "yedim"
    assert normalize_for_match("İstanbul".lower()) == normalize_for_match("İstanbul")
