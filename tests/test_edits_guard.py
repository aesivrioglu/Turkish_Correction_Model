from trgec.edits import token_edit_distance, token_edits
from trgec.guard import apply_guard, change_ratio


def test_token_edits_finds_replacement():
    edits = token_edits("herkez geldi", "herkes geldi")
    assert len(edits) == 1
    assert edits[0].source == ("herkez",) and edits[0].target == ("herkes",)


def test_identical_sentences_have_no_edits():
    assert token_edits("hava güzel", "hava güzel") == []
    assert token_edit_distance("hava güzel", "hava güzel") == 0


def test_split_counts_as_two_token_changes_at_most():
    assert token_edit_distance("bende geldim", "ben de geldim") == 2


def test_guard_keeps_small_changes():
    out, triggered = apply_guard("herkez geldi buraya", "herkes geldi buraya", max_ratio=0.5)
    assert out == "herkes geldi buraya" and not triggered


def test_guard_reverts_rewrites_and_empty_output():
    source = "film çok güzeldi"
    out, triggered = apply_guard(source, "bu yapım gerçekten harika bir eserdi bence", max_ratio=0.5)
    assert out == source and triggered
    assert apply_guard(source, "   ", max_ratio=0.5) == (source, True)


def test_change_ratio_is_zero_for_same_sentence():
    assert change_ratio("a b c", "a b c") == 0.0
