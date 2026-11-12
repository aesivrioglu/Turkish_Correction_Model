import pytest

from trgec.formats import (
    MODE_CORRECT,
    MODE_LABEL,
    OTHER_RULE,
    Edit,
    build_messages,
    build_target,
    parse_output,
)


def test_label_target_round_trip():
    edits = [Edit("ola bilir", "olabilir", "COMP_VERB_ADJ"), Edit("bende", "ben de", "CONJ_DE_SEP")]
    target = build_target("Bu olabilir , ben de gelirim .", edits, MODE_LABEL)
    parsed = parse_output(target)
    assert parsed.well_formed
    assert parsed.corrected == "Bu olabilir , ben de gelirim ."
    assert parsed.edits == edits


def test_clean_sentence_writes_yok():
    target = build_target("Hava güzel .", [], MODE_LABEL)
    assert target.endswith("YOK")
    parsed = parse_output(target)
    assert parsed.edits == [] and parsed.well_formed


def test_correct_only_mode_has_no_separator():
    target = build_target("Hava güzel .", [], MODE_CORRECT)
    assert target == "Hava güzel ."
    assert parse_output(target, expect_edits=False).well_formed
    assert not parse_output(target, expect_edits=True).well_formed


def test_malformed_lines_are_flagged_not_raised():
    parsed = parse_output("Hava güzel .\n###\nbozuk satır\nherkez => herkes")
    assert not parsed.well_formed
    assert parsed.edits == [Edit("herkez", "herkes", OTHER_RULE)]


def test_deletion_edit_keeps_empty_target():
    parsed = parse_output("Geldi .\n###\nde => | CONJ_DE_SEP")
    assert parsed.edits == [Edit("de", "", "CONJ_DE_SEP")]


def test_messages_for_training_and_inference():
    train = build_messages("herkez geldi", MODE_CORRECT, corrected="herkes geldi")
    assert [m["role"] for m in train] == ["user", "assistant"]
    infer = build_messages("herkez geldi", MODE_LABEL)
    assert [m["role"] for m in infer] == ["user"]
    assert "herkez geldi" in infer[0]["content"]


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        build_target("x", [], "bilinmeyen")
