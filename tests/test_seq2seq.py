import pytest

from trgec.seq2seq import clean_output, generation_summary, hit_length_limit, length_sorted_batches, output_budget


def test_length_sorted_batches_covers_every_index_once():
    lengths = [5, 40, 12, 40, 7, 3, 19]
    batches = length_sorted_batches(lengths, batch_size=3)
    assert sorted(index for batch in batches for index in batch) == list(range(len(lengths)))
    assert [len(batch) for batch in batches] == [3, 3, 1]


def test_length_sorted_batches_longest_first_and_stable():
    lengths = [5, 40, 12, 40, 7]
    # En uzunlar başta; eşit uzunlukta (1 ve 3) özgün sıra korunur.
    assert length_sorted_batches(lengths, batch_size=2) == [[1, 3], [2, 4], [0]]
    assert length_sorted_batches([], batch_size=4) == []
    with pytest.raises(ValueError):
        length_sorted_batches(lengths, batch_size=0)


def test_clean_output_makes_single_line():
    assert clean_output("  Bugün hava\nçok  güzel.\r\n") == "Bugün hava çok güzel."
    assert clean_output("") == ""


def test_hit_length_limit_looks_for_eos():
    # mT5'te 0 dolgu ve başlangıç, 1 bitiş (EOS) tokenıdır.
    assert hit_length_limit([0, 250, 318, 1, 0, 0], eos_token_id=1) is False
    assert hit_length_limit([0, 250, 318, 97], eos_token_id=1) is True


def test_generation_summary_counts():
    rows = [
        {"source": "a b", "hypothesis": "a b", "input_tokens": 4, "output_tokens": 5, "truncated_input": False, "hit_length_limit": False},
        {"source": "c d", "hypothesis": "C d.", "input_tokens": 10, "output_tokens": 7, "truncated_input": True, "hit_length_limit": True},
        {"source": "e", "hypothesis": "", "input_tokens": 1, "output_tokens": 2, "truncated_input": False, "hit_length_limit": False},
    ]
    summary = generation_summary(rows)
    assert summary["n"] == 3
    assert summary["truncated_inputs"] == 1
    assert summary["hit_length_limit"] == 1
    assert summary["empty_outputs"] == 1
    assert summary["unchanged"] == 1
    assert summary["input_tokens_max"] == 10 and summary["output_tokens_max"] == 7
    assert summary["input_tokens_mean"] == 5.0
    assert generation_summary([]) == {"n": 0}


def test_output_budget_scales_with_sentence_and_respects_hard_limit():
    assert output_budget(10, hard_limit=256, per_source_token=2.0, margin=64) == 84
    assert output_budget(40, hard_limit=256, per_source_token=2.0, margin=64) == 144
    assert output_budget(200, hard_limit=256, per_source_token=2.0, margin=64) == 256  # üst sınır
    assert output_budget(0, hard_limit=256, per_source_token=2.0, margin=0) == 1
