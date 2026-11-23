import pytest

from trgec.config import load_config
from trgec.leakage import drop_overlap, find_overlap
from trgec.lengths import suggest_max_len, truncation_rate


def test_overlap_ignores_case_and_punctuation():
    train = ["Film çok güzeldi!", "Başka bir cümle."]
    test = ["film çok güzeldi", "Hiç görülmemiş cümle"]
    assert find_overlap(train, test) == ["film çok güzeldi"]
    assert drop_overlap(train, test) == ["Başka bir cümle."]


def test_suggest_max_len_rounds_up_to_step():
    lengths = list(range(1, 201))
    assert suggest_max_len(lengths, quantile=0.95, step=32) == 192
    assert truncation_rate(lengths, 192) == pytest.approx(8 / 200)


def test_suggest_max_len_rejects_empty():
    with pytest.raises(ValueError):
        suggest_max_len([])


def test_config_inheritance(tmp_path):
    (tmp_path / "base.yaml").write_text("seed: 42\ntraining:\n  lr: 0.0002\n  epochs: 1\n", encoding="utf-8")
    (tmp_path / "child.yaml").write_text(
        "inherits: base.yaml\nrun_name: deneme\ntraining:\n  epochs: 2\n", encoding="utf-8"
    )
    config = load_config(tmp_path / "child.yaml")
    assert config == {"seed": 42, "run_name": "deneme", "training": {"lr": 0.0002, "epochs": 2}}


def test_split_by_overlap_checks_source_and_target():
    from trgec.leakage import match_keys, split_by_overlap

    keys = match_keys(["Film çok güzeldi!", "İyi akşamlar", "  "])
    assert keys == {"film çok güzeldi", "iyi akşamlar"}
    records = [
        {"id": "a", "source": "film çok güzeldi", "target": "Film çok güzeldi."},
        {"id": "b", "source": "iyi aksamlar", "target": "İyi akşamlar!"},  # yalnızca hedef eşleşiyor
        {"id": "c", "source": "başka bir cümle", "target": "Başka bir cümle."},
    ]
    kept, leaked = split_by_overlap(records, keys)
    assert [r["id"] for r in kept] == ["c"]
    assert [r["id"] for r in leaked] == ["a", "b"]


def test_dedupe_records_keeps_first_and_counts():
    from trgec.leakage import dedupe_records

    records = [
        {"id": 1, "source": "Merhaba dünya"},
        {"id": 2, "source": "merhaba, dünya!"},
        {"id": 3, "source": "IŞIK"},
        {"id": 4, "source": "ışık"},
        {"id": 5, "source": "isik"},
    ]
    unique, dropped = dedupe_records(records)
    assert [r["id"] for r in unique] == [1, 3, 5]
    assert dropped == 2
