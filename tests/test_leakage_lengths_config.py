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
