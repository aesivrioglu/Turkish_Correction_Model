import pytest

from trgec.bootstrap import bootstrap_ci, paired_bootstrap
from trgec.metrics import overcorrection_rate, prf


def test_prf_weights_precision():
    p, r, f = prf(tp=8, fp=2, fn=8)
    assert p == pytest.approx(0.8) and r == pytest.approx(0.5)
    assert f == pytest.approx(1.25 * 0.8 * 0.5 / (0.25 * 0.8 + 0.5))
    assert f > 2 * p * r / (p + r)  # F0.5, F1'den büyük: precision daha yüksek


def test_prf_handles_zero_counts():
    assert prf(0, 0, 0) == (1.0, 1.0, 1.0)
    assert prf(0, 5, 5)[2] == 0.0


def test_overcorrection_rate():
    sources = ["hava güzel", "film iyiydi", "geldim", "gittim"]
    hyps = ["hava güzel", "film iyi idi", "geldim ", "gittim"]
    assert overcorrection_rate(sources, hyps) == 0.25


def test_bootstrap_ci_contains_score():
    counts = [(1, 0, 0)] * 60 + [(0, 1, 1)] * 40
    result = bootstrap_ci(counts, n_boot=300, seed=1)
    assert result["low"] <= result["score"] <= result["high"]
    assert result["high"] - result["low"] > 0


def test_paired_bootstrap_detects_clear_winner():
    good = [(1, 0, 0)] * 90 + [(0, 1, 1)] * 10
    bad = [(1, 0, 0)] * 40 + [(0, 1, 1)] * 60
    result = paired_bootstrap(good, bad, n_boot=300, seed=1)
    assert result["diff"] > 0 and result["low"] > 0
    assert result["p_a_not_better"] == 0.0


def test_paired_bootstrap_identical_systems_have_zero_diff():
    counts = [(1, 0, 0)] * 50 + [(0, 1, 1)] * 50
    result = paired_bootstrap(counts, counts, n_boot=100, seed=1)
    assert result["diff"] == 0.0 and result["low"] == 0.0 and result["high"] == 0.0
