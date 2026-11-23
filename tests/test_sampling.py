from trgec.sampling import rule_counts, stratified_sample


def _examples():
    data = [{"id": f"c{i}", "rules": []} for i in range(1000)]
    data += [{"id": f"f{i}", "rules": ["SIK"]} for i in range(900)]
    data += [{"id": f"r{i}", "rules": ["NADIR"]} for i in range(30)]
    data += [{"id": f"m{i}", "rules": ["SIK", "ORTA"]} for i in range(70)]
    return data


def test_rare_rule_is_fully_kept_when_below_minimum():
    sample = stratified_sample(_examples(), n_total=400, min_per_rule=50, seed=1)
    counts = rule_counts(sample)
    assert counts["NADIR"] == 30
    assert counts["ORTA"] >= 50


def test_size_and_clean_ratio():
    sample = stratified_sample(_examples(), n_total=400, clean_ratio=0.5, seed=1)
    clean = sum(1 for ex in sample if not ex["rules"])
    assert len(sample) == 400 and clean == 200


def test_same_seed_gives_same_sample():
    a = stratified_sample(_examples(), n_total=300, seed=7)
    b = stratified_sample(_examples(), n_total=300, seed=7)
    assert [ex["id"] for ex in a] == [ex["id"] for ex in b]


def test_no_duplicates():
    sample = stratified_sample(_examples(), n_total=500, seed=3)
    ids = [ex["id"] for ex in sample]
    assert len(ids) == len(set(ids))


def test_reservoir_sample_is_deterministic_and_sized():
    from trgec.sampling import reservoir_sample

    first = reservoir_sample(iter(range(1000)), 50, seed=7)
    assert first == reservoir_sample(iter(range(1000)), 50, seed=7)
    assert first != reservoir_sample(iter(range(1000)), 50, seed=8)
    assert len(first) == 50 and len(set(first)) == 50
    assert set(first) <= set(range(1000))


def test_reservoir_sample_returns_everything_when_stream_is_short():
    from trgec.sampling import reservoir_sample

    assert sorted(reservoir_sample(iter("abc"), 10)) == ["a", "b", "c"]
    assert reservoir_sample(iter("abc"), 0) == []


def test_reservoir_sample_gives_early_and_late_items_equal_chance():
    from trgec.sampling import reservoir_sample

    # 1000 öğeden 100 seçilirse her öğenin şansı %10. 400 tohumda beklenen 40.
    runs = [set(reservoir_sample(iter(range(1000)), 100, seed=s)) for s in range(400)]
    for item in (0, 500, 999):
        hits = sum(item in run for run in runs)
        assert 20 <= hits <= 60, (item, hits)


def test_reservoir_class_matches_function_and_counts_stream():
    from trgec.sampling import Reservoir, reservoir_sample

    reservoir = Reservoir(20, seed=3)
    for item in range(500):
        reservoir.offer(item)
    assert reservoir.seen == 500
    assert reservoir.result() == reservoir_sample(iter(range(500)), 20, seed=3)


def test_two_reservoirs_over_one_stream_are_independent():
    from trgec.sampling import Reservoir

    first, second = Reservoir(30, seed=1), Reservoir(30, seed=2)
    for item in range(2000):
        first.offer(item)
        second.offer(item)
    assert set(first.result()) != set(second.result())


def test_mix_with_clean_ratio_and_nested_errors():
    from trgec.sampling import mix_with_clean

    errors = [f"e{i}" for i in range(100)]
    clean = [f"c{i}" for i in range(100)]
    half = mix_with_clean(errors, clean, n_total=40, clean_ratio=0.5, seed=1)
    quarter = mix_with_clean(errors, clean, n_total=40, clean_ratio=0.25, seed=1)
    assert sorted(x for x in half if x[0] == "c") == sorted(f"c{i}" for i in range(20))
    assert sum(x[0] == "c" for x in quarter) == 10 and sum(x[0] == "e" for x in quarter) == 30
    # Oran değişince hatalı örnekler iç içe kalır.
    assert {x for x in half if x[0] == "e"} <= {x for x in quarter if x[0] == "e"}
    assert mix_with_clean(errors, clean, 40, 0.0) != errors[:40]  # karıştırılmış
    assert sorted(mix_with_clean(errors, clean, 40, 0.0)) == sorted(errors[:40])


def test_mix_with_clean_rejects_bad_input():
    import pytest

    from trgec.sampling import mix_with_clean

    with pytest.raises(ValueError):
        mix_with_clean(["e"], ["c"], n_total=4, clean_ratio=0.5)
    with pytest.raises(ValueError):
        mix_with_clean(["e"], ["c"], n_total=2, clean_ratio=1.5)
