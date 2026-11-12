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
