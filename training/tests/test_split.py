from weaponshield_ml.sample import Sample
from weaponshield_ml.split import SPLITS, split_groups


def make(n_singletons, big_group=0):
    out = [Sample(key=f"s{i}", source="t", image_path=None, width=1, height=1) for i in range(n_singletons)]  # type: ignore[arg-type]
    out += [Sample(key=f"b{i}", source="t", image_path=None, width=1, height=1, group="big") for i in range(big_group)]  # type: ignore[arg-type]
    return out


def test_ratios_are_close_to_target():
    splits = split_groups(make(1000))
    sizes = {k: len(v) for k, v in splits.items()}
    assert sum(sizes.values()) == 1000
    assert abs(sizes["train"] - 800) <= 20
    assert abs(sizes["val"] - 100) <= 20
    assert abs(sizes["test"] - 100) <= 20


def test_no_group_is_split_across_splits():
    splits = split_groups(make(500, big_group=50))
    owner = {}
    for name in SPLITS:
        for s in splits[name]:
            assert owner.setdefault(s.group, name) == name


def test_split_is_deterministic():
    a = split_groups(make(300), seed=7)
    b = split_groups(make(300), seed=7)
    assert {k: [s.key for s in v] for k, v in a.items()} == {k: [s.key for s in v] for k, v in b.items()}
