import numpy as np
from PIL import Image

from weaponshield_ml.grouping import dedupe_and_group, near_duplicate_pairs, phash_int
from weaponshield_ml.sample import Sample


def sample(key, group=""):
    return Sample(key=key, source="t", image_path=None, width=1, height=1, group=group)  # type: ignore[arg-type]


def test_near_duplicate_pairs_respects_distance():
    h0, h1, h2 = 0, 0b111111, 0b1111111  # d(h0,h1)=6, d(h1,h2)=1, d(h0,h2)=7
    assert near_duplicate_pairs([h0, h1, h2], max_distance=6) == [(0, 1), (1, 2)]


def test_near_duplicate_pairs_far_apart():
    assert near_duplicate_pairs([0, (1 << 64) - 1], max_distance=6) == []


def test_dedupe_drops_exact_copies_first_wins():
    kept, dropped = dedupe_and_group([sample("a"), sample("b"), sample("c")], [5, 5, 1 << 40])
    assert [s.key for s in kept] == ["a", "c"]
    assert dropped == 1


def test_groups_merge_near_duplicates_and_shared_hints():
    s = [sample("a"), sample("b"), sample("c", group="vid1"), sample("d", group="vid1"), sample("e")]
    # a/b are 2 bits apart; c, d and e are >= 32 bits from everything else
    hashes = [0, 0b11, 0xFFFF_FFFF_0000_0000, 0x0F0F_0F0F_0F0F_0F0F, 0xFFFF_FFFF_FFFF_FFFF]
    kept, _ = dedupe_and_group(s, hashes)
    g = {x.key: x.group for x in kept}
    assert g["a"] == g["b"]
    assert g["c"] == g["d"]
    assert len({g["a"], g["c"], g["e"]}) == 3


def test_phash_is_stable_and_distinguishes_images(tmp_path):
    rng = np.random.default_rng(0)
    a = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    b = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    a.save(tmp_path / "a.png")
    a.save(tmp_path / "a2.png")
    b.save(tmp_path / "b.png")
    assert phash_int(tmp_path / "a.png") == phash_int(tmp_path / "a2.png")
    assert phash_int(tmp_path / "a.png") != phash_int(tmp_path / "b.png")
