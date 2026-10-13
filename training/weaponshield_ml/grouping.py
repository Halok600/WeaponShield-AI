"""Perceptual-hash de-duplication and grouping, so near-identical frames never straddle train and test."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from .sample import Sample


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def phash_int(path: Path) -> int:
    import imagehash
    from PIL import Image

    with Image.open(path) as im:
        return int(str(imagehash.phash(im.convert("RGB"))), 16)


def compute_hashes(samples: Sequence[Sample]) -> list[int]:
    return [phash_int(s.image_path) for s in samples]


def near_duplicate_pairs(hashes: Sequence[int], max_distance: int = 6) -> list[tuple[int, int]]:
    """All index pairs whose 64-bit hashes differ in at most `max_distance` bits.

    Multi-index hashing: split each hash into 8 bytes. Two hashes within distance < 8 must agree on at least
    one byte, so only hashes sharing a (position, byte) bucket are compared.
    """
    if not 0 <= max_distance < 8:
        raise ValueError("max_distance must be in [0, 8)")
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, h in enumerate(hashes):
        for k in range(8):
            buckets[(k, (h >> (8 * k)) & 0xFF)].append(i)
    pairs: set[tuple[int, int]] = set()
    for members in buckets.values():
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                a, b = members[x], members[y]
                if (hashes[a] ^ hashes[b]).bit_count() <= max_distance:
                    pairs.add((a, b) if a < b else (b, a))
    return sorted(pairs)


def dedupe_and_group(samples: list[Sample], hashes: list[int], max_distance: int = 6) -> tuple[list[Sample], int]:
    """Drop exact pHash duplicates (first wins), then merge near-duplicates and shared group hints into groups."""
    if len(samples) != len(hashes):
        raise ValueError("samples and hashes must have the same length")
    seen: set[int] = set()
    kept: list[Sample] = []
    kept_hashes: list[int] = []
    for s, h in zip(samples, hashes, strict=True):
        if h in seen:
            continue
        seen.add(h)
        kept.append(s)
        kept_hashes.append(h)

    uf = UnionFind(len(kept))
    first_with_hint: dict[str, int] = {}
    for i, s in enumerate(kept):
        uf.union(i, first_with_hint.setdefault(s.group, i))
    for a, b in near_duplicate_pairs(kept_hashes, max_distance):
        uf.union(a, b)
    for i, s in enumerate(kept):
        s.group = f"g{uf.find(i)}"
    return kept, len(samples) - len(kept)
