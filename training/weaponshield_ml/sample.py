from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .classes import WEAPON_IDS
from .labels import Box


@dataclass
class Sample:
    """One image on its way into the dataset. `group` ties together images that must share a split."""

    key: str
    source: str
    image_path: Path
    width: int
    height: int
    boxes: list[Box] = field(default_factory=list)
    group: str = ""

    def __post_init__(self) -> None:
        if not self.group:
            self.group = self.key

    @property
    def weapon_classes(self) -> set[int]:
        return {b.cls for b in self.boxes if b.cls in WEAPON_IDS}
