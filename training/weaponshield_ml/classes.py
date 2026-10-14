"""Class schema shared by every data source, the trainer, the web app and the edge tool."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Literal

CLASS_NAMES: Final = ("person", "pistol", "rifle", "knife")
PERSON, PISTOL, RIFLE, KNIFE = 0, 1, 2, 3
WEAPON_IDS: Final = frozenset({PISTOL, RIFLE, KNIFE})

DROP_IMAGE: Final = "drop-image"
Resolved = int | None | Literal["drop-image"]
"""int: keep the box as that class; None: discard the box; DROP_IMAGE: discard the whole image."""

# Roboflow Universe 'weapon-detection' by yolov7test (CC BY 4.0), raw class names.
ROBOFLOW_LABELS: Final[dict[str, Resolved]] = {
    "Pistol": PISTOL,
    "pistol": PISTOL,
    "pistols": PISTOL,
    "handgun": PISTOL,
    "gun": PISTOL,  # v16 export: its 'gun' boxes are handguns (long guns are 'heavy-weapon')
    "Rifle": RIFLE,
    "rifle": RIFLE,
    "Shotgun": RIFLE,
    "shotgun": RIFLE,
    "Long guns": RIFLE,
    "larga": RIFLE,
    "Heavy Gun": RIFLE,
    "heavyweapon": RIFLE,
    "heavy-weapon": RIFLE,
    "Knife": KNIFE,
    "Knife_Deploy": KNIFE,
    "Knife_Weapon": KNIFE,
    "knife": KNIFE,
    # Generic labels that do not say which weapon it is: the image cannot be labelled correctly.
    "weapons": DROP_IMAGE,
    "weapon": DROP_IMAGE,
    "Guns": DROP_IMAGE,
    "guns": DROP_IMAGE,
    "Guns perspective": DROP_IMAGE,
    "al": DROP_IMAGE,
    # People and scene context: boxes are discarded (people are re-labelled uniformly by autolabel).
    "Person": None,
    "person": None,
    "Aggressor": None,
    "Victim": None,
    "Blood": None,
    "Hand": None,
    "Stabbing": None,
    "violence": None,
}

OI_WEAPON_CLASSES: Final = ("Handgun", "Rifle", "Shotgun", "Knife", "Kitchen knife", "Dagger")

OI_LABELS: Final[dict[str, Resolved]] = {
    "Handgun": PISTOL,
    "Rifle": RIFLE,
    "Shotgun": RIFLE,
    "Knife": KNIFE,
    "Kitchen knife": KNIFE,
    "Dagger": KNIFE,
    "Weapon": DROP_IMAGE,
    "Sword": DROP_IMAGE,
}

# Look-alike objects people hold: phones, tablets, tools. Used for training negatives and the hard-negative set.
OI_NEGATIVE_CLASSES: Final = (
    "Mobile phone",
    "Tablet computer",
    "Remote control",
    "Drill (Tool)",
    "Hair dryer",
    "Umbrella",
    "Camera",
    "Flashlight",
    "Hammer",
    "Screwdriver",
    "Wrench",
    "Scissors",
    "Microphone",
    "Baseball bat",
)


def resolve(name: str, table: Mapping[str, Resolved], *, strict: bool) -> Resolved:
    if name in table:
        return table[name]
    if strict:
        raise KeyError(f"Unmapped source label {name!r}")
    return None
