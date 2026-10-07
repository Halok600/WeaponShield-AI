import pytest

from weaponshield_ml.classes import (
    CLASS_NAMES,
    DROP_IMAGE,
    KNIFE,
    OI_LABELS,
    OI_NEGATIVE_CLASSES,
    OI_WEAPON_CLASSES,
    PERSON,
    PISTOL,
    RIFLE,
    ROBOFLOW_LABELS,
    WEAPON_IDS,
    resolve,
)

# Exact raw class names of the Roboflow Universe 'weapon-detection' project (yolov7test), checked 2026-10-07.
RF_NAMES = [
    "weapons",
    "Aggressor",
    "Blood",
    "Guns",
    "Guns perspective",
    "Hand",
    "Heavy Gun",
    "Knife",
    "Knife_Deploy",
    "Knife_Weapon",
    "Long guns",
    "Person",
    "Pistol",
    "Rifle",
    "Shotgun",
    "Stabbing",
    "Victim",
    "al",
    "guns",
    "handgun",
    "heavyweapon",
    "larga",
    "person",
    "pistol",
    "pistols",
    "rifle",
    "shotgun",
    "violence",
    "weapon",
]


def test_schema_matches_web_app():
    assert CLASS_NAMES == ("person", "pistol", "rifle", "knife")
    assert (PERSON, PISTOL, RIFLE, KNIFE) == (0, 1, 2, 3)
    assert WEAPON_IDS == {PISTOL, RIFLE, KNIFE}


def test_every_roboflow_label_has_an_explicit_rule():
    assert sorted(ROBOFLOW_LABELS) == sorted(RF_NAMES)


@pytest.mark.parametrize("name", ["Pistol", "pistol", "pistols", "handgun"])
def test_handguns_are_pistols(name):
    assert resolve(name, ROBOFLOW_LABELS, strict=True) == PISTOL


@pytest.mark.parametrize(
    "name", ["Rifle", "rifle", "Shotgun", "shotgun", "Long guns", "larga", "Heavy Gun", "heavyweapon"]
)
def test_long_guns_are_rifles(name):
    assert resolve(name, ROBOFLOW_LABELS, strict=True) == RIFLE


@pytest.mark.parametrize("name", ["Knife", "Knife_Deploy", "Knife_Weapon"])
def test_knives(name):
    assert resolve(name, ROBOFLOW_LABELS, strict=True) == KNIFE


@pytest.mark.parametrize("name", ["weapons", "weapon", "Guns", "guns", "Guns perspective", "al"])
def test_generic_weapon_labels_drop_the_image(name):
    assert resolve(name, ROBOFLOW_LABELS, strict=True) == DROP_IMAGE


@pytest.mark.parametrize("name", ["Person", "person", "Aggressor", "Victim", "Blood", "Hand", "Stabbing", "violence"])
def test_people_and_context_boxes_are_discarded(name):
    assert resolve(name, ROBOFLOW_LABELS, strict=True) is None


def test_unknown_labels():
    with pytest.raises(KeyError, match="Revolver"):
        resolve("Revolver", ROBOFLOW_LABELS, strict=True)
    assert resolve("Mobile phone", OI_LABELS, strict=False) is None


def test_open_images_tables():
    assert {resolve(c, OI_LABELS, strict=False) for c in OI_WEAPON_CLASSES} == {PISTOL, RIFLE, KNIFE}
    assert resolve("Kitchen knife", OI_LABELS, strict=False) == KNIFE
    assert resolve("Sword", OI_LABELS, strict=False) == DROP_IMAGE
    assert resolve("Weapon", OI_LABELS, strict=False) == DROP_IMAGE
    assert not set(OI_NEGATIVE_CLASSES) & set(OI_LABELS)
