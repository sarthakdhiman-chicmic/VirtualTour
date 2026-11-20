"""Automatic room classification (SOW IN-04: "Autopilot step 1 classifies rooms automatically"), from
what OneFormer sees in the photo (tour_segment's scene summary: each class's share of the photo
and how many of it, by the common names of vision/segment.COMMON). Used only where the upload
carries no room type (IN-03: the user's choice always wins); every room it names is flagged
room_auto:<photo>:<type>, so it can be reviewed. Every type it names is one of the backend's
TourRoomType values. Class names are matched trimmed and lower-case.

COCO has no stove, hood, bathtub, shower, desk, stool, armchair, wardrobe or pool class: kitchens
are told by an oven / microwave / toaster or a fridge at a counter, bathrooms by a toilet or a
towel by a sink, offices by a computer, keyboard or mouse; a pool is water; COCO's "fence" takes in
railings.

Rules, first match wins:
  outdoors       no ceiling and sky or a building in view: water -> pool_area, steps up to the
                 house with little sky -> entry, a railing -> balcony, grass or trees filling the
                 view -> backyard, a building filling the view -> front_exterior, else exterior
  covered        a railing with the outdoors (sky, trees, water) beyond, under a roof: a sofa ->
                 outdoor_patio, else balcony
  pool_area      water under a ceiling (the enum has no indoor pool)
  bedroom        a bed taking BED_MIN of the photo
  living_room    sofas taking LOUNGE_MIN (before the bathroom: a stray fixture in a lounge)
  bathroom       a toilet; or a towel by a sink, mirror or counter with nothing to cook on
  kitchen        two kinds of kitchen fitting, one of them for cooking; or a fridge by a sink or counter
  garage         a car under a ceiling
  library        shelves taking LIBRARY_MIN of the photo, with books
  living_room    a sofa taking SOFA_MIN, a sofa with a table or rug, or a TV with seats
  dining_room    a table with 3+ chairs
  home_office    a computer, a keyboard or a mouse
  staircase      stairs taking STAIRS_MIN of the photo
  hallway        a railing indoors taking RAIL_INDOOR_MIN (a landing)
  none           nothing a viewer would name: the photo stays unknown (never defaulted)
"""
from __future__ import annotations

BED_MIN, SOFA_MIN, LOUNGE_MIN, LIBRARY_MIN, STAIRS_MIN = 0.06, 0.03, 0.08, 0.15, 0.03
RAIL_INDOOR_MIN, VIEW_MIN = 0.05, 0.12
SKY_MIN, BUILDING_MIN, CEILING_MAX = 0.06, 0.12, 0.02
FRONT_MIN, GREEN_MIN = 0.25, 0.25     # house share: the front of the home; grass + trees: a yard
POOL_MIN, RAILING_MIN = 0.03, 0.03
COOKING = ("oven", "microwave", "toaster")
KITCHEN = COOKING + ("sink", "refrigerator", "counter")
OUTSIDE = ("sky", "tree", "grass", "sea", "water", "river", "mountain")
BUILT = ("house", "building", "roof")


def classify(scene: dict | None) -> tuple[str | None, str]:
    """(room type or None, why) from a scene summary {"classes": {label: {area, count, ...}}}."""
    if not scene:
        return None, "no_scene"
    c = {str(k).strip().lower(): v for k, v in (scene.get("classes") or {}).items()}
    area = lambda *ks: sum(float((c.get(k) or {}).get("area", 0.0)) for k in ks)   # noqa: E731
    count = lambda k: int((c.get(k) or {}).get("count", 0))                         # noqa: E731
    has = lambda *ks: any(k in c for k in ks)                                        # noqa: E731
    indoors = area("ceiling") > CEILING_MAX
    if not indoors and (area("sky") >= SKY_MIN or area(*BUILT) >= BUILDING_MIN):
        if area("water") > POOL_MIN and area("water") > area("sea", "river"):
            return "pool_area", "water, outdoors"
        if area("stairs") >= 0.05 and area("sky") < SKY_MIN:
            return "entry", "steps up to the house"
        if area("fence") >= RAILING_MIN and area("grass") < 0.05:
            return "balcony", "railing, outdoors"
        if area("grass", "tree") >= GREEN_MIN and area(*BUILT) < FRONT_MIN:
            return "backyard", "grass / trees"
        if area(*BUILT) >= FRONT_MIN:
            return "front_exterior", "the house fills the view"
        return "exterior", "sky/building, no ceiling"
    if area("fence") >= RAILING_MIN and area(*OUTSIDE) >= VIEW_MIN:
        return ("outdoor_patio", "covered, with sofas") if has("sofa") else ("balcony", "covered, railing")
    if area("water") > POOL_MIN:
        return "pool_area", "water indoors"
    if area("bed") >= BED_MIN:
        return "bedroom", "bed"
    if area("sofa") >= LOUNGE_MIN:
        return "living_room", "sofas"
    kinds = [k for k in KITCHEN if has(k)]
    cooking = any(k in COOKING for k in kinds)
    if has("toilet") or (has("towel") and has("sink", "mirror", "counter") and not cooking):
        return "bathroom", "toilet / towel"
    if len(kinds) >= 2 and cooking:
        return "kitchen", "+".join(kinds[:3])
    if has("refrigerator") and has("sink", "counter"):
        return "kitchen", "fridge at a counter"
    if has("car") and indoors:
        return "garage", "car indoors"
    if area("shelf") >= LIBRARY_MIN and has("book"):
        return "library", "shelves of books"
    if (area("sofa") >= SOFA_MIN or (has("sofa") and has("table", "rug"))
            or (has("television receiver") and has("sofa", "chair"))):
        return "living_room", "sofa / tv"
    if has("table") and count("chair") >= 3:
        return "dining_room", "table and chairs"
    if has("computer", "keyboard", "mouse"):
        return "home_office", "computer"
    if area("stairs") >= STAIRS_MIN:
        return "staircase", "stairs"
    if area("fence") >= RAIL_INDOOR_MIN:
        return "hallway", "a landing's rail"
    return None, "nothing named"


def fill_room_types(photos: list[tuple], scenes: dict) -> tuple[list[tuple], list[str]]:
    """photos: [(photo_id, image, room_type)]; the ones with no room type get classify()'s, where
    it names one. Returns (photos, warnings room_auto:<photo>:<type>)."""
    out, warnings = [], []
    for pid, img, rtype in photos:
        if not rtype:
            guess, _ = classify((scenes or {}).get(pid) or (scenes or {}).get(str(pid)))
            if guess:
                rtype = guess
                warnings.append(f"room_auto:{pid}:{guess}")
        out.append((pid, img, rtype))
    return out, warnings
