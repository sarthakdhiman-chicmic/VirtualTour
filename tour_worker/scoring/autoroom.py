"""Automatic room classification (SOW IN-04: "Autopilot step 1 classifies rooms automatically"), from
what Mask2Former sees in the photo (tour_segment's scene summary: each class's share of the photo
and how many of it). Used only where the upload carries no room type (IN-03: the user's choice
always wins); every room it names is flagged room_auto:<photo>:<type>, so it can be reviewed.
Every type it names is one of the backend's TourRoomType values.

Class names are matched trimmed and lower-case: Mask2Former's ADE20K id2label spells some with a
trailing space ("bed "), which hid every bed.

Rules, first match wins:
  outdoors       no ceiling and sky or a building in view: a pool -> pool_area, steps up to the
                 house with little sky -> entry, a railing -> balcony, a building filling the view
                 -> front_exterior, else exterior
  covered        a railing with the outdoors (sky, trees, water) beyond, under a roof: a sofa ->
                 outdoor_patio, else balcony
  pool_area      a swimming pool (indoors too: the enum has no indoor pool)
  nursery        a cradle
  bedroom        a bed taking BED_MIN of the photo
  living_room    sofas taking LOUNGE_MIN (before the bathroom: a stray fixture in a lounge)
  bathroom       a toilet, bath or shower; or a towel by a sink, mirror or counter with nothing to cook on
  kitchen        two kinds of kitchen fitting, one of them for cooking; or stools at a counter,
                 island, cabinet or table
  laundry_room   a washer
  garage         a car under a ceiling
  game_room      a pool table or an arcade machine
  library        bookcases taking LIBRARY_MIN of the photo
  living_room    a sofa taking SOFA_MIN, a sofa with a coffee table or rug, armchairs taking
                 LOUNGE_MIN, or a TV / fireplace with an armchair
  dining_room    a table with 3+ chairs
  home_office    a desk, a computer, a monitor or a swivel chair
  walk_in_closet wardrobes or clothes taking CLOSET_MIN of the photo
  staircase      stairs, a stairway or steps taking STAIRS_MIN of the photo
  hallway        a stair rail or railing indoors taking RAIL_INDOOR_MIN (a landing)
  none           nothing a viewer would name: the photo stays unknown (never defaulted)
"""
from __future__ import annotations

BED_MIN, SOFA_MIN, LOUNGE_MIN, LIBRARY_MIN, CLOSET_MIN, STAIRS_MIN = 0.06, 0.03, 0.08, 0.15, 0.10, 0.03
RAIL_INDOOR_MIN, VIEW_MIN, STOOL_MIN = 0.05, 0.12, 0.02
SKY_MIN, BUILDING_MIN, CEILING_MAX = 0.06, 0.12, 0.02
FRONT_MIN = 0.25                       # house / building share: the front of the home
POOL_MIN, RAILING_MIN = 0.03, 0.03
COOKING = ("stove", "oven", "hood", "dishwasher", "microwave", "kitchen island")
KITCHEN = COOKING + ("sink", "refrigerator", "countertop", "counter")
BATH = ("toilet", "bathtub", "shower")
OUTSIDE = ("sky", "tree", "grass", "palm", "sea", "water", "lake", "river", "mountain", "hill")


def classify(scene: dict | None) -> tuple[str | None, str]:
    """(room type or None, why) from a scene summary {"classes": {label: {area, count, ...}}}."""
    if not scene:
        return None, "no_scene"
    c = {str(k).strip().lower(): v for k, v in (scene.get("classes") or {}).items()}
    area = lambda *ks: sum(float((c.get(k) or {}).get("area", 0.0)) for k in ks)   # noqa: E731
    count = lambda k: int((c.get(k) or {}).get("count", 0))                         # noqa: E731
    has = lambda *ks: any(k in c for k in ks)                                        # noqa: E731
    indoors = area("ceiling") > CEILING_MAX
    if not indoors and (area("sky") >= SKY_MIN or area("house", "building") >= BUILDING_MIN):
        if area("swimming pool") > POOL_MIN:
            return "pool_area", "pool, outdoors"
        if area("stairs", "step") >= 0.05 and area("sky") < SKY_MIN:
            return "entry", "steps up to the house"
        if area("railing", "bannister") >= RAILING_MIN:
            return "balcony", "railing, outdoors"
        if area("house", "building") >= FRONT_MIN:
            return "front_exterior", "the house fills the view"
        return "exterior", "sky/building, no ceiling"
    if area("railing") >= RAILING_MIN and area(*OUTSIDE) >= VIEW_MIN:
        return ("outdoor_patio", "covered, with sofas") if has("sofa") else ("balcony", "covered, railing")
    if area("swimming pool") > POOL_MIN:
        return "pool_area", "pool"
    if has("cradle"):
        return "nursery", "cradle"
    if area("bed") >= BED_MIN:
        return "bedroom", "bed"
    if area("sofa") >= LOUNGE_MIN:
        return "living_room", "sofas"
    kinds = [k for k in KITCHEN if has(k)]
    cooking = any(k in COOKING for k in kinds)
    if has(*BATH) or (has("towel") and has("sink", "mirror", "countertop", "counter") and not cooking):
        return "bathroom", "bath fixture / towel"
    if len(kinds) >= 2 and cooking:
        return "kitchen", "+".join(kinds[:3])
    if count("stool") >= 2 and area("stool") >= STOOL_MIN and has("countertop", "counter", "kitchen island", "cabinet", "table"):
        return "kitchen", "stools at a counter"
    if has("washer"):
        return "laundry_room", "washer"
    if has("car") and indoors:
        return "garage", "car indoors"
    if has("pool table", "arcade machine"):
        return "game_room", "pool table / arcade"
    if area("bookcase") >= LIBRARY_MIN:
        return "library", "bookcases"
    if (area("sofa") >= SOFA_MIN or (has("sofa") and has("coffee table", "rug")) or area("armchair") >= LOUNGE_MIN
            or (has("television receiver", "fireplace") and has("armchair"))):
        return "living_room", "sofa / tv"
    if has("table") and count("chair") >= 3:
        return "dining_room", "table and chairs"
    if has("desk", "computer", "monitor", "swivel chair"):
        return "home_office", "desk"
    if area("wardrobe", "apparel") >= CLOSET_MIN:
        return "walk_in_closet", "wardrobes / clothes"
    if area("stairs", "stairway", "step") >= STAIRS_MIN:
        return "staircase", "stairs"
    if area("railing", "bannister") >= RAIL_INDOOR_MIN:
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
