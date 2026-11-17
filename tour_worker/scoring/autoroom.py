"""Automatic room classification (SOW IN-04: "Autopilot step 1 classifies rooms automatically"), from
what Mask2Former sees in the photo (tour_segment's scene summary: each class's share of the photo
and how many of it). Used only where the upload carries no room type (IN-03: the user's choice
always wins); every room it names is flagged room_auto:<photo>:<type>, so it can be reviewed.

Rules, first match wins (the same dominance the 3D planner's room typing uses):
  exterior     sky or a building in view and no ceiling (a pool in view: pool_area)
  bathroom     a toilet, bath or shower
  bedroom      a bed taking BED_MIN of the photo
  kitchen      two kinds of kitchen fitting, one of them for cooking
  living_room  a sofa taking SOFA_MIN, or a TV / fireplace with an armchair
  dining_room  a table with 3+ chairs
  home_office  a desk, a computer or a monitor
  none         nothing a viewer would name: the photo stays unknown (never defaulted)
"""
from __future__ import annotations

BED_MIN, SOFA_MIN = 0.06, 0.05
SKY_MIN, BUILDING_MIN, CEILING_MAX = 0.06, 0.12, 0.02
COOKING = ("stove", "oven", "hood", "dishwasher", "microwave", "kitchen island")
KITCHEN = COOKING + ("sink", "refrigerator", "countertop", "counter")
BATH = ("toilet", "bathtub", "shower")


def classify(scene: dict | None) -> tuple[str | None, str]:
    """(room type or None, why) from a scene summary {"classes": {label: {area, count, ...}}}."""
    if not scene:
        return None, "no_scene"
    c = scene.get("classes") or {}
    area = lambda k: float((c.get(k) or {}).get("area", 0.0))       # noqa: E731
    count = lambda k: int((c.get(k) or {}).get("count", 0))          # noqa: E731
    has = lambda k: k in c                                            # noqa: E731
    if area("ceiling") <= CEILING_MAX and (area("sky") >= SKY_MIN or area("house") + area("building") >= BUILDING_MIN):
        return ("pool_area", "pool") if area("swimming pool") > 0.03 else ("exterior", "sky/building, no ceiling")
    if any(has(k) for k in BATH) and area("bed") < BED_MIN:
        return "bathroom", "bath fixture"
    if area("bed") >= BED_MIN:
        return "bedroom", "bed"
    kinds = [k for k in KITCHEN if has(k)]
    if len(kinds) >= 2 and any(k in COOKING for k in kinds):
        return "kitchen", "+".join(kinds[:3])
    if area("sofa") >= SOFA_MIN or ((has("television receiver") or has("fireplace")) and has("armchair")):
        return "living_room", "sofa / tv"
    if has("table") and count("chair") >= 3:
        return "dining_room", "table and chairs"
    if has("desk") or has("computer") or has("monitor"):
        return "home_office", "desk"
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
