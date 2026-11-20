"""Automatic room classification (scoring/autoroom.py, SOW IN-04): from what the photo shows, only
where the upload has no room type - the user's choice always wins, and every guess is flagged.
Scene classes are OneFormer COCO's, by their common names (vision/segment.COMMON)."""
from tour_worker.scoring.autoroom import classify, fill_room_types


def _scene(**areas):
    return {"classes": {k.replace("_", " "): ({"area": v, "count": 1} if not isinstance(v, tuple)
                                              else {"area": v[0], "count": v[1]}) for k, v in areas.items()}}


def test_each_room_type_from_what_it_shows():
    assert classify(_scene(sky=0.3, house=0.2))[0] == "exterior"
    assert classify(_scene(sky=0.3, water=0.1))[0] == "pool_area"
    assert classify(_scene(toilet=0.05, sink=0.03, ceiling=0.1))[0] == "bathroom"
    assert classify(_scene(bed=0.25, ceiling=0.1))[0] == "bedroom"
    assert classify(_scene(oven=0.04, sink=0.02, refrigerator=0.05, ceiling=0.1))[0] == "kitchen"
    assert classify(_scene(sink=0.02, counter=0.05, ceiling=0.1))[0] is None             # no cooking, no fridge
    assert classify(_scene(sofa=0.15, ceiling=0.1))[0] == "living_room"
    assert classify(_scene(table=0.1, chair=(0.08, 4), ceiling=0.1))[0] == "dining_room"
    assert classify(_scene(computer=0.03, chair=(0.03, 1), ceiling=0.1))[0] == "home_office"
    assert classify(_scene(wall=0.6, floor=0.2))[0] is None                            # nothing named
    assert classify(None)[0] is None
    # a bed beats a stray bath fixture (an en-suite seen through a door); a ceiling means indoors
    assert classify(_scene(bed=0.2, toilet=0.01))[0] == "bedroom"
    assert classify(_scene(sky=0.1, ceiling=0.2, sofa=0.1))[0] == "living_room"


def test_class_names_are_matched_trimmed():
    assert classify({"classes": {"bed ": {"area": 0.2, "count": 1}, "ceiling": {"area": 0.1, "count": 1}}})[0] == "bedroom"


def test_more_of_the_room_type_enum_from_what_the_photo_shows():
    assert classify(_scene(house=0.43, stairs=0.11, sky=0.005))[0] == "entry"            # steps up to the door
    assert classify(_scene(sky=0.2, fence=0.1, floor=0.2))[0] == "balcony"               # COCO's fence: railings
    assert classify(_scene(sky=0.2, grass=0.2, tree=0.15))[0] == "backyard"
    assert classify(_scene(house=0.5, sky=0.2))[0] == "front_exterior"
    assert classify(_scene(ceiling=0.1, fence=0.07, tree=0.13, sky=0.05, sofa=0.12))[0] == "outdoor_patio"   # covered
    assert classify(_scene(ceiling=0.1, fence=0.15, sky=0.17, chair=0.1))[0] == "balcony"
    assert classify(_scene(ceiling=0.13, water=0.3))[0] == "pool_area"                  # indoors
    assert classify(_scene(ceiling=0.1, sofa=0.17, toilet=0.004))[0] == "living_room"   # a stray fixture
    assert classify(_scene(ceiling=0.1, towel=0.01, sink=0.02, mirror=0.1))[0] == "bathroom"
    assert classify(_scene(ceiling=0.1, refrigerator=0.05, counter=0.04))[0] == "kitchen"
    assert classify(_scene(ceiling=0.1, television_receiver=0.04, chair=(0.05, 2)))[0] == "living_room"
    assert classify(_scene(ceiling=0.1, car=0.3))[0] == "garage"
    assert classify(_scene(ceiling=0.1, shelf=0.2, book=0.03))[0] == "library"
    assert classify(_scene(ceiling=0.1, stairs=0.1))[0] == "staircase"
    assert classify(_scene(ceiling=0.1, fence=0.18, wall=0.3))[0] == "hallway"           # a landing


def test_every_type_it_names_is_a_known_room_type():
    """The backend's TourRoomType values are the scorer's tables (scoring/rooms.py); every rule's
    answer must be one of them."""
    import inspect
    import re
    from tour_worker.scoring import autoroom
    from tour_worker.scoring.rooms import ROOM_STAGE, ROOM_TIER
    named = set(re.findall(r'return \(?"([a-z_]+)"', inspect.getsource(autoroom.classify)))
    assert named and named <= set(ROOM_TIER) | set(ROOM_STAGE), named - (set(ROOM_TIER) | set(ROOM_STAGE))


def test_only_untyped_photos_are_named_and_each_guess_is_flagged():
    photos = [("a", b"", "Kitchen"), ("b", b"", None), ("c", b"", None)]
    scenes = {"a": _scene(bed=0.3), "b": _scene(bed=0.3), "c": _scene(wall=0.9)}
    out, warn = fill_room_types(photos, scenes)
    assert [t for _, _, t in out] == ["Kitchen", "bedroom", None]
    assert warn == ["room_auto:b:bedroom"]


def test_the_tour_plays_entrance_living_kitchen_dining_primary_suite_then_the_rest():
    """The user's order (2026-11-20): the entrance (front of the home, then the entry), living room,
    kitchen, dining, the primary bedroom and its bathroom, then the other rooms; stairs and landings
    lead from the primary suite to the others (never the opening); every other exterior plays after
    the house."""
    from tour_worker.scoring.rooms import stage_for, stage_index
    want = ["front_exterior", "foyer", "living_room", "kitchen", "dining_room", "primary_bedroom",
            "primary_bathroom", "staircase", "bedroom", "home_office", "garage", "backyard", "exterior"]
    stages = [stage_index(stage_for(t)) for t in want]
    assert stages == sorted(stages)
    assert stage_for("primary_bedroom") == stage_for("primary_bathroom")
    assert stage_for(None) == stage_for("bedroom")                         # unknown: with the other rooms


def _c(area, bbox=None, centroid=None):
    d = {"area": area, "count": 1}
    if bbox:
        d["bbox"] = bbox
    if centroid:
        d["centroid"] = centroid
    return d


def test_a_desk_at_a_window_and_a_stair_hall_are_indoors():
    """2026-11-20: OneFormer saw only sky, buildings, a chair and a book at a desk by a window, and
    a stair hall with clouds in its window had almost no ceiling: both were called exteriors."""
    from tour_worker.scoring.autoroom import is_indoors
    desk = {"classes": {"sky": _c(0.44), "building": _c(0.13), "chair": _c(0.04), "book": _c(0.03)}}
    stairs = {"classes": {"wall": _c(0.34), "floor": _c(0.19), "stairs": _c(0.12), "sky": _c(0.04), "ceiling": _c(0.01)}}
    assert is_indoors(desk) and is_indoors(stairs)
    assert classify(desk)[0] not in ("exterior", "front_exterior")
    assert classify(stairs)[0] == "staircase"
    assert is_indoors({"classes": {"sky": _c(0.3), "house": _c(0.2), "tree": _c(0.2)}}) is False
    assert is_indoors(None) is None


def test_a_city_from_a_terrace_is_not_the_front_of_the_house_and_a_path_to_the_door_is():
    terrace = {"classes": {"sky": _c(0.47, [0, 0, 0.95, 0.52]), "building": _c(0.44, [0, 0, 1, 1], [0.46, 0.70]),
                           "sidewalk": _c(0.09)}}
    assert classify(terrace)[0] == "balcony"
    front = {"classes": {"sky": _c(0.30, [0, 0, 1, 0.39]), "house": _c(0.17, [0, 0.2, 1, 0.7], [0.48, 0.42]),
                         "sidewalk": _c(0.37), "tree": _c(0.08)}}
    assert classify(front)[0] == "front_exterior"
