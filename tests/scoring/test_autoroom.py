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


def test_stairs_and_hallways_play_between_the_living_spaces_and_the_bedrooms():
    """2026-11-20: a tour with no exterior opened on its spiral stair and landing (typed entry /
    hallway). Hallways and stairs are circulation: after the living spaces, before the bedrooms;
    a real entry (foyer) still opens."""
    from tour_worker.scoring.rooms import stage_for, stage_index
    order = sorted(["bedroom", "staircase", "hallway", "living_room", "kitchen", "foyer"],
                   key=lambda t: stage_index(stage_for(t)))
    assert order[0] == "foyer"
    assert order.index("staircase") > order.index("kitchen") and order.index("hallway") < order.index("bedroom")
