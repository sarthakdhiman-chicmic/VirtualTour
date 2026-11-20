"""Automatic room classification (scoring/autoroom.py, SOW IN-04): from what the photo shows, only
where the upload has no room type - the user's choice always wins, and every guess is flagged."""
from tour_worker.scoring.autoroom import classify, fill_room_types


def _scene(**areas):
    return {"classes": {k.replace("_", " "): ({"area": v, "count": 1} if not isinstance(v, tuple)
                                              else {"area": v[0], "count": v[1]}) for k, v in areas.items()}}


def test_each_room_type_from_what_it_shows():
    assert classify(_scene(sky=0.3, house=0.2))[0] == "exterior"
    assert classify(_scene(sky=0.3, swimming_pool=0.1))[0] == "pool_area"
    assert classify(_scene(toilet=0.05, sink=0.03, ceiling=0.1))[0] == "bathroom"
    assert classify(_scene(bed=0.25, ceiling=0.1))[0] == "bedroom"
    assert classify(_scene(stove=0.04, sink=0.02, refrigerator=0.05, ceiling=0.1))[0] == "kitchen"
    assert classify(_scene(sink=0.02, refrigerator=0.05, ceiling=0.1))[0] is None      # no cooking: not a kitchen
    assert classify(_scene(sofa=0.15, ceiling=0.1))[0] == "living_room"
    assert classify(_scene(table=0.1, chair=(0.08, 4), ceiling=0.1))[0] == "dining_room"
    assert classify(_scene(desk=0.06, chair=(0.03, 1), ceiling=0.1))[0] == "home_office"
    assert classify(_scene(wall=0.6, floor=0.2))[0] is None                            # nothing named
    assert classify(None)[0] is None
    # a bed beats a stray bath fixture (an en-suite seen through a door); a ceiling means indoors
    assert classify(_scene(bed=0.2, toilet=0.01))[0] == "bedroom"
    assert classify(_scene(sky=0.1, ceiling=0.2, sofa=0.1))[0] == "living_room"


def test_only_untyped_photos_are_named_and_each_guess_is_flagged():
    photos = [("a", b"", "Kitchen"), ("b", b"", None), ("c", b"", None)]
    scenes = {"a": _scene(bed=0.3), "b": _scene(bed=0.3), "c": _scene(wall=0.9)}
    out, warn = fill_room_types(photos, scenes)
    assert [t for _, _, t in out] == ["Kitchen", "bedroom", None]
    assert warn == ["room_auto:b:bedroom"]


def test_class_names_are_matched_trimmed_ade20k_spells_bed_with_a_trailing_space():
    assert classify({"classes": {"bed ": {"area": 0.2, "count": 1}, "ceiling": {"area": 0.1, "count": 1}}})[0] == "bedroom"


def test_more_of_the_room_type_enum_from_what_the_photo_shows():
    assert classify(_scene(house=0.43, stairs=0.11, sky=0.005))[0] == "entry"            # steps up to the door
    assert classify(_scene(sky=0.2, railing=0.1, floor=0.2))[0] == "balcony"
    assert classify(_scene(house=0.5, sky=0.2))[0] == "front_exterior"
    assert classify(_scene(ceiling=0.1, railing=0.07, tree=0.13, sky=0.05, sofa=0.12))[0] == "outdoor_patio"   # covered
    assert classify(_scene(ceiling=0.1, railing=0.15, sky=0.17, chair=0.1))[0] == "balcony"
    assert classify(_scene(ceiling=0.13, swimming_pool=0.3))[0] == "pool_area"          # indoors
    assert classify(_scene(ceiling=0.1, sofa=0.17, toilet=0.004))[0] == "living_room"   # a stray fixture
    assert classify(_scene(ceiling=0.1, cabinet=0.03, stool=(0.04, 3)))[0] == "kitchen"  # stools at a counter
    assert classify(_scene(ceiling=0.1, table=0.1, chair=(0.08, 6), stool=(0.007, 2)))[0] == "dining_room"
    assert classify(_scene(ceiling=0.1, armchair=(0.15, 4)))[0] == "living_room"
    assert classify(_scene(ceiling=0.1, washer=0.05))[0] == "laundry_room"
    assert classify(_scene(ceiling=0.1, car=0.3))[0] == "garage"
    assert classify(_scene(ceiling=0.1, pool_table=0.1))[0] == "game_room"
    assert classify(_scene(ceiling=0.1, bookcase=0.2))[0] == "library"
    assert classify(_scene(ceiling=0.1, cradle=0.05))[0] == "nursery"
    assert classify(_scene(ceiling=0.1, wardrobe=0.2))[0] == "walk_in_closet"
    assert classify(_scene(ceiling=0.1, stairs=0.1))[0] == "staircase"
    assert classify(_scene(ceiling=0.1, railing=0.18, wall=0.3))[0] == "hallway"         # a landing


def test_every_type_it_names_is_a_known_room_type():
    """The backend's TourRoomType values are the scorer's tables (scoring/rooms.py); every rule's
    answer must be one of them."""
    import inspect
    import re
    from tour_worker.scoring import autoroom
    from tour_worker.scoring.rooms import ROOM_STAGE, ROOM_TIER
    named = set(re.findall(r'return \(?"([a-z_]+)"', inspect.getsource(autoroom.classify)))
    assert named and named <= set(ROOM_TIER) | set(ROOM_STAGE), named - (set(ROOM_TIER) | set(ROOM_STAGE))
