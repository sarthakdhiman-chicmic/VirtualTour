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
