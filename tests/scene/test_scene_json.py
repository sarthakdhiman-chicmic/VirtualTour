"""scene.json is optional, but when given it is validated strictly."""
import copy
import json

import pytest

from tour_worker.scene import SceneInvalid
from tour_worker.scene.scene_json import parse_scene_json

SHA = "cd" * 32
GOOD = {
    "version": 1, "splat_sha256": SHA, "up_axis": [0, 0, 1], "metres_per_unit": 1.0, "floor_height": 0.0,
    "capture_poses": [{"position": [0, 0, 1.5], "forward": [0, 1, 0]}],
    "rooms": [{"room_key": "kitchen_1", "room_type": "kitchen", "label": "Kitchen",
               "anchors": [{"position": [0, 0, 1.5], "look_at": [0, 3, 1.2], "hfov_deg": 70, "hero": True}]}],
}


def test_a_good_scene_json_parses_from_bytes_and_dict():
    for data in (json.dumps(GOOD).encode(), GOOD):
        sj = parse_scene_json(data, model_sha256=SHA)
        assert sj.rooms[0].anchors[0].hero and sj.capture_poses[0].forward == [0, 1, 0]


def _bad(mutate, **kw):
    d = copy.deepcopy(GOOD)
    mutate(d)
    with pytest.raises(SceneInvalid):
        parse_scene_json(d, **kw)


def test_strict_validation():
    _bad(lambda d: d.update(extra=1))                                  # unknown key
    _bad(lambda d: d.update(version=2))
    _bad(lambda d: d.update(up_axis=[0, 0, 0]))
    _bad(lambda d: d.update(metres_per_unit=0))
    _bad(lambda d: d.update(splat_sha256="nothex"))
    _bad(lambda d: d["rooms"][0].update(room_type="spaceship"))
    _bad(lambda d: d["rooms"][0].update(label=""))
    _bad(lambda d: d["rooms"][0].update(anchors=[]))
    _bad(lambda d: d["rooms"][0]["anchors"][0].update(look_at=[0, 0, 1.5]))   # looks at itself
    _bad(lambda d: d["rooms"][0]["anchors"][0].update(hfov_deg=150))
    _bad(lambda d: d["rooms"][0]["anchors"][0].update(position=[0, 0]))
    _bad(lambda d: d["rooms"].append(copy.deepcopy(d["rooms"][0])))           # duplicate room_key
    _bad(lambda d: d.update(rooms=[]))


def test_scene_json_must_name_the_scene_file_it_was_made_for():
    with pytest.raises(SceneInvalid):
        parse_scene_json(GOOD, model_sha256="ef" * 32)


def test_garbage_is_scene_invalid():
    for data in (b"not json", b"[1, 2]", "null"):
        with pytest.raises(SceneInvalid):
            parse_scene_json(data)
