"""Plan 1.6 (3D scene tours): old plans keep their bytes and hash, a 1.6 plan round-trips byte-exact
through the planner schema and validates in the contract, and the 1.6 rules hold."""
import copy
import json
import os

import pytest

from tour_worker.contract.plan import compute_plan_hash, parse_plan
from tour_worker.planner.schemas import ShotPlan

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures", "plans")


def _load(name):
    return json.load(open(os.path.join(FIXTURES, name)))


def _canon(d):
    return json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@pytest.mark.parametrize("name", ["plan_1_5_turkey_trot.json", "plan_1_6_box_room.json"])
def test_fixture_plans_round_trip_byte_exact_and_keep_their_hash(name):
    p = _load(name)
    assert compute_plan_hash(p) == p["provenance"]["plan_hash"]
    again = ShotPlan.model_validate(p).model_dump(mode="json")
    assert _canon(again) == _canon(p)
    parse_plan(p)


def test_a_photo_plan_has_no_scene_key_and_its_shots_keep_image_and_source():
    p = _load("plan_1_5_turkey_trot.json")
    assert "scene" not in p
    assert all("image_uri" in s and "source" in s for s in p["shots"])


def test_a_3d_plan_has_no_photo_fields_on_its_shots():
    p = _load("plan_1_6_box_room.json")
    assert all("image_uri" not in s and "source" not in s for s in p["shots"])
    assert "json_uri" not in p["scene"]          # unset scene.json fields are left out


def _rehash(p):
    p["provenance"]["plan_hash"] = compute_plan_hash(p)
    return p


def _bad(mutate, name="plan_1_6_box_room.json"):
    p = copy.deepcopy(_load(name))
    mutate(p)
    with pytest.raises(ValueError):
        parse_plan(_rehash(p))


def test_16_needs_a_scene_and_mode_cinematic_3d():
    _bad(lambda p: p.pop("scene"))
    _bad(lambda p: p.update(mode="cinematic"))


def test_scene_fields_are_checked():
    _bad(lambda p: p["scene"].update(format="obj"))
    _bad(lambda p: p["scene"].update(sha256="xyz"))
    _bad(lambda p: p["scene"].update(uri="ftp://x/y.ply"))
    _bad(lambda p: p["scene"].update(json_uri="s3://b/scene.json"))      # without its sha256
    _bad(lambda p: p["scene"].update(up_axis=[0, 0, 0]))
    _bad(lambda p: p["scene"].update(up_axis=[0, 1]))


def test_3d_shots_have_no_photo_or_motion_regions():
    _bad(lambda p: p["shots"][0].update(image_uri="s3://b/p.jpg"))
    _bad(lambda p: p["shots"][0].update(source={"width": 10, "height": 10}))
    _bad(lambda p: p["shots"][0].update(motion_regions=[{"id": "r", "effect": "calm_water", "mask_uri": "s3://m",
                                                         "bbox": [0, 0, 1, 1], "amplitude_px": 1, "freq_hz": 0.5,
                                                         "anchor": "none", "seed": 1}]))


def test_bezier_path_rules():
    _bad(lambda p: p["shots"][0]["camera"].update(hfov_deg=70.0))                   # one value
    _bad(lambda p: p["shots"][0]["camera"].update(hfov_deg=[10.0, 70.0]))           # out of range
    _bad(lambda p: p["shots"][0]["camera"]["segments"][0]["eye"].pop())            # 3 points
    _bad(lambda p: p["shots"][0]["camera"]["segments"][0]["eye"][1].append(1.0))   # a 4D point
    _bad(lambda p: p["shots"][0]["camera"]["segments"][0]["eye"].__setitem__(1, [float("nan"), 0, 0]))
    _bad(lambda p: p["shots"][1]["camera"]["segments"][1]["eye"][0].__setitem__(0, 9.0))   # not joined
    _bad(lambda p: p["shots"][0]["camera"].update(segments=p["shots"][0]["camera"]["segments"] * 4))
    _bad(lambda p: p["shots"][0]["camera"].update(projection="rotate"))
    _bad(lambda p: p["shots"][0]["camera"].update(punches=[{"at_s": 1.0, "amount": 0.02, "decay_s": 0.3}]))


def test_3d_cameras_and_scenes_are_16_only_and_photo_shots_still_need_their_photo():
    def to_15(p):
        p.update(plan_version="1.5")
    _bad(to_15)
    _bad(lambda p: p.update(scene={"uri": "s3://b/s.ply", "sha256": "ab" * 32, "format": "ply"}), "plan_1_5_turkey_trot.json")
    _bad(lambda p: p["shots"][0].pop("image_uri"), "plan_1_5_turkey_trot.json")
    _bad(lambda p: p["shots"][0].pop("source"), "plan_1_5_turkey_trot.json")


def test_photo_cameras_reject_the_3d_shapes():
    def kf_list_hfov(p):
        p["shots"][0]["camera"].update(projection="rotate", hfov_deg=[60.0, 70.0])
    _bad(kf_list_hfov, "plan_1_5_turkey_trot.json")
