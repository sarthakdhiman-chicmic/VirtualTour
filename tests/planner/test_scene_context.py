"""Scene context (planner/scene_context.py): the planner's inputs come from what each photo shows
(user 2026-10-19: "make sure the planner knows all the right input from the image context")."""

import math

import pytest
from test_cinematic import check_contract, frames_for, shot, tour
from test_plan_integration import _assets, report  # noqa: F401  (fixture)

from tour_worker.planner.allocate import allocate_shots, natural_length
from tour_worker.planner.cinematic import (
    ZOOM_SHARE,
    candidates,
    category,
    descends_from_interest,
    plan_cameras,
    zoom_led,
)
from tour_worker.planner.config import PlannerConfig
from tour_worker.planner.scene_context import class_weight, from_scene


def cls(area, bbox):
    x0, y0, x1, y1 = bbox
    return {"area": area, "count": 1, "bbox": list(bbox),
            "centroid": [(x0 + x1) / 2, (y0 + y1) / 2]}


# an entry: a tall front door mid-frame, lanterns and urns either side, steps and paving below
ENTRY = {"classes": {
    "door-stuff": cls(0.06, (0.40, 0.20, 0.62, 0.60)),
    "light": cls(0.01, (0.17, 0.17, 0.23, 0.33)),
    "potted plant": cls(0.03, (0.20, 0.33, 0.32, 0.62)),
    "vase": cls(0.02, (0.68, 0.45, 0.76, 0.63)),
    "stairs": cls(0.08, (0.20, 0.60, 0.82, 0.80)),
    "pavement-merged": cls(0.25, (0.0, 0.78, 1.0, 1.0)),
    "wall-other-merged": cls(0.30, (0.0, 0.0, 1.0, 0.65)),
}}
# a plain facade: the house and a tree, lots of lawn and sky
FACADE = {"classes": {
    "house": cls(0.30, (0.15, 0.25, 0.85, 0.70)),
    "tree-merged": cls(0.08, (0.0, 0.10, 0.18, 0.70)),
    "grass-merged": cls(0.30, (0.0, 0.70, 1.0, 1.0)),
    "sky-other-merged": cls(0.25, (0.0, 0.0, 1.0, 0.25)),
}}
# a staged living room: many things across the frame
LIVING = {"classes": {k: cls(0.03, b) for k, b in {
    "couch": (0.10, 0.55, 0.45, 0.80), "table-merged": (0.40, 0.65, 0.60, 0.80),
    "light": (0.45, 0.05, 0.55, 0.20), "window-other": (0.70, 0.20, 0.95, 0.60),
    "chair": (0.70, 0.55, 0.85, 0.80), "tv": (0.05, 0.25, 0.25, 0.45),
    "mirror-stuff": (0.30, 0.20, 0.40, 0.40), "potted plant": (0.88, 0.45, 0.98, 0.75),
    "fireplace": (0.05, 0.40, 0.25, 0.60),
}.items()}}


def test_class_names_from_either_oneformer():
    assert class_weight("door-stuff") == class_weight("door") == 1.0
    assert class_weight("wall-other-merged") == class_weight("floor-wood") == 0.0
    assert class_weight("pavement-merged") == class_weight("sky-other-merged") == 0.0
    assert class_weight("rug-merged") == 0.0
    assert 0 < class_weight("some-new-thing") < 1


def test_entry_interest_is_the_door_not_the_paving():
    c = from_scene(ENTRY, peak_x=0.5, peak_y=0.8)
    assert c.semantic and c.interest_y < 0.6          # the steps / paving carry no interest
    assert c.tall_subject                             # the door
    assert c.open_floor > 0.5


def test_richness_separates_plain_and_rich_photos():
    plain, rich = from_scene(FACADE), from_scene(LIVING)
    assert plain.richness < 0.2 < 0.7 < rich.richness
    cfg = PlannerConfig()
    assert natural_length("hero", plain, cfg) < 3.5 < 4.5 < natural_length("hero", rich, cfg)
    assert natural_length("detail", plain, cfg) == pytest.approx(cfg.normal_shot_s[0], abs=0.15)


def test_no_scene_summary_falls_back_to_saliency():
    c = from_scene(None, peak_x=0.3, peak_y=0.4, spread=0.47)
    assert not c.semantic and (c.interest_x, c.interest_y) == (0.3, 0.4)


def test_natural_lengths_nothing_stretched(report):  # noqa: F811
    """Shots keep their own lengths; no lower limit, max_total_s is the ceiling."""
    cfg = PlannerConfig()
    shots, _ = allocate_shots(report, cfg, {})
    for s in shots[:-1]:
        lo, hi = cfg.hero_shot_s if s.role == "hero" else cfg.normal_shot_s
        assert lo - 1e-6 <= s.final_duration <= hi + 1e-6
    assert sum(s.final_duration for s in shots) <= cfg.max_total_s
    tight = PlannerConfig(max_total_s=20.0)
    few, warns = allocate_shots(report, tight, {})
    assert sum(s.final_duration for s in few) <= 20.0 and "trimmed" in warns


def test_entry_never_sinks_onto_the_steps():
    """Review 2026-10-19: the entry pushed DOWN onto the steps; moves now rise to the door."""
    s = shot(2, "entry", cls="medium", role="detail", dur=3.4, px=0.5, py=0.8)
    s.ctx = from_scene(ENTRY, 0.5, 0.8)
    s.peak_x, s.peak_y = s.ctx.interest_x, s.ctx.interest_y
    for c in candidates(s, category(s, False), 1920, 1080):
        assert not descends_from_interest(c["keyframes"], s), (c["variant"], c["mode"])


def test_tall_subject_gets_a_rising_crane():
    """A tall subject (the front door) gets the crane reveal rising to it (review 2026-10-19:
    the constant-zoom pedestal and the gimbal track read worse on the entry)."""
    s = shot(1, "entry", cls="medium", role="hero", dur=3.6)
    s.ctx = from_scene(ENTRY, 0.5, 0.5)
    shots = [s, shot(2, "living_room")]
    res = plan_cameras(shots, frames_for(shots), 1920, 1080)
    assert res.notes[0]["variant"] == "crane" and res.notes[0]["mode"] == "reveal"
    k = res.cameras[0].keyframes
    assert k[-1].cy < k[0].cy                                              # rises


def test_no_bank_and_exteriors_keep_the_older_moves():
    shots = tour(n=12, dur=3.6)
    for s in shots:
        s.ctx = from_scene(LIVING)
    ext = shot(1, "front_exterior", stage="exterior_open", dur=3.0)
    ext.ctx = from_scene(FACADE)
    shots = [ext, *shots]
    res = plan_cameras(shots, frames_for(shots), 1920, 1080)
    assert res.notes[0]["variant"] in ("rise", "tilt_up", "pull_back")
    for n, cam in zip(res.notes, res.cameras, strict=True):
        assert n["variant"] not in ("gimbal_orbit", "pedestal")
        if n["variant"] == "gimbal_track":
            assert not any(k.roll_deg for k in cam.keyframes)                # level


def test_aerials_may_still_come_down():
    s = shot(1, "aerial", stage="exterior_open", dur=4.0)
    s.ctx = from_scene(FACADE)
    kfs = [{"t": 0, "cx": 0.5, "cy": 0.4, "zoom": 1.05},
           {"t": 1, "cx": 0.5, "cy": 0.6, "zoom": 1.2}]
    assert descends_from_interest(kfs, s, "medium")
    assert not descends_from_interest(kfs, s, "aerial")


def test_zoom_moves_are_a_minority_and_never_back_to_back():
    shots = tour(n=12, dur=3.6)
    for s in shots:
        s.ctx = from_scene(LIVING)
    res = plan_cameras(shots, frames_for(shots), 1920, 1080)
    zl = [n["variant"] in ("pull_back", "crane") or n["mode"] == "push" for n in res.notes]
    assert sum(zl) <= math.ceil(ZOOM_SHARE * len(shots))
    assert not any(a and b for a, b in zip(zl, zl[1:], strict=False))
    assert "gimbal_track" in {n["variant"] for n in res.notes}
    assert zoom_led({"variant": "slider", "mode": "push"})
    check_contract(shots, res)


@pytest.mark.parametrize("w, h", [(1536, 1152), (1920, 1280), (4000, 3000)])
def test_gimbal_moves_fit_low_resolution_photos(w, h):
    """1536 px listing photos leave ~1.25x zoom: the banked orbit must still fit, no fallback."""
    shots = [shot(i + 1, "living_room", dur=3.6, w=w, h=h) for i in range(4)]
    for s in shots:
        s.ctx = from_scene(LIVING)
    res = plan_cameras(shots, frames_for(shots), 1920, 1080)
    assert not res.warnings
    check_contract(shots, res)


def test_interest_above_a_foreground_of_paving_means_rise_to_it():
    """An entry whose glass door OneFormer calls a window (not tall): the steps and paving in
    front still say the camera should come up to the interest (Turkey mix entry, 2026-10-19)."""
    walk_up = {"classes": {k: v for k, v in ENTRY["classes"].items() if k != "door-stuff"}}
    walk_up["classes"]["window-other"] = cls(0.08, (0.30, 0.20, 0.75, 0.60))
    walk_up["classes"]["pavement-merged"] = cls(0.30, (0.0, 0.70, 1.0, 1.0))
    c = from_scene(walk_up)
    assert not c.tall_subject and c.rises_to_interest
    no_tall = {"classes": {k: v for k, v in LIVING["classes"].items() if k != "window-other"}}
    assert not from_scene(no_tall).rises_to_interest       # no floor in front, nothing tall


# a narrow bathroom: mirror over a vanity, a shower door beside it - content stacked up
BATH = {"classes": {
    "mirror-stuff": cls(0.06, (0.35, 0.10, 0.60, 0.42)),
    "light": cls(0.01, (0.40, 0.03, 0.55, 0.09)),
    "counter": cls(0.05, (0.30, 0.50, 0.65, 0.62)),
    "sink": cls(0.01, (0.42, 0.52, 0.52, 0.57)),
    "cabinet-merged": cls(0.06, (0.30, 0.62, 0.65, 0.92)),
    "door-stuff": cls(0.08, (0.68, 0.12, 0.88, 0.90)),
    "wall-tile": cls(0.30, (0.0, 0.0, 1.0, 1.0)),
}}


def test_narrow_tall_spaces_get_the_vertical_glide():
    """User 2026-10-19: shots with little width but enough vertical content (the bathroom) need
    their own move: the vertical glide rises up the room to its interest."""
    s = shot(1, "bathroom", cls="tight", role="detail", dur=3.3)
    s.ctx = from_scene(BATH)
    shots = [shot(0, "bedroom", cls="medium"), s, shot(2, "living_room")]
    res = plan_cameras(shots, frames_for(shots), 1920, 1080)
    assert res.notes[1]["variant"] == "vertical_glide"
    k = res.cameras[1].keyframes
    assert k[-1].cy < k[0].cy and k[-1].zoom < k[0].zoom          # rises, easing out
    assert abs(k[-1].cx - k[0].cx) < 0.05                         # vertical, not sideways
    wide_room = shot(1, "bedroom", cls="medium", role="detail", dur=3.3)
    wide_room.ctx = from_scene(LIVING)
    wide = plan_cameras([shots[0], wide_room, shots[2]], frames_for(shots), 1920, 1080)
    assert wide.notes[1]["variant"] != "vertical_glide"
    check_contract(shots, res)
