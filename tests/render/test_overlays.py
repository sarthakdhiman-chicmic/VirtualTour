"""tour_worker.render.overlays: room labels (timing, placement, scale, fonts) and the staged disclosure,
and the plan contract's checks on them."""
import types

import numpy as np
import pytest

from tour_worker.contract.plan import LABEL_FONTS, LABEL_PLACEMENTS, RoomLabels, compute_plan_hash, parse_plan
from tour_worker.render.overlays import (END_GAP, HOLD_MAX, LABEL_IN, OUT_S, WRITE_AT, WRITE_S, Overlays)

W, H, FPS = 1920, 1080, 30
FRAME = np.full((H, W, 3), 128, np.uint8)       # mid grey: the text and the gradient both show


def _overlays(shots, frames, placement="bottom_left", scale=0.8, font="sacramento", labels=True):
    """Overlays for shots [(room_key, caption, staged)] of `frames` frames each, back to back."""
    out = types.SimpleNamespace(width=W, height=H, fps=FPS,
                                room_labels=RoomLabels(font=font, scale=scale, placement=placement) if labels else None)
    sh = [types.SimpleNamespace(room_key=k, caption=c, staged=s) for k, c, s in shots]
    tl, at = [], 0
    for n in frames:
        tl.append({"render_from": at, "render_frames": n})
        at += n
    return Overlays(types.SimpleNamespace(output=out, shots=sh), tl)


def _changed(a, b=FRAME):
    """The box (x0, y0, x1, y1) of pixels that differ by more than the gradient's shading (at most
    40% of 128 = 51 levels), or None."""
    d = np.abs(a.astype(int) - b.astype(int)).max(axis=2) > 60
    ys, xs = np.nonzero(d)
    return None if not len(ys) else (xs.min(), ys.min(), xs.max(), ys.max())


def test_a_label_animates_in_holds_and_is_gone_before_the_room_ends():
    ov = _overlays([("k", "Kitchen", False)], [6 * FPS])
    f = lambda t: ov.apply(FRAME, round(t * FPS))      # noqa: E731
    assert f(LABEL_IN - 0.1) is FRAME                       # nothing yet
    half = f(WRITE_AT + WRITE_S / 2)                        # the name is half written
    full = f(WRITE_AT + WRITE_S + 0.3)
    hb, fb = _changed(half), _changed(full)
    assert hb[2] < fb[2] - 40                               # written left to right
    end = LABEL_IN + HOLD_MAX + OUT_S
    assert f(end + 0.05) is FRAME                           # faded out after 3.6 s + 0.5 s


def test_a_short_room_fades_out_before_its_end_and_one_too_short_gets_no_label():
    ov = _overlays([("k", "Kitchen", False), ("b", "Bedroom", False)], [round(2.6 * FPS), round(1.5 * FPS)])
    assert len(ov.labels) == 1                              # 1.5 s cannot write a name out
    t_out = ov.labels[0][4]
    assert t_out == pytest.approx(2.6 - END_GAP - OUT_S)


def test_one_label_per_room_numbered_in_order():
    ov = _overlays([("k", "Kitchen", False), ("k", None, False), ("b", "Bedroom", False)], [120, 120, 120])
    assert [(i, n) for i, n, *_ in ov.labels] == [(1, "Kitchen"), (2, "Bedroom")]
    # the kitchen's label may hold across its second shot (the room), up to 3.6 s
    assert ov.labels[0][4] == pytest.approx(LABEL_IN + HOLD_MAX)


@pytest.mark.parametrize("placement", LABEL_PLACEMENTS)
def test_each_placement_puts_the_label_in_its_part_of_the_frame(placement):
    box = _changed(_overlays([("k", "Kitchen", False)], [180], placement=placement).apply(FRAME, 60))
    x0, y0, x1, y1 = box
    vert, horiz = placement.split("_")
    assert (y0 > H * 0.8) if vert == "bottom" else (y1 < H * 0.2)
    if horiz == "left":
        assert x1 < W * 0.3
    elif horiz == "right":
        assert x0 > W * 0.7
    else:
        assert abs((x0 + x1) / 2 - W / 2) < 40


def test_the_scale_sets_the_size_and_every_font_draws():
    heights = {}
    for scale in (0.6, 0.8, 1.0):
        x0, y0, x1, y1 = _changed(_overlays([("k", "Kitchen", False)], [180], scale=scale).apply(FRAME, 60))
        heights[scale] = y1 - y0
    assert heights[0.6] < heights[0.8] < heights[1.0]
    for font in LABEL_FONTS:
        assert _changed(_overlays([("k", "Kitchen", False)], [180], font=font).apply(FRAME, 60)) is not None


def test_the_staged_label_is_on_the_whole_shot_and_the_mls_copy_keeps_only_it():
    ov = _overlays([("k", "Kitchen", True), ("b", "Bedroom", False)], [150, 150])
    mls = ov.apply(FRAME, 75, captions=False)
    x0, y0, x1, y1 = _changed(mls)
    assert x0 > W * 0.75 and y0 > H * 0.85                  # bottom right, nothing else
    assert _changed(ov.apply(FRAME, 1, captions=False)) is None    # fading in from the cut
    assert ov.apply(FRAME, 200, captions=False) is FRAME    # the unstaged room: nothing in the MLS copy
    assert _changed(ov.apply(FRAME, 60)) != _changed(mls)   # the branded frame has the room label too


def test_the_staged_label_moves_left_when_the_room_label_sits_bottom_right():
    ov = _overlays([("k", "Kitchen", True)], [150], placement="bottom_right")
    x0, _, x1, _ = _changed(ov.apply(FRAME, 75, captions=False))
    assert x1 < W * 0.25


def test_no_labels_and_nothing_staged_is_no_overlay():
    assert not _overlays([("k", "Kitchen", False)], [150], labels=False).active


def test_the_same_frame_is_the_same_bytes():
    a = _overlays([("k", "Kitchen", True)], [150]).apply(FRAME, 45)
    b = _overlays([("k", "Kitchen", True)], [150]).apply(FRAME, 45)
    assert np.array_equal(a, b)


# ---- the plan contract ---------------------------------------------------------------------------

def _plan(output_extra=None, caption="Kitchen"):
    shot = {"shot_id": "s1", "order": 1, "asset_id": "a", "image_uri": "s3://b/i.jpg",
            "source": {"width": 1920, "height": 1080}, "room_key": "k", "room_type": "kitchen",
            "narrative_stage": "living", "role": "hero", "duration_s": 3.0,
            "camera": {"type": "ken_burns", "start": {"cx": 0.5, "cy": 0.5, "zoom": 1.0},
                       "end": {"cx": 0.5, "cy": 0.5, "zoom": 1.1}},
            "transition_in": {"type": "cut", "duration_s": 0.0}}
    if caption is not None:
        shot["caption"] = caption
    p = {"plan_version": "1.2", "listing_id": "L", "mode": "cinematic", "preset": "p",
         "output": {"width": 1920, "height": 1080, "fps": 30, "codec": "h264", **(output_extra or {})},
         "total_duration_s": 3.0, "shots": [shot], "warnings": [],
         "provenance": {"scoring_config_version": "0", "planner_config_version": "0"}}
    p["provenance"]["plan_hash"] = compute_plan_hash(p)
    return p


def test_the_contract_accepts_labels_and_rejects_bad_ones():
    ok = parse_plan(_plan({"room_labels": {"font": "pacifico", "scale": 0.6, "placement": "top_right"}}))
    assert ok.output.room_labels.font == "pacifico" and ok.shots[0].caption == "Kitchen"
    for bad in ({"font": "comic_sans"}, {"placement": "middle"}, {"scale": 0.5}, {"scale": 1.2}):
        with pytest.raises(ValueError, match="room_labels"):
            parse_plan(_plan({"room_labels": bad}))
    with pytest.raises(ValueError, match="no room_labels"):
        parse_plan(_plan())                                  # a caption needs the labels' settings
    with pytest.raises(ValueError, match="caption"):
        parse_plan(_plan({"room_labels": {}}, caption="x" * 41))
    assert not parse_plan(_plan(caption=None)).output.mls_copy


# ---- branding: title card, end card, corner logo --------------------------------------------------

def _branded(tmp_path, agent=True, logo=True, corner=True, title=True):
    from PIL import Image as PImage
    from tour_worker.contract.plan import Branding
    from tour_worker.io.images import branding_path
    imgs = str(tmp_path)
    (tmp_path / "branding").mkdir(exist_ok=True)
    PImage.new("RGB", (300, 400), (200, 60, 60)).save(branding_path(imgs, "s3://b/agent.jpg"), "PNG")
    PImage.new("RGBA", (400, 100), (60, 200, 60, 255)).save(branding_path(imgs, "s3://b/logo.png"), "PNG")
    b = Branding(address="1030 Turkey Trot" if title else None, locality="Greensboro, GA" if title else None,
                 agent={"name": "Jordan Avery", "phone": "555", "email": "j@x.com", "photo_uri": "s3://b/agent.jpg"} if agent else None,
                 logo_uri="s3://b/logo.png" if logo else None, corner_logo=corner)
    out = types.SimpleNamespace(width=W, height=H, fps=FPS, room_labels=RoomLabels(), branding=b)
    shots = [types.SimpleNamespace(room_key=k, caption=c, staged=False) for k, c in (("e", "The Residence"), ("k", "Kitchen"))]
    return Overlays(types.SimpleNamespace(output=out, shots=shots),
                    [{"render_from": 0, "render_frames": 150}, {"render_from": 150, "render_frames": 240}], imgs)


def test_the_title_card_opens_the_tour_and_replaces_the_first_room_label(tmp_path):
    ov = _branded(tmp_path)
    assert [n for _, n, *_ in ov.labels] == ["Kitchen"]            # the title names the first room
    x0, y0, x1, y1 = _changed(ov.apply(FRAME, 75))
    assert abs((x0 + x1) / 2 - W / 2) < 60                          # centred
    assert _changed(ov.apply(FRAME, 75, captions=False)) is not None   # the MLS copy keeps the address


def test_the_end_card_and_corner_logo_are_in_the_branded_copy_only(tmp_path):
    ov = _branded(tmp_path)
    last = 389
    end = ov.apply(FRAME, last)
    assert end.mean() < FRAME.mean() - 30                           # the dark scrim
    assert (np.abs(end.astype(int) - [60, 60, 200]).sum(axis=2) < 30).any()   # the agent's photo (BGR red)
    assert (np.abs(end.astype(int) - [60, 200, 60]).sum(axis=2) < 30).any()   # the logo
    assert ov.apply(FRAME, last, captions=False) is FRAME           # unbranded: no card
    mid = ov.apply(FRAME, 200)                                      # the corner logo, top right
    x0, y0, x1, y1 = _changed(mid[:H // 4], FRAME[:H // 4])
    assert x0 > W * 0.7
    assert _changed(ov.apply(FRAME, 200, captions=False)[:H // 4], FRAME[:H // 4]) is None


def test_a_missing_branding_image_leaves_its_place_empty(tmp_path):
    ov = _branded(tmp_path)
    import os
    from tour_worker.io.images import branding_path
    os.remove(branding_path(str(tmp_path), "s3://b/agent.jpg"))
    end = ov.apply(FRAME, 389)
    assert not (np.abs(end.astype(int) - [60, 60, 200]).sum(axis=2) < 30).any()


def test_no_branding_no_cards(tmp_path):
    ov = _branded(tmp_path, agent=False, logo=False, corner=False, title=False)
    assert ov.title is None and ov.end_card is None and ov.corner is None


def test_the_contract_checks_branding():
    with pytest.raises(ValueError, match="branding"):
        parse_plan(_plan({"room_labels": {}, "branding": {"address": "x" * 200}}))
    with pytest.raises(ValueError, match="branding image"):
        parse_plan(_plan({"room_labels": {}, "branding": {"logo_uri": "ftp://x/logo.png"}}))
    ok = parse_plan(_plan({"room_labels": {}, "branding": {"address": "1 Main St", "corner_logo": True}}))
    assert ok.output.branding.address == "1 Main St"
