"""Transitions chosen by content (client review 2026-11-05: "hard cuts on the beat make an excellent
default"; motion transitions carry the viewer from one part of the home to the next):
  - same room, or the same part of the home: a hard cut, on the beat;
  - into a new part of the home (the narrative stage changes: outside -> entry, living -> private,
    private -> amenities): a whip when the incoming camera moves sideways, a push (zoom through)
    when it moves into depth - TRANSITION_S long, centred on the cut, so the cut still lands on the
    beat (tour_worker/render/timeline.py);
  - daylight -> twilight (or back): a fade, FADE_S.
Every transition stays at most half of the shorter shot either side."""
from __future__ import annotations

import cv2
import numpy as np

LATERAL_MOVES = frozenset({"reveal_pan", "reveal_slider", "reveal_gimbal_track", "push_slider", "slide",
                           "push_slide"})
TRANSITION_S, FADE_S = 0.33, 0.8


def is_dusk(img: np.ndarray) -> bool:
    """A twilight exterior: the upper third (the sky) dark and the whole frame dim."""
    small = cv2.resize(img, (160, int(160 * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
    v = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[..., 2].astype(np.float32) / 255.0
    return float(v[: v.shape[0] // 3].mean()) < 0.38 and float(v.mean()) < 0.40


def choose_transitions(shots, dusk_photos: frozenset[str] = frozenset()) -> None:
    """In place: each shot's transition_in (the first stays a cut)."""
    for prev, shot in zip(shots, shots[1:]):
        limit = 0.5 * min(prev.duration_s, shot.duration_s)
        kind, dur = "cut", 0.0
        if (prev.asset_id in dusk_photos) != (shot.asset_id in dusk_photos):
            kind, dur = "fade", FADE_S
        elif shot.room_key != prev.room_key and shot.narrative_stage != prev.narrative_stage:
            move = getattr(shot.camera, "move", None) or getattr(shot.camera, "preset", None) or ""
            kind, dur = ("whip" if move in LATERAL_MOVES else "push"), TRANSITION_S
        shot.transition_in.type = kind
        shot.transition_in.duration_s = round(min(dur, limit), 3)
