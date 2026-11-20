"""Transitions chosen by content (client review 2026-11-20, item 8: no full-frame smear effects -
movement between parts of the home is carried by the camera itself, cut on the beat):
  - every shot change: a hard cut, on the beat;
  - daylight -> twilight (or back): a fade, FADE_S - the only transition kept.
A fade stays at most half of the shorter shot either side."""
from __future__ import annotations

import cv2
import numpy as np

FADE_S = 0.8


def is_dusk(img: np.ndarray) -> bool:
    """A twilight exterior: the upper third (the sky) dark and the whole frame dim."""
    small = cv2.resize(img, (160, int(160 * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
    v = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[..., 2].astype(np.float32) / 255.0
    return float(v[: v.shape[0] // 3].mean()) < 0.38 and float(v.mean()) < 0.40


def choose_transitions(shots, dusk_photos: frozenset[str] = frozenset()) -> None:
    """In place: each shot's transition_in (the first stays a cut)."""
    for prev, shot in zip(shots, shots[1:]):
        kind, dur = "cut", 0.0
        if (prev.asset_id in dusk_photos) != (shot.asset_id in dusk_photos):
            kind, dur = "fade", FADE_S
        shot.transition_in.type = kind
        shot.transition_in.duration_s = round(min(dur, 0.5 * min(prev.duration_s, shot.duration_s)), 3)
