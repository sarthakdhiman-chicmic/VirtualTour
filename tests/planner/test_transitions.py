"""planner/transitions.py: transitions chosen by content (client review 2026-11-05)."""
import types

import numpy as np

from tour_worker.planner.transitions import FADE_S, choose_transitions, is_dusk


def _shot(asset, room, stage, move="push_in", d=4.0):
    return types.SimpleNamespace(asset_id=asset, room_key=room, narrative_stage=stage, duration_s=d,
                                 camera=types.SimpleNamespace(move=move, preset=None),
                                 transition_in=types.SimpleNamespace(type="cut", duration_s=0.0))


def test_every_change_of_shot_is_a_cut_even_into_a_new_part_of_the_home():
    shots = [_shot("a", "ext", "exterior_open"), _shot("b", "entry", "entry", move="reveal_slider"),
             _shot("c", "liv", "living", move="push_in"), _shot("d", "kit", "living"), _shot("e", "kit", "living")]
    choose_transitions(shots)
    assert [s.transition_in.type for s in shots] == ["cut"] * 5
    assert all(s.transition_in.duration_s == 0.0 for s in shots)


def test_daylight_to_twilight_fades_and_no_transition_outlasts_half_a_shot():
    shots = [_shot("a", "ext", "exterior_open"), _shot("b", "pool", "amenity_close", d=1.2)]
    choose_transitions(shots, frozenset({"b"}))
    assert shots[1].transition_in.type == "fade" and shots[1].transition_in.duration_s == min(FADE_S, 0.6)


def test_dusk_is_a_dark_sky_and_a_dim_frame():
    day = np.full((300, 400, 3), 190, np.uint8)
    dusk = np.full((300, 400, 3), 40, np.uint8)
    dusk[200:] = 120                                            # lit windows / garden lights below
    assert not is_dusk(day) and is_dusk(dusk)
