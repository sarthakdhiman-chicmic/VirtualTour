"""Beat sync v2 (planner/music_sync_v2.py): cuts on the music's hero moments, the strongest of a
run gets the cut and the rest a punch, the music may start mid-track, and anything v2 cannot do
falls back to the v3 sync."""
import subprocess
import sys

import numpy as np
import pytest

from tour_worker.planner import music_sync, music_sync_v2
from tour_worker.planner.config import PlannerConfig
from tour_worker.planner.music_sync import apply_sync, sync_to_music

CFG = PlannerConfig()
BEAT = 0.5
TOL = (1 + CFG.music_cut_lead_frames) / CFG.fps


def analysis(heroes, duration=90.0, phrases_every=8.0):
    """120 bpm, a candidate on every beat (value 0), bars every 4 beats, phrases every 8 s."""
    times = np.arange(0.0, duration - 0.05, BEAT)
    cands = [{"time": round(float(t), 4), "bar_start": i % 4 == 0, "kind": "beat", "value": 0.0,
              "change": 0.0, "hit": 1.0, "what_happens": "beat"} for i, t in enumerate(times)]
    return {"analysis_version": "4", "duration": duration, "music_start": 0.0, "music_end": duration - 0.5,
            "rhythm_end": duration - 0.5, "tempo_bpm": 120.0, "has_beat": True, "beat_period_s": BEAT,
            "beats": [{"time": c["time"], "bar_start": c["bar_start"]} for c in cands], "candidates": cands,
            "whooshes": [], "sections": [], "phrases": [round(float(p), 4) for p in np.arange(0.0, duration, phrases_every)],
            "heroes": heroes}


def track(heroes, **kw):
    return {"track_id": "t", "s3_uri": "s3://m/t.mp3", "sha256": "ab" * 32, "analysis": analysis(heroes, **kw)}


def hero(t, tier=2, kind="whoosh", strength=0.7):
    return {"t": t, "tier": tier, "kind": kind, "strength": strength}


def cuts_of(r):
    return np.cumsum(r.durations)[:-1]


TARGETS = [3.3] * 8


def test_without_heroes_v3_syncs_exactly_as_before():
    an = analysis([])
    del an["heroes"]
    tr = {"track_id": "t", "s3_uri": "s3://m/t.mp3", "sha256": "ab" * 32, "analysis": an}
    r = sync_to_music(TARGETS, tr, CFG)
    v3 = music_sync._sync_v3(TARGETS, tr, CFG)
    assert r.durations == v3.durations and r.music.sync_version is None
    assert "sync_version" not in r.music.model_dump(mode="json")


def test_a_drop_gets_its_cut():
    r = sync_to_music(TARGETS, track([hero(10.0, 1, "drop", 0.9)], phrases_every=100.0), CFG)
    assert r.music.sync_version == 2
    start = r.music.start_s
    assert np.min(np.abs(cuts_of(r) - (10.0 - start))) <= TOL
    assert any(h.kind == "cut" and h.tier == 1 for h in r.music.hooks)


PUNCHY = PlannerConfig(music_punches=True)        # punches are off by default (user 2026-10-23)


def test_a_run_of_heroes_cuts_the_strongest_and_punches_the_rest():
    run = [hero(12.0, 2, "whoosh", 0.6), hero(12.5, 1, "drop", 0.9), hero(13.5, 2, "big_hit", 0.7)]
    r = sync_to_music(TARGETS, track(run, phrases_every=100.0), PUNCHY)
    s = r.music.start_s
    rows = {round(h.t + s, 2): h for h in r.music.hooks}
    assert rows[12.5].kind == "cut"                                 # the drop, not its neighbours
    assert {rows[12.0].kind, rows[13.5].kind} <= {"accent", "missed"}
    assert sum(h.kind == "accent" for h in r.music.hooks) >= 1      # "covered" no longer means nothing
    assert all(len(a) == 3 and 0 < a[2] <= 0.06 for a in r.accents)


def test_a_hero_beats_a_plain_beat_in_reach():
    # 7.0 s: the planned cuts sit near 3.3 / 6.6 s; the hero is a beat further and wins
    r = sync_to_music(TARGETS, track([hero(7.0, 2, "whoosh", 0.6)], phrases_every=100.0), CFG)
    assert np.min(np.abs(cuts_of(r) - (7.0 - r.music.start_s))) <= TOL


def test_the_tour_opens_on_a_change_not_just_before_one():
    # a dropout at 20 s (a phrase starts at 18 s): open ON it, never on the phrase 2 s before
    an = analysis([hero(20.0, 2, "dropout", 0.8)], phrases_every=6.0)
    an["phrases"].append(18.0)
    r = sync_to_music(TARGETS, {"track_id": "t", "s3_uri": "s3://m/t.mp3", "sha256": "ab" * 32, "analysis": an}, CFG)
    assert r.music.start_s != 18.0
    from tour_worker.planner.music_sync_v2 import opening_score
    assert opening_score(an, 20.0, CFG) > 0 > opening_score(an, 18.0, CFG)


def _with_prints(an, same=True):
    """Bar fingerprints: all bars alike (any join is seamless) or all different."""
    bars = [c["time"] for c in an["candidates"] if c["bar_start"]]
    rng = np.random.default_rng(0)
    v = np.ones(26) / np.sqrt(26)
    an["bar_prints"] = [{"t": t, "v": list(v if same else (lambda x: x / np.linalg.norm(x))(rng.normal(size=26)))}
                        for t in bars]
    return an


def test_music_is_trimmed_so_a_drop_lands_on_a_cut():
    # a drop at 26.5 s is out of reach of 8 shots of 3.3 s started anywhere sensible - unless bars
    # come out of the music before it; with seamless joins the planner trims, and the drop is cut
    an = _with_prints(analysis([hero(26.5, 1, "drop", 1.0)], phrases_every=100.0))
    r = sync_to_music(TARGETS, {"track_id": "t", "s3_uri": "s3://m/t.mp3", "sha256": "ab" * 32, "analysis": an}, CFG)
    assert any(h.kind == "cut" and h.tier == 1 for h in r.music.hooks)
    m = r.music.model_dump(mode="json")
    assert m["segments"] and m["crossfade_s"] == CFG.music_v2_crossfade_s
    for e in m.get("edits") or []:
        assert e["cost"] <= CFG.music_v2_splice_max


def test_no_trim_where_the_join_would_be_heard():
    an = _with_prints(analysis([hero(26.5, 1, "drop", 1.0)], phrases_every=100.0), same=False)
    r = sync_to_music(TARGETS, {"track_id": "t", "s3_uri": "s3://m/t.mp3", "sha256": "ab" * 32, "analysis": an}, CFG)
    assert not r.music.edits and len(r.music.segments) == 1


def test_music_segments_cover_the_tour():
    an = _with_prints(analysis([hero(26.5, 1, "drop", 1.0)], phrases_every=100.0))
    r = sync_to_music(TARGETS, {"track_id": "t", "s3_uri": "s3://m/t.mp3", "sha256": "ab" * 32, "analysis": an}, CFG)
    assert sum(s.to_s - s.from_s for s in r.music.segments) >= r.music.end_s


def test_a_hero_may_stretch_a_shot_to_the_wide_bounds():
    # 8.0 s needs two 4.0 s shots: past the role bounds (1.15 x 3.4 = 3.91 s), inside the wide
    # ones (1.2 x 3.4 = 4.08 s)
    targets = [3.4] * 8
    r = sync_to_music(targets, track([hero(8.0, 2, "whoosh", 0.7)], phrases_every=100.0), CFG)
    assert np.min(np.abs(cuts_of(r) - (8.0 - r.music.start_s))) <= TOL


def test_the_music_may_start_late_so_the_first_drop_opens_an_early_shot():
    r = sync_to_music(TARGETS, track([hero(34.0, 1, "drop", 1.0)], duration=90.0, phrases_every=4.0), CFG)
    assert r.music.start_s > 0
    k = int(np.argmin(np.abs(cuts_of(r) - (34.0 - r.music.start_s))))
    assert abs(cuts_of(r)[k] - (34.0 - r.music.start_s)) <= TOL and k <= 2   # opens shot 2-4


def test_punch_sizes_follow_the_hero_and_reach_apply_sync():
    run = [hero(12.0, 2, "whoosh", 0.6), hero(12.5, 1, "drop", 0.9), hero(13.5, 2, "big_hit", 0.7)]
    r = sync_to_music(TARGETS, track(run, phrases_every=100.0), PUNCHY)

    class Cam:
        time_warp = punches = None

    class Shot:
        def __init__(self, d):
            self.duration_s, self.camera = d, Cam()

    shots = [Shot(d) for d in TARGETS]
    apply_sync(shots, r, PUNCHY)
    amounts = sorted(p.amount for s in shots for p in s.camera.punches or [])
    assert amounts and all(0 < a <= 0.06 for a in amounts)


def test_by_default_heroes_without_a_cut_play_uncut():
    run = [hero(12.0, 2, "whoosh", 0.6), hero(12.5, 1, "drop", 0.9), hero(13.5, 2, "big_hit", 0.7)]
    r = sync_to_music(TARGETS, track(run, phrases_every=100.0), CFG)
    kinds = {round(h.t + r.music.start_s, 2): h.kind for h in r.music.hooks}
    assert kinds[12.5] == "cut" and not r.accents
    assert all(k in ("cut", "uncut") for k in kinds.values())


@pytest.mark.parametrize("broken", ["raise", "none"])
def test_v2_trouble_falls_back_to_v3(monkeypatch, broken):
    def bad(*a, **k):
        if broken == "raise":
            raise RuntimeError("boom")
        return None

    monkeypatch.setattr(music_sync_v2, "plan_v2", bad)
    tr = track([hero(10.0, 1, "drop", 0.9)])
    r = sync_to_music(TARGETS, tr, CFG)
    assert r.music.sync_version is None
    assert r.extra_warnings == [f"music_sync_v2_fallback:{'error_RuntimeError' if broken == 'raise' else 'no_solution'}"]


def test_the_planner_never_imports_librosa():
    code = ("import sys; import tour_worker.planner.planner, tour_worker.planner.music_sync_v2;"
            "print('librosa' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
