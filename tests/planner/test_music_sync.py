"""Beat sync (plan 1.5): planner/music_sync.py and its place in build_shot_plan."""

import numpy as np
import pytest
from test_plan_integration import _assets, report  # noqa: F401  (fixture; tests dir is on sys.path)

from tour_worker.planner import planner as planner_mod
from tour_worker.planner.config import PlannerConfig
from tour_worker.planner.music_sync import (
    apply_sync,
    frame_align,
    length_bounds,
    ramp_warp,
    sync_to_music,
    time_warp,
)
from tour_worker.planner.planner import _compute_plan_hash, build_shot_plan

CFG = PlannerConfig()
# cuts sit music_cut_lead_frames before their beat (the tour end does not)
TOL = (1 + CFG.music_cut_lead_frames) / CFG.fps


def analysis(duration=60.0, bpm=120.0, offset=0.25, has_beat=True, version="1"):
    """A steady track: a candidate on every beat, every 4th a bar start, every 8th bar louder."""
    beat = 60.0 / bpm
    times = np.arange(offset, duration - 0.05, beat)
    cands = [{"time": round(float(t), 4), "bar_start": i % 4 == 0, "kind": "beat",
              "value": 0.5 if i % 32 == 0 else 0.0, "change": 0.0, "what_happens": "beat"}
             for i, t in enumerate(times)]
    return {"analysis_version": version, "duration": duration, "music_start": 0.0,
            "music_end": duration - 0.5, "tempo_bpm": bpm, "has_beat": has_beat,
            "beat_period_s": beat, "beats": [], "candidates": cands if has_beat else []}


def track(**kw):
    return {"track_id": "trk1", "s3_uri": "s3://music/trk1.mp3", "sha256": "ab" * 32,
            "analysis": analysis(**kw)}


TARGETS = [5.0, 4.2, 6.0, 5.5, 4.0, 6.0]


def boundaries(durations):
    return np.cumsum(durations)


def test_every_cut_is_on_a_beat_within_the_rules():
    tr = track()
    r = sync_to_music(TARGETS, tr, CFG)
    assert r.music.synced and r.warning is None and r.music.loops == 1
    cand_t = np.array([c["time"] for c in tr["analysis"]["candidates"]])
    for b in boundaries(r.durations):           # on a beat, to within one video frame
        assert np.abs(cand_t - b).min() <= TOL
    lo_b, hi_b = length_bounds(TARGETS, None, CFG)              # per-shot bounds (role: detail)
    for k, d in enumerate(r.durations):
        lo = max(lo_b[k], CFG.final_shot_min_s) if k == len(r.durations) - 1 else lo_b[k]
        assert lo - 0.02 - TOL <= d <= hi_b[k] + 0.02 + TOL     # +- the cut lead
    for b, p in zip(boundaries(r.durations), boundaries(TARGETS), strict=True):
        assert abs(b - p) <= CFG.music_max_drift + TOL
    bar_t = cand_t[[c["bar_start"] for c in tr["analysis"]["candidates"]]]
    assert np.abs(bar_t - boundaries(r.durations)[-1]).min() <= 1.0 / CFG.fps   # ends on a bar
    assert r.music.end_s == pytest.approx(sum(r.durations)) and r.music.cuts_on_bar >= 1


def test_cuts_land_on_exact_frames_of_the_worker_timeline():
    r = sync_to_music(TARGETS, track(offset=0.2371), CFG)
    cum = np.cumsum(r.durations)
    frames = np.round(cum * CFG.fps)
    assert np.allclose(cum * CFG.fps, frames, atol=0.16)     # centiseconds land on a frame


def test_frame_align_round_trips_random_boundaries():
    rng = np.random.default_rng(0)
    for _ in range(200):
        b = np.cumsum(rng.uniform(3.5, 6.0, 15))
        d = frame_align(b, 30)
        assert list(np.round(np.cumsum(d) * 30).astype(int)) == list(np.round(b * 30).astype(int))


def test_a_short_track_is_queued_after_itself_at_its_loop_point():
    # a 14 s file whose music ends at 11.5 s (fade / silence after): copies join on the last bar
    # start before 11.5 s, never playing the silent tail
    tr = track(duration=14.0)
    tr["analysis"]["music_end"] = 11.5
    tr["analysis"]["beats"] = [{"time": c["time"], "bar_start": c["bar_start"]}
                               for c in tr["analysis"]["candidates"]]
    r = sync_to_music(TARGETS, tr, CFG)
    loop = r.music.loop_s
    assert loop <= 11.5 and loop > 11.5 - 2.0 + 1e-6            # last bar (2 s bars at 120 BPM)
    assert r.music.synced and r.music.loops >= 3
    assert (r.music.loops - 1) * loop + 11.5 >= r.music.end_s     # the queued music covers the tour
    beats = np.array([c["time"] for c in tr["analysis"]["candidates"] if c["time"] < loop])
    for b in boundaries(r.durations)[:-1]:          # every cut on a beat of its copy's music
        local = b - (b // loop) * loop
        assert np.abs(beats - local).min() <= TOL


@pytest.mark.parametrize("kw, reason", [({"has_beat": False}, "no_beat"),
                                        ({"version": "99"}, "analysis_version")])
def test_fallback_keeps_the_planned_lengths(kw, reason):
    r = sync_to_music(TARGETS, track(**kw), CFG)
    assert not r.music.synced and r.music.fallback_reason == reason
    assert r.warning == f"music_not_synced:{reason}"
    assert r.durations == pytest.approx(TARGETS, abs=0.02)


def test_planner_plans_cameras_at_normal_length_then_retimes(report, monkeypatch):  # noqa: F811
    seen = {}
    real = planner_mod.plan_cameras

    def spy(inputs, frames, *a, **k):
        seen["durations"] = [i.duration_s for i in inputs]
        return real(inputs, frames, *a, **k)

    monkeypatch.setattr(planner_mod, "plan_cameras", spy)
    planned = [s.duration_s for s in build_shot_plan(report, "t", _assets(), cfg=CFG).shots]
    tr = track(duration=90.0, bpm=100.0)
    plan = build_shot_plan(report, "t", _assets(), cfg=CFG, music_track=tr)
    assert plan.plan_version == "1.5" and plan.music is not None and plan.music.synced
    assert seen["durations"] == planned                      # cameras: the NORMAL lengths
    durs = [s.duration_s for s in plan.shots]
    assert plan.total_duration_s == pytest.approx(sum(durs)) == pytest.approx(plan.music.end_s)
    for s, p in zip(plan.shots, planned, strict=True):
        w = s.camera.time_warp
        if abs(s.duration_s - p) < 1 / CFG.fps:
            assert w is None
            continue
        assert w[0] == [0.0, 0.0] and w[-1] == [1.0, 1.0]
        assert all(b[0] > a[0] and b[1] > a[1] for a, b in zip(w, w[1:], strict=False))
        # the first part plays at normal pace: move time / output time = synced / planned
        assert w[1][1] / w[1][0] == pytest.approx(s.duration_s / p, rel=1e-3)
    d = plan.model_dump(mode="json")
    assert d["provenance"]["plan_hash"] == _compute_plan_hash(d)


def test_time_warp_lands_the_move_on_the_cut_and_ramps_the_rear():
    # hook at 2.7 s, shot planned for 3.0 s: normal pace first, the rear faster, ending on the cut
    w = time_warp(3.0, 2.7, CFG)
    assert w[0] == [0.0, 0.0] and w[-1] == [1.0, 1.0]
    first = w[2][1] / w[2][0]                               # move per output, first part
    rear = (1 - w[2][1]) / (1 - w[2][0])                    # ... the rear
    assert first == pytest.approx(0.9, rel=1e-3) and rear > first
    assert rear / first <= 1.6                               # within the ramp limit
    # tightest detail bound (75%): still within the limit
    w = time_warp(4.0, 3.0, CFG)
    assert ((1 - w[2][1]) / (1 - w[2][0])) / (w[2][1] / w[2][0]) <= 1.6
    assert time_warp(4.0, 4.01, CFG) is None                # within a frame: unchanged


def test_hooks_get_cuts_and_heroes_stay_long():
    """A drop (big musical change) 0.4 s before a planned cut gets the cut; hero shots keep
    85-125% of their length, details 75-125%."""
    tr = track(duration=60.0, version="2")
    an = tr["analysis"]
    drop = 9.0 - 0.4 + 0.25                                  # off the planned 9.2 s boundary
    an["candidates"].append({"time": drop, "bar_start": False, "kind": "beat", "value": 0.3,
                             "change": 0.9, "what_happens": "drop"})
    an["candidates"].sort(key=lambda c: c["time"])
    roles = ["hero", "detail", "hero", "detail", "detail", "hero"]
    r = sync_to_music(TARGETS, tr, CFG, roles)
    cuts = boundaries(r.durations)[:-1]
    assert np.abs(cuts - drop).min() <= TOL          # the cut is on the drop
    assert r.hooks_hit >= 1
    lo, hi = length_bounds(TARGETS, roles, CFG)
    for k, d in enumerate(r.durations[:-1]):
        assert lo[k] - 0.02 <= d <= hi[k] + 0.02
        if roles[k] == "hero":
            assert d >= 0.85 * TARGETS[k] - 0.02


def test_no_music_leaves_the_plan_as_before(report):  # noqa: F811
    plan = build_shot_plan(report, "t", _assets(), cfg=CFG)
    assert plan.music is None and plan.plan_version == "1.3"


def test_cuts_land_on_whooshes_rather_than_the_next_bar():
    """v2 analysis: whoosh candidates (transition effects, often just before a bar) win over
    the bar line ~0.5-1 s later. Review: "the transitions should be at that whoosh"."""
    tr = track(version="2")
    an = tr["analysis"]
    whooshes = [round(b - 0.55, 4) for b in np.arange(2.25, 40.0, 2.0)]   # each bar, 0.55 s early
    an["candidates"] += [{"time": w, "bar_start": False, "kind": "whoosh", "value": 1.0,
                          "change": 1.0, "what_happens": "whoosh"} for w in whooshes]
    an["candidates"].sort(key=lambda c: c["time"])
    r = sync_to_music(TARGETS, tr, CFG)
    cuts = boundaries(r.durations)[:-1]
    on_whoosh = [np.abs(np.array(whooshes) - c).min() <= TOL for c in cuts]
    assert r.music.synced and sum(on_whoosh) >= len(cuts) - 1


def test_queued_copies_join_where_the_beat_stops_not_in_the_tail():
    tr = track(duration=14.0, version="2")
    an = tr["analysis"]
    an["music_end"] = 13.6          # a held note / reverb rings on to 13.6 s ...
    an["rhythm_end"] = 10.25        # ... but the beat stops at 10.25 s
    an["beats"] = [{"time": c["time"], "bar_start": c["bar_start"]} for c in an["candidates"]]
    r = sync_to_music(TARGETS, tr, CFG)
    assert r.music.loop_s == pytest.approx(10.25, abs=0.01)     # the bar at the stop
    assert r.music.synced


def with_hooks(tr, hooks):
    """Add hook candidates (big musical changes) at the given times."""
    an = tr["analysis"]
    an["analysis_version"] = "2"
    an["candidates"] = [c for c in an["candidates"]
                        if all(abs(c["time"] - h) > 1e-3 for h in hooks)]
    an["candidates"] += [{"time": h, "bar_start": False, "kind": "beat", "value": 0.3,
                          "change": 0.9, "what_happens": "drop"} for h in hooks]
    an["candidates"].sort(key=lambda c: c["time"])
    return tr


def test_every_hook_moment_gets_a_cut_and_close_pairs_become_cut_plus_accent():
    """Hooks first (review 2026-10-19: 3 of 7 hooks hit). A riser/drop pair 2.4 s apart is one
    moment: one cut, the other hook an accent. All shots heroes: the role bounds alone cannot
    reach every hook, the wide bounds can."""
    hooks = [5.85, 15.35, 17.75, 25.25]
    tr = with_hooks(track(duration=60.0), hooks)
    roles = ["hero"] * len(TARGETS)
    r = sync_to_music(TARGETS, tr, CFG, roles)
    assert r.music.synced and r.music.hooks_required and r.warning is None
    kinds = {h.t: h.kind for h in r.music.hooks}
    assert kinds[5.85] == kinds[25.25] == "cut"
    assert sorted([kinds[15.35], kinds[17.75]]) == ["accent", "cut"]
    acc = next(h for h in r.music.hooks if h.kind == "accent")
    assert acc.reason == "merged"
    cuts = boundaries(r.durations)[:-1]
    for h in r.music.hooks:
        if h.kind == "cut":
            assert np.abs(cuts - h.t).min() <= TOL
    assert len(r.accents) == 1 and len(r.hook_cuts) == 3


def test_cut_lead_puts_cuts_a_frame_early_but_not_the_end():
    d = frame_align([2.0, 4.0, 6.0], 30, lead_frames=1)
    b = np.cumsum(d)
    assert np.round(b * 30).tolist() == [59, 119, 180]


def test_ramp_warp_speeds_into_the_cut():
    w = ramp_warp(CFG)
    assert w[0] == [0.0, 0.0] and w[-1] == [1.0, 1.0]
    assert all(b[0] > a[0] and b[1] > a[1] for a, b in zip(w, w[1:], strict=False))
    start = w[1][1] / w[1][0]
    end = (w[-1][1] - w[-2][1]) / (w[-1][0] - w[-2][0])
    assert end / start > 1.3                                   # visibly faster at the cut
    assert time_warp(4.0, 4.0, CFG, ramp=True) == w           # ramps even at the planned length


def test_apply_sync_ramps_hook_cuts_and_punches_the_hit(report):  # noqa: F811
    hooks = [7.35, 15.85, 18.25, 26.75]
    plan = build_shot_plan(report, "t", _assets(), cfg=CFG)
    tr = with_hooks(track(duration=90.0), hooks)
    r = sync_to_music([s.duration_s for s in plan.shots], tr, CFG, [s.role for s in plan.shots])
    apply_sync(plan.shots, r, CFG)
    for k in r.hook_cuts:
        assert plan.shots[k].camera.time_warp == ramp_warp(CFG)
        if k + 1 < len(plan.shots):
            p = plan.shots[k + 1].camera.punches[0]
            assert p.at_s == 0.0 and p.amount == CFG.music_punch_cut
    for k, at in r.accents:
        assert any(abs(p.at_s - at) < 1e-9 and p.amount == CFG.music_punch_accent
                   for p in plan.shots[k].camera.punches)
    assert [s.duration_s for s in plan.shots] == r.durations


def test_cuts_follow_the_hardest_drum_hits_not_the_guessed_bar_line():
    """Analysis v3 "hit": the tracker's bar starts are on beat 1 but the hardest hit of every bar
    is beat 3 (sample4.mp3). Review 2026-10-19: cuts on the bar came half a bar after the hit."""
    tr = track(duration=60.0, version="3")
    for i, c in enumerate(tr["analysis"]["candidates"]):
        c["hit"] = 1.6 if i % 4 == 2 else 0.9
    r = sync_to_music(TARGETS, tr, CFG)
    strong = np.array([c["time"] for c in tr["analysis"]["candidates"] if c["hit"] > 1.5])
    cuts = boundaries(r.durations)[:-1]
    on = [np.abs(strong - c).min() <= TOL for c in cuts]
    assert r.music.synced and all(on)
