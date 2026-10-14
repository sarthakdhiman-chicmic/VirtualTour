"""Tour v6 planner (worker harness; backend port later): every shot is a v3-style rotate reveal.

Review of v5: the v3 tilt reveal through the "rotate" projection looked best; the v4 compound
effects were too much; and every shot should zoom OUT - a push-in without room context zooms at
nothing in particular, so each shot starts on part of the room and ends on all of it.

So every camera is the v3 keyframe_path (plan 1.3, projection "rotate", the v3 global time
profile and monotone cubic: the exact v3 look), built from the v3 move builders
(tools/tour_v2/planner.py) and varied only where it starts and which way it travels:

  tilt_up      the v3 tilt_reveal: far low corner -> across and up to eye level -> whole room
  tilt_down    the same mirrored vertically: far high corner -> across and down -> whole room
  pan          the same at one height: far side, mid height -> across -> whole room
  pull_back    the v3 pull_back_reveal: on the subject -> straight back (slight drift) -> whole room
  arc          (v7) the aim sweeps along a curve: one side -> over the top (or under) -> the other
               side -> whole room; a circular look-around from where the camera stands (a real
               orbit needs depth: near and far things moving differently)
  crane        (v7, crop, wide rooms and exteriors) a gentle rise from low to eye level with a
               slight pull-back: a sliding window reads as the camera rising, unlike a tilt

each mirrored left/right (d). A small beam search picks one per shot: fit for the room
(coverage = saliency spread; exterior, bathroom, final), how interesting the opening frame is,
pan direction alternating shot to shot, and no variant twice running. Rotate shots are pre-warped
by the worker (modules/prewarp.py), so renders are deterministic.

  python -m tools.tour_v6.planner --baseline fixtures/downing/baseline_plan.json \\
      --saliency fixtures/downing/saliency.json --images-dir <photos> --out-dir out/tour_v6
"""
import argparse
import copy
import json
import math
import os
import statistics

from modules.camera_path import KeyframePath, base_window, corner_rms_px, motion_px, vec_angle
from modules.plan_io import compute_plan_hash, parse_plan
from modules.timeline import compute_timeline
from tools.tour_v2 import planner as v2
from tools.tour_v3 import planner as P3
from tools.tour_v2.saliency import load_gray, spectral_residual

PLANNER_VERSION = "tour_v8-0.1"
OUT_W, OUT_H, FPS = 1920, 1080, 30
HFOV_DEG = 75.0
BEAM = 300

# seconds per category before scaling to the baseline total; clamped to SHOT_RANGE_S after
DURATION_S = {"exterior": 5.5, "wide": 7.0, "medium": 6.0, "tight": 5.0, "bath": 4.5, "final": 6.0}
SHOT_RANGE_S = (4.0, 8.0)
TOTAL_EXTRA_S = 5.0       # the tour may run up to 5 s over the baseline total (as v4/v5)
V_MAX = 3.3               # px/frame: about the v3 rotate pace (2.75 at 7.5 s); shorter shots
SHRINK_STEP = 0.05        # get a smaller reveal instead of a faster one

SUIT = {
    # v8: the exterior rises vertically as a crop (as v3 / v5 did; a rotate tilt leaned the house)
    "exterior": {"rise": 4.0, "tilt_up": 1.5, "pull_back": 1.0},
    "wide": {"tilt_up": 3.0, "crane": 2.75, "arc": 2.75, "pan": 2.5, "tilt_down": 2.0, "pull_back": 1.0},
    "medium": {"tilt_up": 2.5, "pan": 2.5, "crane": 2.25, "arc": 2.25, "tilt_down": 2.0, "pull_back": 1.5},
    "tight": {"pull_back": 3.0, "crane": 2.5, "pan": 2.0, "tilt_up": 2.0, "tilt_down": 1.5},
    "bath": {"pull_back": 3.0, "pan": 2.0, "tilt_up": 1.5},
    "final": {"pull_back": 2.5, "tilt_up": 2.5, "pan": 2.0, "tilt_down": 1.5},
}
HERO_BONUS = 0.5          # heroes lean toward the travelling reveals
TRAVELLING = {"tilt_up", "tilt_down", "pan", "arc", "crane", "rise"}
CROP_VARIANTS = {"crane", "rise"}                  # everything else turns the camera in place ("rotate")
ARC = {"z": (1.16, 1.14, 1.11, 1.07, 1.012), "r": (1.0, 0.95, 0.8, 0.5), "overlap": 0.18}
VARIANT_DURATION_S = {"arc": 8.0}         # a circle is a long path: give it time to stay slow
# crane: lateral travel and start zoom grow with the room's coverage (0 = at/below the tour's
# median spread, 1 = the widest photo). At 0 it is the v7 crane that worked in smaller rooms; at
# 1 it slides ~16% of the width while rising, so it reads as a move, not a stabiliser wobble
# (v7's 2.5% drift did in big rooms). Lateral room in a crop needs zoom: +-1/(2z) of the width.
CRANE = {"z0": (1.10, 1.28), "z1": (1.035, 1.06), "rise": 0.16, "drift": (0.025, 0.11)}
# rise (exterior): low -> as high as the photo allows, zooming out to near-full (crop)
RISE_V8 = {"z0": 1.10, "z1": 1.03, "x_toward_peak": 0.3}


def arc_camera(g, d):
    """The arc as a compound camera (addendum 1.3.2): a keyframe path's per-channel curves turned
    8-11 deg/frame on a circle; overlapping phases left -> top -> right -> bottom -> centre blend
    into a round spiral, and compound's distance-based, turn-aware timing keeps the speed even
    round it (3.1 deg/frame, like the tilt reveals). The aim circles once (direction d) while the
    circle shrinks into the centre and the camera pulls back to the whole room; each point's
    radius is the rotate camera's reach at that zoom times the spiral's shrink."""
    z, r, ov = ARC["z"], ARC["r"], ARC["overlap"]
    pts = []
    for k, ang in enumerate((180, 90, 0, -90)):
        rx = g.fit(1.0, 0.5, z[k])[0] - 0.5
        ry = g.fit(0.5, 1.0, z[k])[1] - 0.5
        a = math.radians(ang)
        ox = math.cos(a) if k < 3 else 0.0
        pts.append(P3._state(*g.fit(0.5 + d * r[k] * rx * ox, 0.5 - r[k] * ry * math.sin(a), z[k]), z[k]))
    end = P3._state(*g.fit(0.5, 0.5, z[-1]), z[-1])
    win = [(0, 0.25 + ov), (0.25 - ov, 0.5 + ov), (0.5 - ov, 0.75 + ov), (0.75 - ov, 1)]
    segs = [P3._seg(round(max(0.0, a), 3), round(min(1.0, b), 3), p) for (a, b), p in zip(win, pts[1:] + [end])]
    return pts[0], segs


COMPOUND_VARIANTS = {"arc"}


def geometry(s, variant):
    proj = "crop" if variant in CROP_VARIANTS else "rotate"
    return v2.Geometry(s["source"]["width"], s["source"]["height"], OUT_W, OUT_H, proj, HFOV_DEG)


def _flip_y(kfs):
    return [dict(k, cy=round(1 - k["cy"], 6)) for k in kfs]


def variant_keyframes(variant, g, sal, d):
    if variant == "tilt_up":
        return v2.move_tilt_reveal(g, sal, d)
    if variant == "tilt_down":
        # mirror the saliency target too, so K1 sits at the mirrored eye level
        s2 = dict(sal, peak={"x": sal["peak"]["x"], "y": 1 - sal["peak"]["y"]})
        return _flip_y(v2.move_tilt_reveal(g, s2, d))
    if variant == "pan":
        kfs = v2.move_tilt_reveal(g, sal, d)
        kfs[0] = dict(kfs[0], cy=kfs[1]["cy"])          # start at the height it travels at
        return kfs
    if variant == "pull_back":
        kfs = v2.move_pull_back_reveal(g, sal, d)
        # a slight drift in direction d: start a little to the other side of the subject
        cx, cy = g.fit(kfs[0]["cx"] - d * 0.03, kfs[0]["cy"], kfs[0]["zoom"])
        kfs[0] = dict(kfs[0], cx=round(cx, 6), cy=round(cy, 6))
        return kfs
    if variant == "crane":
        c, f = CRANE, sal.get("coverage", 0.0)
        z0 = c["z0"][0] + f * (c["z0"][1] - c["z0"][0])
        drift = c["drift"][0] + f * (c["drift"][1] - c["drift"][0])
        cx0, cy0 = g.fit(0.5 - d * drift, 0.5 + c["rise"] / 2 + 0.04, z0)
        z1 = c["z1"][0] + f * (c["z1"][1] - c["z1"][0])
        cx1, cy1 = g.fit(0.5 + d * drift, 0.5 - c["rise"] / 2 + 0.04, z1)
        return [v2.kf(0, cx0, cy0, z0), v2.kf(1, cx1, cy1, z1)]
    if variant == "rise":
        r = RISE_V8
        cx = 0.5 + r["x_toward_peak"] * (sal["peak"]["x"] - 0.5)
        cx0, cy0 = g.fit(cx, 1.0, r["z0"])           # as low as the photo allows
        cx1, cy1 = g.fit(cx, 0.0, r["z1"])           # as high as it allows, near-full view
        return [v2.kf(0, cx0, cy0, r["z0"]), v2.kf(1, cx1, cy1, r["z1"])]
    raise ValueError(variant)


class SalMap:
    def __init__(self, path):
        self.m = spectral_residual(load_gray(path))
        self.mean = float(self.m.mean()) or 1e-9

    def window_interest(self, g, k):
        h, w = self.m.shape
        ww, wh = g.width(k["zoom"]), g.height(k["zoom"])
        x0, x1 = int((k["cx"] - ww / 2) * w), int(math.ceil((k["cx"] + ww / 2) * w))
        y0, y1 = int((k["cy"] - wh / 2) * h), int(math.ceil((k["cy"] + wh / 2) * h))
        return float(self.m[max(0, y0):y1, max(0, x0):x1].mean()) / self.mean


def category(shots, i, spread, thr, q25):
    s = shots[i]
    if i == len(shots) - 1:
        return "final"
    if v2.is_exterior(s):
        return "exterior"
    if v2.is_bathroom(s):
        return "bath"
    return "wide" if spread > thr else "medium" if spread > q25 else "tight"


def lateral_travel(kfs):
    """Which way the shot travels across the room (the opening phase), +1 right, -1 left."""
    dx = kfs[1]["cx"] - kfs[0]["cx"] if len(kfs) > 2 else kfs[-1]["cx"] - kfs[0]["cx"]
    return 1 if dx > 1e-4 else -1 if dx < -1e-4 else 0


def select(cands):
    beams = [(0.0, [], {})]
    for i, cs in enumerate(cands):
        nxt = []
        for score, seq, uses in beams:
            for ci, c in enumerate(cs):
                s = score + c["suit"] + c["interest"]
                if seq:
                    prev = cands[i - 1][seq[-1]]
                    if c["travel"] and c["travel"] == prev["travel"]:
                        s -= 2.0    # travel the opposite way to the last room
                    if c["variant"] == prev["variant"]:
                        s -= 2.5
                    if len(seq) > 1 and cands[i - 2][seq[-2]]["variant"] == c["variant"]:
                        s -= 1.0
                s -= 1.0 * uses.get((c["variant"], c["d"]), 0)
                s -= 1.5 * max(0, uses.get(c["variant"], 0) - 2)    # a fourth use of a variant costs more
                u = dict(uses)
                u[(c["variant"], c["d"])] = u.get((c["variant"], c["d"]), 0) + 1
                u[c["variant"]] = u.get(c["variant"], 0) + 1
                nxt.append((s, seq + [ci], u))
        nxt.sort(key=lambda b: (-b[0], b[1]))
        beams = nxt[:BEAM]
    best = beams[0]
    return [cands[i][ci] for i, ci in enumerate(best[1])], best[0]


def allocate(cats, total, variants):
    want = [VARIANT_DURATION_S.get(v, DURATION_S[c]) for c, v in zip(cats, variants)]
    total = min(sum(want), total + TOTAL_EXTRA_S)
    d = list(want)
    for _ in range(30):
        f = total / sum(d)
        d = [min(max(x * f, SHOT_RANGE_S[0]), SHOT_RANGE_S[1]) for x in d]
        if abs(sum(d) - total) < 1e-6:
            break
    cs = [math.floor(x * 100) for x in d]
    for i in sorted(range(len(d)), key=lambda i: -(d[i] * 100 - cs[i]))[:round(total * 100) - sum(cs)]:
        cs[i] += 1
    return [c / 100 for c in cs]


def motion_stats(kfs, n, g):
    fr = KeyframePath(kfs).frames(n)
    sp = [corner_rms_px(a, b, g.src_w, g.src_h, OUT_W, OUT_H) for a, b in zip(fr, fr[1:])]
    v = [motion_px(a, b, g.src_w, g.src_h, OUT_W, OUT_H) for a, b in zip(fr, fr[1:])]
    turn = [math.degrees(vec_angle(a, b)) for a, b in zip(v, v[1:])]
    return {"speed_first_px_f": round(sp[0], 3), "speed_max_px_f": round(max(sp), 3),
            "speed_min_px_f": round(min(sp), 3), "speed_cut_px_f": round(sp[-1], 3),
            "turn_max_deg_f": round(max(turn), 3)}


def cap_speed(kfs, n, g, note):
    """Shrink the reveal toward its end (the whole room) until no frame moves faster than V_MAX."""
    end = kfs[-1]
    f = 1.0
    out = kfs
    while motion_stats(out, n, g)["speed_max_px_f"] > V_MAX and f > SHRINK_STEP:
        f = round(f - SHRINK_STEP, 3)
        out = [dict(k, cx=round(end["cx"] + f * (k["cx"] - end["cx"]), 6),
                    cy=round(end["cy"] + f * (k["cy"] - end["cy"]), 6),
                    zoom=round(end["zoom"] * (k["zoom"] / end["zoom"]) ** f, 6)) for k in kfs]
        out = v2.enforce(out, n, g, dict(note))
    return out, f


def build(baseline, saliency, images_dir):
    base = copy.deepcopy(baseline.get("plan", baseline))
    plan = copy.deepcopy(base)
    plan["plan_version"] = "1.3"
    shots = plan["shots"]
    sal = saliency["photos"]
    names = [s["image_uri"].split("/")[-1] for s in shots]
    maps = {n: SalMap(os.path.join(images_dir, n)) for n in set(names)}
    spreads = sorted(sal[n]["spread"] for n in names)
    thr, q25 = statistics.median(spreads), statistics.quantiles(spreads, n=4)[0]
    cats = [category(shots, i, sal[n]["spread"], thr, q25) for i, n in enumerate(names)]
    cands = []
    for i, (s, n, cat) in enumerate(zip(shots, names, cats)):
        cs = []
        for variant, suit in SUIT[cat].items():
            g = geometry(s, variant)
            if s["role"] == "hero" and variant in TRAVELLING:
                suit += HERO_BONUS
            for d in (1, -1):
                if variant in COMPOUND_VARIANTS:
                    start, segs = arc_camera(g, d)
                    kfs, opening, travel = None, start, d
                else:
                    cov = max(0.0, min(1.0, (sal[n]["spread"] - thr) / max(1e-9, spreads[-1] - thr)))
                    kfs = variant_keyframes(variant, g, dict(sal[n], coverage=cov), d)
                    opening, travel = kfs[0], lateral_travel(kfs)
                interest = 1.5 * max(-1, min(1, maps[n].window_interest(g, opening) - 1))
                cs.append({"variant": variant, "d": d, "keyframes": kfs, "suit": suit,
                           "interest": round(interest, 4), "travel": travel, "category": cat})
        cands.append(cs)
    chosen, score = select(cands)
    durations = allocate(cats, base["total_duration_s"], [c["variant"] for c in chosen])
    for s, d in zip(shots, durations):
        s["duration_s"] = d
        s["transition_in"] = {"type": "cut", "duration_s": 0.0}
    plan["total_duration_s"] = round(sum(durations), 2)
    timeline = compute_timeline(v2.parse_plan_loose(plan))
    notes = []
    for i, (s, c) in enumerate(zip(shots, chosen)):
        g = geometry(s, c["variant"])
        n = timeline[i]["render_frames"]
        note = {"shot_id": s["shot_id"], "room_type": s["room_type"], "role": s["role"],
                "category": c["category"], "variant": c["variant"], "d": c["d"], "travel": c["travel"],
                "suit": c["suit"], "interest": c["interest"], "duration_s": s["duration_s"],
                "frames": [timeline[i]["render_from"], timeline[i]["render_from"] + n]}
        if c["variant"] in COMPOUND_VARIANTS:
            P3.V_MAX = V_MAX                       # same speed cap as the keyframe reveals
            start, segs = arc_camera(g, c["d"])
            start, segs, timing = P3.fit_speed(g, start, segs, n, note)   # speed, margin, density
            s["camera"] = {"type": "compound", "move": f"reveal_{c['variant']}", "start": start,
                           "segments": segs, "timing": timing}
            m = P3.measured(P3.make_path(g, start, segs, timing), n, g)
            note.update({k: m[k] for k in ("speed_first_px_f", "speed_max_px_f", "speed_min_px_f",
                                           "speed_cut_px_f", "turn_max_deg_f", "min_spp", "min_margin_out_px")})
            note.update({"speed_scale": note["shrink"], "shrink_scale": 1.0,
                         "zoom_start_end": [start["zoom"], segs[-1]["zoom"]], "keyframes": None})
        else:
            kfs = v2.enforce(copy.deepcopy(c["keyframes"]), n, g, note)   # fit, density floor, every-frame margin
            kfs, note["speed_scale"] = cap_speed(kfs, n, g, note)
            s["camera"] = {"type": "keyframe_path", "move": f"reveal_{c['variant']}", "keyframes": kfs}
            note.update(motion_stats(kfs, n, g))
            note["zoom_start_end"] = [kfs[0]["zoom"], kfs[-1]["zoom"]]
            note["keyframes"] = kfs
        if g.projection == "rotate":
            s["camera"].update({"projection": "rotate", "hfov_deg": HFOV_DEG})
        notes.append(note)
    plan["provenance"]["planner_config_version"] = f"{PLANNER_VERSION}/v8"
    plan["provenance"].pop("plan_hash", None)
    plan["provenance"]["plan_hash"] = compute_plan_hash(plan)
    parse_plan(plan)
    summary = {"score": round(score, 3), "variants": [c["variant"] for c in chosen],
               "travel": [c["travel"] for c in chosen], "categories": cats,
               "spread_median": thr, "spread_q25": q25, "total_duration_s": plan["total_duration_s"]}
    return plan, {"summary": summary, "shots": notes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--saliency", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    plan, notes = build(json.load(open(args.baseline)), json.load(open(args.saliency)), args.images_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    name = os.path.basename(os.path.normpath(args.out_dir)).replace("tour_", "")
    json.dump(plan, open(os.path.join(args.out_dir, f"{name}.plan.json"), "w"), indent=1)
    json.dump(notes, open(os.path.join(args.out_dir, f"{name}_plan_notes.json"), "w"), indent=1)
    print(json.dumps(notes["summary"]))
    for n in notes["shots"]:
        print(f"{n['shot_id']} {n['category']:<8} {n['variant']:<9} travel {n['travel']:+d} {n['duration_s']:>4}s "
              f"zoom {n['zoom_start_end'][0]:.3f}->{n['zoom_start_end'][1]:.3f} | shrink {n['shrink_scale']} "
              f"scale {n['speed_scale']} spp {n['min_spp']} margin {n['min_margin_out_px']} | speed first {n['speed_first_px_f']} "
              f"max {n['speed_max_px_f']} min {n['speed_min_px_f']} cut {n['speed_cut_px_f']} turn {n['turn_max_deg_f']}")


if __name__ == "__main__":
    main()
