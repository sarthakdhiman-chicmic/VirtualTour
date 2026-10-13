"""Tour v2 planner (worker harness; backend port later). Builds two plans from the recovered baseline:

  v1: baseline ken_burns moves, every transition a hard cut (plan 1.2)
  v2: multi-step keyframe_path moves aimed with image saliency, hard cuts (plan 1.3)

There is no room context (no object, walkway or doorway data): moves are aimed only with the
saliency targets in fixtures/downing/saliency.json. Selection, geometry, direction continuity,
the 1 px margin and the zoom cap follow the tour v2 brief; every decision is written to the notes
file next to the plan.

  python -m tools.tour_v2.planner --baseline fixtures/downing/baseline_plan.json \\
      --saliency fixtures/downing/saliency.json --out-dir out/tour_v2
"""
import argparse
import copy
import itertools
import json
import math
import os
import statistics

from modules.camera_path import KeyframePath, base_window, margin_out_px, source_px_per_output_px
from modules.plan_io import compute_plan_hash, parse_plan
from modules.timeline import compute_timeline

PLANNER_VERSION = "tour_v2-0.1"
MIN_MARGIN_OUT_PX = 1.0       # contract addendum 1.3, checked at every frame
PLAN_MARGIN_OUT_PX = 2.0      # where keyframes are placed, so frames between have room
MIN_SPP = 0.75                # source px per output px at 1080p, every frame
SHRINK_STEP = 0.10

REVEAL_MAX_FRACTION = 0.40
REVEAL_DURATION_S = 7.5       # brief: 7-8 s
SHOT_RANGE_S = (3.0, 8.0)
FINAL_MIN_S = 5.0
SETTLE_BELOW_S = 4.0
CONTINUITY_OVERRIDE_RATIO = 1.5
LATERAL_MOVES = {"reveal_sweep", "push_to_subject", "slide"}

# move parameters (each inside the brief's range)
REVEAL = {"z0": 1.035, "t1": 0.35, "z1": 1.28, "t2": 0.70, "z3": 1.08, "peak_inset": 0.8}
PUSH = {"z0": 1.02, "gain": 1.175, "lead": 0.06, "x_lead_frac": 0.7, "y_lead_frac": 0.4}
SLIDE = {"travel": 0.10, "push": 1.04}
PULL = {"z0": 1.175, "z1": 1.02}
RISE = {"z0": 1.01, "gain": 1.065, "x_toward_peak": 0.5}
SETTLE = {"z0": 1.02, "gain": 1.07, "toward_peak": 0.35}


class Geometry:
    def __init__(self, src_w, src_h, out_w, out_h):
        self.src_w, self.src_h, self.out_w, self.out_h = src_w, src_h, out_w, out_h
        self.bw, self.bh = base_window(src_w, src_h, out_w, out_h)
        # zoom at which source px per output px reaches MIN_SPP
        self.zoom_cap = self.bw * src_w / (MIN_SPP * out_w)

    def half(self, z, margin_px=PLAN_MARGIN_OUT_PX):
        """Half window size plus the margin, normalised."""
        hw, hh = self.bw / (2 * z), self.bh / (2 * z)
        return hw + margin_px * (self.bw / z) / self.out_w, hh + margin_px * (self.bh / z) / self.out_h

    def clamp(self, cx, cy, z):
        hw, hh = self.half(z)
        return min(max(cx, hw), 1 - hw), min(max(cy, hh), 1 - hh)

    def width(self, z):
        return self.bw / z

    def height(self, z):
        return self.bh / z


def kf(t, cx, cy, z):
    return {"t": round(t, 4), "cx": round(cx, 6), "cy": round(cy, 6), "zoom": round(z, 6)}


# ---------------------------------------------------------------------------------------------
# moves: each returns keyframes; d is the lateral direction (+1 right, -1 left) where it applies
# ---------------------------------------------------------------------------------------------

def move_reveal_sweep(g, sal, d):
    """Push toward one side (-d), scroll to the other side (d), pull back from that corner."""
    px, py = sal["peak"]["x"], sal["peak"]["y"]
    p = REVEAL
    k0 = (*g.clamp(0.5, 0.5 + 0.5 * (py - 0.5), p["z0"]), p["z0"])
    hw1, _ = g.half(p["z1"])
    w1 = g.width(p["z1"])
    cx1 = hw1 if d > 0 else 1 - hw1          # push toward the side opposite the scroll direction
    if abs(px - cx1) > p["peak_inset"] * w1 / 2:  # keep the peak inside the frame
        cx1 = px + math.copysign(p["peak_inset"] * w1 / 2, cx1 - px)
    cx1, cy1 = g.clamp(cx1, py, p["z1"])
    cx2, _ = g.clamp(1 - hw1 if d > 0 else hw1, cy1, p["z1"])
    # zoom out keeping the window's edge on that side (and the nearer vertical edge) fixed
    w2, h2 = g.width(p["z1"]), g.height(p["z1"])
    w3, h3 = g.width(p["z3"]), g.height(p["z3"])
    edge_x = cx2 + d * w2 / 2
    cx3 = edge_x - d * w3 / 2
    v = 1 if cy1 >= 0.5 else -1
    edge_y = cy1 + v * h2 / 2
    cy3 = edge_y - v * h3 / 2
    cx3, cy3 = g.clamp(cx3, cy3, p["z3"])
    return [kf(0, *k0), kf(p["t1"], cx1, cy1, p["z1"]), kf(p["t2"], cx2, cy1, p["z1"]), kf(1, cx3, cy3, p["z3"])]


def push_target(g, sal, z):
    """Window centre that puts the peak on the rule-of-thirds point nearest to it."""
    px, py = sal["peak"]["x"], sal["peak"]["y"]
    tx = 1 / 3 if px < 0.5 else 2 / 3
    ty = 1 / 3 if py < 0.5 else 2 / 3
    return g.clamp(px - (tx - 0.5) * g.width(z), py - (ty - 0.5) * g.height(z), z)


def move_push_to_subject(g, sal, d):
    p = PUSH
    z0, z2 = p["z0"], p["z0"] * p["gain"]
    cx2, cy2 = push_target(g, sal, z2)
    cx0, cy0 = g.clamp(cx2 - d * p["lead"] * g.width(z0), 0.5, z0)
    z1 = math.sqrt(z0 * z2)
    # x leads y: a slight lateral curve
    cx1, cy1 = g.clamp(cx0 + p["x_lead_frac"] * (cx2 - cx0), cy0 + p["y_lead_frac"] * (cy2 - cy0), z1)
    return [kf(0, cx0, cy0, z0), kf(0.5, cx1, cy1, z1), kf(1, cx2, cy2, z2)]


def move_slide(g, sal, d):
    p = SLIDE
    # lowest zoom whose window plus the travel fits inside the image
    z0 = 1.0
    while True:
        hw, _ = g.half(z0)
        if 2 * hw + p["travel"] * g.width(z0) <= 1.0:
            break
        z0 = round(z0 + 0.005, 4)
    z1 = z0 * p["push"]
    travel = p["travel"] * g.width(z0)
    mx = sal["peak"]["x"]
    hw0, _ = g.half(z0)
    mx = min(max(mx, hw0 + travel / 2), 1 - hw0 - travel / 2)
    cy = sal["peak"]["y"]
    cx0, cy0 = g.clamp(mx - d * travel / 2, cy, z0)
    cx1, cy1 = g.clamp(mx + d * travel / 2, cy, z1)
    return [kf(0, cx0, cy0, z0), kf(1, cx1, cy1, z1)]


def move_pull_back_reveal(g, sal, d):
    p = PULL
    cx0, cy0 = g.clamp(sal["peak"]["x"], sal["peak"]["y"], p["z0"])
    cx1, cy1 = g.clamp(0.5, 0.5, p["z1"])
    return [kf(0, cx0, cy0, p["z0"]), kf(1, cx1, cy1, p["z1"])]


def move_rise(g, sal, d):
    p = RISE
    z0, z1 = p["z0"], p["z0"] * p["gain"]
    cx = 0.5 + p["x_toward_peak"] * (sal["peak"]["x"] - 0.5)
    _, hh0 = g.half(z0)
    _, hh1 = g.half(z1)
    cx0, cy0 = g.clamp(cx, 1 - hh0, z0)    # low
    cx1, cy1 = g.clamp(cx, hh1, z1)        # high
    return [kf(0, cx0, cy0, z0), kf(1, cx1, cy1, z1)]


def move_settle(g, sal, d):
    p = SETTLE
    z0, z1 = p["z0"], p["z0"] * p["gain"]
    cx0, cy0 = g.clamp(0.5, 0.5, z0)
    cx1, cy1 = g.clamp(0.5 + p["toward_peak"] * (sal["peak"]["x"] - 0.5),
                       0.5 + p["toward_peak"] * (sal["peak"]["y"] - 0.5), z1)
    return [kf(0, cx0, cy0, z0), kf(1, cx1, cy1, z1)]


MOVES = {"reveal_sweep": move_reveal_sweep, "push_to_subject": move_push_to_subject, "slide": move_slide,
         "pull_back_reveal": move_pull_back_reveal, "rise": move_rise, "settle": move_settle}


def natural_direction(move, sal, g=None):
    side = 1 if sal["side"] == "right" else -1
    if move == "slide":
        return side                     # slide toward the salient side
    if move == "reveal_sweep":
        return -side                    # push toward the salient side, then scroll away from it
    if move == "push_to_subject":
        # the window travels from near-centre to the thirds framing of the peak
        cx2, _ = push_target(g, sal, PUSH["z0"] * PUSH["gain"])
        return 1 if cx2 >= 0.5 else -1
    return 0


# ---------------------------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------------------------

def is_exterior(s):
    return s["room_type"] == "exterior" or s["narrative_stage"].startswith("exterior")


def is_bathroom(s):
    return "bath" in s["room_type"]


def select_moves(shots, sal, notes):
    n = len(shots)
    spreads = [sal[s["image_uri"].split("/")[-1]]["spread"] for s in shots]
    thr = statistics.median(spreads)
    moves = [None] * n
    for i, s in enumerate(shots):
        sp = spreads[i]
        if i == n - 1:
            moves[i], why = "pull_back_reveal", "final shot"
        elif is_exterior(s):
            moves[i], why = ("rise", "exterior, first shot") if i == 0 else ("rise", "exterior")
        elif is_bathroom(s):
            moves[i], why = "settle", "bathroom"
        elif s["role"] == "hero":
            moves[i], why = ("reveal_sweep", f"hero, spread {sp:.3f} > median {thr:.3f}") if sp > thr else \
                            ("push_to_subject", f"hero, spread {sp:.3f} <= median {thr:.3f}")
        else:
            why = "other"
        notes[i]["selection"] = why
        notes[i]["spread"] = sp
    # reveal_sweep: never on two consecutive shots, at most 40% of shots (keep the highest spread)
    max_rev = math.floor(REVEAL_MAX_FRACTION * n)
    rev = sorted((i for i in range(n) if moves[i] == "reveal_sweep"), key=lambda i: -spreads[i])
    kept = []
    for i in rev:
        if len(kept) < max_rev and all(abs(i - j) > 1 for j in kept):
            kept.append(i)
        else:
            moves[i] = "push_to_subject"
            notes[i]["selection"] += "; reveal_sweep dropped (consecutive or over 40%) -> push_to_subject"
    # other shots: slide / push_to_subject chosen together so that no two consecutive shots share
    # a move (checked against both neighbours); ties prefer strict alternation starting with slide
    free = [i for i in range(n) if moves[i] is None]
    best = None
    for combo in itertools.product(("slide", "push_to_subject"), repeat=len(free)):
        trial = list(moves)
        for i, m in zip(free, combo):
            trial[i] = m
        repeats = sum(trial[i] == trial[i + 1] for i in range(n - 1))
        alt_breaks = sum(a == b for a, b in zip(combo, combo[1:])) + (combo[0] != "slide" if combo else 0)
        key = (repeats, alt_breaks)
        if best is None or key < best[0]:
            best = (key, combo)
    for i, m in zip(free, best[1]):
        moves[i] = m
        notes[i]["selection"] = "other shot, slide/push chosen against both neighbours"
    notes[0]["consecutive_repeats"] = best[0][0]
    return moves, thr


def allocate_durations(shots, moves, total):
    """reveal_sweep gets REVEAL_DURATION_S; the rest are scaled to keep the total, clamped to
    3..8 s (final >= 5 s) and re-scaled until stable; integer centiseconds, total preserved."""
    n = len(shots)
    lo = [SHOT_RANGE_S[0]] * n
    hi = [SHOT_RANGE_S[1]] * n
    lo[-1] = max(lo[-1], FINAL_MIN_S)
    fixed = {i: REVEAL_DURATION_S for i in range(n) if moves[i] == "reveal_sweep"}
    d = [fixed.get(i, shots[i]["duration_s"]) for i in range(n)]
    for _ in range(20):
        free = [i for i in range(n) if i not in fixed]
        f = (total - sum(d[i] for i in fixed)) / sum(d[i] for i in free)
        d = [d[i] if i in fixed else d[i] * f for i in range(n)]
        clamped = [min(max(x, lo[i]), hi[i]) for i, x in enumerate(d)]
        newly = {i: clamped[i] for i in free if clamped[i] != d[i]}
        d = clamped
        if not newly:
            break
        fixed.update(newly)
    cs = [math.floor(x * 100) for x in d]
    rest = round(total * 100) - sum(cs)
    for i in sorted(range(n), key=lambda i: -(d[i] * 100 - cs[i]))[:rest]:
        cs[i] += 1
    return [c / 100 for c in cs]


def stage_directions(shots, moves, sal, notes):
    """Lateral direction constant within a narrative stage, flipped at each stage change; the
    first stage with a lateral move takes its first lateral shot's natural direction. Conflicts
    with the salient side keep continuity unless the photo's mass ratio is above 1.5."""
    stage_dir, prev_stage, out = None, None, []
    for i, s in enumerate(shots):
        st = s["narrative_stage"]
        if st != prev_stage and prev_stage is not None and stage_dir is not None:
            stage_dir = -stage_dir
        prev_stage = st
        p = sal[s["image_uri"].split("/")[-1]]
        g = Geometry(s["source"]["width"], s["source"]["height"], 1920, 1080)
        nat = natural_direction(moves[i], p, g)
        if moves[i] not in LATERAL_MOVES:
            out.append(nat or 1)
            notes[i]["direction"] = {"natural": nat, "stage": stage_dir, "chosen": nat or 1,
                                     "rule": "not a lateral move"}
            continue
        if stage_dir is None:
            stage_dir = nat
        if nat == stage_dir:
            chosen, rule = nat, "natural = stage"
        elif p["mass_ratio"] > CONTINUITY_OVERRIDE_RATIO:
            chosen, rule = nat, f"salient side wins (mass ratio {p['mass_ratio']:.3f} > {CONTINUITY_OVERRIDE_RATIO})"
        else:
            chosen, rule = stage_dir, f"continuity wins (mass ratio {p['mass_ratio']:.3f} <= {CONTINUITY_OVERRIDE_RATIO})"
        out.append(chosen)
        notes[i]["direction"] = {"natural": nat, "stage": stage_dir, "chosen": chosen, "rule": rule}
    return out


# ---------------------------------------------------------------------------------------------
# margin / cap enforcement on every frame
# ---------------------------------------------------------------------------------------------

def frame_stats(keyframes, n, g):
    frames = KeyframePath(keyframes).frames(n)
    margins = [margin_out_px(cx, cy, z, g.src_w, g.src_h, g.out_w, g.out_h) for cx, cy, z in frames]
    spp = [source_px_per_output_px(z, g.src_w, g.src_h, g.out_w, g.out_h) for _, _, z in frames]
    return frames, margins, spp


def enforce(keyframes, n, g, note):
    # zoom cap: lower the zoom (not the direction); monotone interpolation never exceeds keyframes
    capped = []
    for k in keyframes:
        if k["zoom"] > g.zoom_cap:
            capped.append({"t": k["t"], "from": k["zoom"], "to": round(g.zoom_cap, 6)})
            k["zoom"] = round(g.zoom_cap, 6)
            k["cx"], k["cy"] = (round(v, 6) for v in g.clamp(k["cx"], k["cy"], k["zoom"]))
    note["zoom_capped"] = capped
    scale, base = 1.0, copy.deepcopy(keyframes)
    while True:
        _, margins, spp = frame_stats(keyframes, n, g)
        if min(margins) >= MIN_MARGIN_OUT_PX:
            break
        scale = round(scale - SHRINK_STEP, 3)
        if scale <= 0:
            raise RuntimeError("path cannot be made to fit")
        keyframes = [dict(k, cx=round(0.5 + (b["cx"] - 0.5) * scale, 6), cy=round(0.5 + (b["cy"] - 0.5) * scale, 6))
                     for k, b in zip(keyframes, base)]
    note["shrink_scale"] = scale
    note["min_margin_out_px"] = round(min(margins), 4)
    note["min_spp"] = round(min(spp), 4)
    return keyframes


# ---------------------------------------------------------------------------------------------

def finish(plan):
    plan["provenance"].pop("plan_hash", None)
    plan["provenance"]["plan_hash"] = compute_plan_hash(plan)
    parse_plan(plan)  # validates contract 1.2 / 1.3, including every-frame bounds for paths
    return plan


def build(baseline, saliency):
    base = copy.deepcopy(baseline.get("plan", baseline))
    out_w, out_h = base["output"]["width"], base["output"]["height"]

    # v1: baseline moves, hard cuts
    v1 = copy.deepcopy(base)
    for s in v1["shots"]:
        s["transition_in"] = {"type": "cut", "duration_s": 0.0}
    v1["provenance"]["planner_config_version"] = f"{PLANNER_VERSION}/v1_cuts"
    finish(v1)

    # v2: multi-step moves, hard cuts
    v2 = copy.deepcopy(base)
    v2["plan_version"] = "1.3"
    shots = v2["shots"]
    sal = saliency["photos"]
    notes = [{"shot_id": s["shot_id"], "role": s["role"], "room_type": s["room_type"],
              "narrative_stage": s["narrative_stage"], "baseline_duration_s": s["duration_s"]} for s in shots]
    moves, thr = select_moves(shots, sal, notes)
    durations = allocate_durations(shots, moves, base["total_duration_s"])
    # shots that ended under 4 s become settle
    for i, d in enumerate(durations):
        if d < SETTLE_BELOW_S and moves[i] != "settle" and i != len(shots) - 1:
            notes[i]["selection"] += f"; duration {d} s < {SETTLE_BELOW_S} s -> settle"
            moves[i] = "settle"
    directions = stage_directions(shots, moves, sal, notes)
    for i, s in enumerate(shots):
        s["duration_s"] = durations[i]
        s["transition_in"] = {"type": "cut", "duration_s": 0.0}
    v2["total_duration_s"] = round(sum(durations), 2)
    timeline = compute_timeline(parse_plan_loose(v2))
    for i, s in enumerate(shots):
        p = sal[s["image_uri"].split("/")[-1]]
        g = Geometry(s["source"]["width"], s["source"]["height"], out_w, out_h)
        keys = MOVES[moves[i]](g, p, directions[i])
        notes[i]["keyframes_designed"] = copy.deepcopy(keys)
        keys = enforce(keys, timeline[i]["render_frames"], g, notes[i])
        s["camera"] = {"type": "keyframe_path", "move": moves[i], "keyframes": keys}
        # direction the path actually travels (reveal_sweep: its scroll K1 -> K2)
        a, b = (keys[1], keys[2]) if moves[i] == "reveal_sweep" else (keys[0], keys[-1])
        dx = b["cx"] - a["cx"]
        achieved = (dx > 1e-6) - (dx < -1e-6)
        notes[i]["direction"]["achieved"] = achieved
        notes[i]["direction"]["achieved_dx"] = round(dx, 6)
        if moves[i] in LATERAL_MOVES and achieved != directions[i]:
            notes[i]["direction"]["conflict"] = (
                "push_to_subject ends with the peak on its nearest thirds point; from a near-full start "
                "that fixes the travel direction, so the chosen direction cannot be honoured"
                if moves[i] == "push_to_subject" else "geometry could not honour the chosen direction")
        notes[i].update({"move": moves[i], "duration_s": durations[i], "frames": [timeline[i]["render_from"],
                         timeline[i]["render_from"] + timeline[i]["render_frames"]],
                         "keyframes": keys, "zoom_cap": round(g.zoom_cap, 6),
                         "saliency": {k: p[k] for k in ("peak", "side", "mass_ratio", "spread")}})
    v2["provenance"]["planner_config_version"] = f"{PLANNER_VERSION}/v2_moves"
    finish(v2)
    summary = {"spread_median_threshold": thr, "moves": moves,
               "reveal_sweep_count": moves.count("reveal_sweep"), "total_duration_s": v2["total_duration_s"],
               "baseline_total_s": base["total_duration_s"]}
    return v1, v2, {"summary": summary, "shots": notes}


def parse_plan_loose(plan):
    """Plan object for timeline maths before cameras exist (no validation)."""
    from modules.plan_io import Plan
    p = copy.deepcopy(plan)
    for s in p["shots"]:
        s["camera"] = {"type": "ken_burns", "start": {"cx": .5, "cy": .5, "zoom": 1}, "end": {"cx": .5, "cy": .5, "zoom": 1}}
    p["provenance"].setdefault("plan_hash", "")
    return Plan(**p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--saliency", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    v1, v2, notes = build(json.load(open(args.baseline)), json.load(open(args.saliency)))
    os.makedirs(args.out_dir, exist_ok=True)
    for name, obj in (("v1_cuts.plan.json", v1), ("v2_moves.plan.json", v2), ("v2_plan_notes.json", notes)):
        json.dump(obj, open(os.path.join(args.out_dir, name), "w"), indent=1)
    print(json.dumps(notes["summary"]))
    for n in notes["shots"]:
        d = n["direction"]
        print(f"{n['shot_id']} {n['role']:<6} {n['room_type']:<17} {n['narrative_stage']:<13} {n['move']:<17} "
              f"{n['baseline_duration_s']}->{n['duration_s']}s dir nat {d['natural']:+d} stage {d['stage']} chosen {d['chosen']:+d} "
              f"| shrink {n['shrink_scale']} cap {n['zoom_capped']} margin {n['min_margin_out_px']} spp {n['min_spp']} | {n['selection']} | {d['rule']}")


if __name__ == "__main__":
    main()
