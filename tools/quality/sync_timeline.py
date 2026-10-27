"""A picture of a synced tour against its music: loudness per beat, sections, the hero moments
(tier 1 red, tier 2 blue), the cuts (black) and the zoom punches (orange), on the track as played.

  python -m tools.quality.sync_timeline <plan.json> <analysis v4 .json> <out.png>
"""
from __future__ import annotations

import json
import sys

from types import SimpleNamespace

from PIL import Image, ImageDraw

from tools.quality.sync_quality import played

W, ROW_H, PAD, PX_PER_S = 2400, 260, 40, None
COL = {"bg": (250, 250, 247), "grid": (225, 225, 220), "loud": (150, 160, 172), "t1": (208, 64, 43),
       "t2": (43, 111, 208), "cut": (28, 28, 26), "punch": (230, 140, 20), "text": (90, 90, 85),
       "sec": [(240, 238, 230), (232, 236, 242)], "edit": (140, 60, 200)}


def draw(plan: dict, an: dict, out: str) -> None:
    m = plan.get("music") or {}
    start, end = float(m.get("start_s") or 0.0), float(plan["total_duration_s"])
    music = SimpleNamespace(**{**m, "segments": [SimpleNamespace(**s) for s in m.get("segments") or []] or None})

    def tour_times(rows, key):
        return [(t, r) for t, r in played(an, music, rows, key) if 0 <= t <= end]

    rows = max(1, int(end // 30) + 1)                     # 30 s per row
    img = Image.new("RGB", (W, rows * ROW_H + 2 * PAD), COL["bg"])
    g = ImageDraw.Draw(img)
    span = 30.0
    px = (W - 2 * PAD) / span

    def xy(t):
        r = int(t // span)
        return PAD + (t - r * span) * px, PAD + r * ROW_H

    loud = tour_times(an.get("beat_energy") or [], "t")
    lo = min((r["loud_db"] for _, r in loud), default=-40)
    hi = max((r["loud_db"] for _, r in loud), default=0)
    for i, (t, sec) in enumerate(tour_times(an.get("sections") or [], "start")):
        x, y = xy(t)
        g.text((x + 3, y + 4), sec.get("level", ""), fill=COL["text"])
    for t, r in loud:                                     # loudness bars
        x, y = xy(t)
        h = (r["loud_db"] - lo) / max(1e-6, hi - lo) * (ROW_H - 90)
        g.rectangle([x, y + ROW_H - 40 - h, x + 2, y + ROW_H - 40], fill=COL["loud"])
    for r_ in range(rows):                                # time grid
        y = PAD + r_ * ROW_H
        for s in range(0, 31, 5):
            x = PAD + s * px
            g.line([x, y + 20, x, y + ROW_H - 40], fill=COL["grid"])
            g.text((x + 2, y + ROW_H - 36), f"{r_ * span + s:.0f}s", fill=COL["text"])
    for t, h in tour_times(an.get("heroes") or [], "t"):
        x, y = xy(t)
        c = COL["t1"] if h["tier"] == 1 else COL["t2"]
        g.ellipse([x - 6, y + 22, x + 6, y + 34], fill=c)
        g.text((x + 7, y + 22), h["kind"], fill=c)
    cum = 0.0
    for s in plan["shots"]:
        for p in s["camera"].get("punches") or []:
            x, y = xy(cum + p["at_s"])
            g.polygon([(x, y + 40), (x - 6, y + 52), (x + 6, y + 52)], fill=COL["punch"])
        cum += s["duration_s"]
        if cum < end - 1e-6:
            x, y = xy(cum)
            g.line([x, y + 38, x, y + ROW_H - 40], fill=COL["cut"], width=3)
    for e in m.get("edits") or []:                       # music edits: where the track jumps
        x, y = xy(float(e["tour_s"]))
        for yy in range(int(y + 20), int(y + ROW_H - 40), 8):
            g.line([x, yy, x, yy + 4], fill=COL["edit"], width=2)
        g.text((x + 4, y + ROW_H - 56), f"edit {e['from_s']:.1f}->{e['to_s']:.1f}s", fill=COL["edit"])
    g.text((PAD, 8), f"{m.get('track_id', '')} from {start:.1f} s · black: cuts · orange: punches · "
                     f"red: tier-1 heroes · blue: tier-2 heroes · purple dashes: music edits · grey: loudness per beat",
           fill=COL["text"])
    img.save(out)


if __name__ == "__main__":
    plan = json.load(open(sys.argv[1]))
    draw(plan.get("plan", plan), json.load(open(sys.argv[2])), sys.argv[3])
