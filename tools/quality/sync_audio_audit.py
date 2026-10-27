"""Audit a rendered tour against ITS OWN audio: every cut, scored on the shared accent scale
(tour_worker/music/analysis.py accent_curve) - the hit the cut lands on (strike) and the hardest
sound up to a beat before it (pre). pre > LATE_RATIO x strike: the cut reads late, a beat after a
harder sound (review 2026-10-23: "some are now after the ting").

  python -m tools.quality.sync_audio_audit <tour.mp4> <plan.json> <track analysis v4 .json>

The render's audio is scaled with the TRACK's reference (analysis accent_ref): a render's own
per-band medians differ (a 68 s cut of the track is not the track), which made the same moment
score differently in the planner and here.

The planner optimises the same numbers (analysis v4 strike / pre on the track), so a cut this
flags on a render is one the planner could see.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile

import librosa
import numpy as np

from tour_worker.music.analysis import LATE_RATIO, PRE_WIN, SR, STRIKE_WIN, accent_curve, accent_peaks
from tour_worker.planner.music_sync_v2 import clarity, peaks_window_max

UNCLEAR_LIMIT = 1.8      # = tools/music/sync_regression.py: a hit that barely stands out


def audit(video: str, plan: dict, analysis: dict) -> list[dict]:
    plan = plan.get("plan", plan)
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-vn", "-ac", "1", "-ar", str(SR), tmp.name],
                       check=True)
        y, _ = librosa.load(tmp.name, sr=SR)
    acc, fr, _ = accent_curve(y, ref=analysis.get("accent_ref"))
    peaks = accent_peaks(acc, fr)
    pt = np.array([p[0] for p in peaks]); pv = np.array([p[1] for p in peaks])
    fps = plan["output"]["fps"]
    lead = (plan.get("music") or {}).get("cut_lead_frames") or 0
    rows = []
    for c in np.cumsum([s["duration_s"] for s in plan["shots"]])[:-1]:
        beat = float(c) + lead / fps                       # the cut leads its beat by lead frames
        s, p = peaks_window_max(pt, pv, beat, STRIKE_WIN), peaks_window_max(pt, pv, beat, PRE_WIN)
        cl = clarity(pt, pv, beat, s)
        rows.append({"cut": round(float(c), 3), "strike": round(s, 2), "pre": round(p, 2), "late": p > LATE_RATIO * s,
                     "clarity": round(cl, 2), "unclear": cl < UNCLEAR_LIMIT})
    return rows


def main(argv=None) -> int:
    a = (argv or sys.argv[1:])
    rows = audit(a[0], json.load(open(a[1])), json.load(open(a[2])))
    late = [r for r in rows if r["late"]]
    unclear = [r for r in rows if r["unclear"]]
    for r in late:
        print(f"  late: cut {r['cut']:7.2f}s  strike {r['strike']:5.1f}  harder sound before {r['pre']:5.1f}")
    for r in unclear:
        print(f"  unclear: cut {r['cut']:7.2f}s  stands out only x{r['clarity']:.2f} from the music around it")
    print(f"{len(late)} of {len(rows)} cuts read late (a harder sound up to a beat before); "
          f"{len(unclear)} sit on a beat that does not stand out")
    return 1 if late or unclear else 0


if __name__ == "__main__":
    sys.exit(main())
