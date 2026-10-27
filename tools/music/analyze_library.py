"""Re-analyse every track of a music library folder (analysis v4) - run once after the analysis
changes; each track's JSON is written next to it as <name>.analysis.v4.json.

  python -m tools.music.analyze_library <folder> [--force]
"""
from __future__ import annotations

import argparse
import json
import os
import time

from tour_worker.music.analysis import ANALYSIS_VERSION, make_analysis_v2

AUDIO = (".mp3", ".wav", ".m4a", ".flac", ".ogg")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for name in sorted(os.listdir(a.folder)):
        if not name.lower().endswith(AUDIO):
            continue
        out = os.path.join(a.folder, f"{os.path.splitext(name)[0]}.analysis.v{ANALYSIS_VERSION}.json")
        if os.path.exists(out) and not a.force:
            print(f"{name}: up to date")
            continue
        t0 = time.time()
        an = make_analysis_v2(os.path.join(a.folder, name))
        an["audio"] = name                                 # not the machine's path
        with open(out, "w") as f:
            json.dump(an, f, indent=1)
        print(f"{name}: {time.time() - t0:.0f} s, {len(an['sections'])} sections, "
              f"{sum(h['tier'] == 1 for h in an['heroes'])} tier-1 / "
              f"{sum(h['tier'] == 2 for h in an['heroes'])} tier-2 heroes")


if __name__ == "__main__":
    main()
