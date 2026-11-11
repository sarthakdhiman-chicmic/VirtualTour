"""Stage timings of one tour, as the client asked for them (review 2026-11-05: "a 4K render ... on the
A100 80GB from a cold cache, with timings for each stage"; target: a 90 s 4K tour in under 3 minutes,
up to 6 for complex listings).

  python -m tools.timing_report <timings.json> [--markdown out.md]

<timings.json>: the backend tour job's extra["timings"] ({"segment": ..., "plan": ..., "render": ...}),
or one job result with a "timings" key. Each job's timings come from tour_worker/timing.py:
{"stages": {name: s}, "total_s", "host": {...}, ...}."""
import argparse
import json
import sys

TARGET_S, COMPLEX_S, REFERENCE_VIDEO_S = 180.0, 360.0, 90.0
ORDER = {"segment": ("model_load", "fetch", "inference", "mask_upload"),
         "plan": ("fetch", "scoring", "planning"),
         "render": ("fetch", "draw_encode", "audio", "join", "verify", "upload")}


def report(timings: dict) -> str:
    if "timings" in timings:                      # a single job result
        timings = {"render" if "video_uri" in timings else "plan" if "plan" in timings else "segment":
                   timings["timings"]}
    rows, total = [], 0.0
    for job in ("segment", "plan", "render"):
        t = timings.get(job)
        if not t:
            continue
        stages = t.get("stages") or {}
        names = [n for n in ORDER[job] if n in stages] + [n for n in stages if n not in ORDER[job]]
        for n in names:
            rows.append((job, n, stages[n]))
        rows.append((job, "**total**", t.get("total_s", sum(stages.values()))))
        total += float(t.get("total_s", sum(stages.values())))
    render = timings.get("render") or {}
    host = render.get("host") or next((t.get("host") for t in timings.values() if isinstance(t, dict) and t.get("host")), {})
    video_s = render.get("video_s")
    lines = ["| Job | Stage | Seconds |", "|---|---|---:|"]
    lines += [f"| {j} | {n} | {s:.1f} |" for j, n, s in rows]
    lines += ["", f"**End to end: {total:.1f} s ({total / 60:.1f} min)**"]
    if video_s:
        scaled = total * REFERENCE_VIDEO_S / float(video_s)
        verdict = ("within the 3-minute target" if scaled <= TARGET_S else
                   "within the 6-minute complex-listing bound" if scaled <= COMPLEX_S else "over both targets")
        lines.append(f"Video {video_s:.1f} s at {render.get('width')}x{render.get('height')}; scaled to a "
                     f"90 s tour: {scaled:.0f} s, {verdict}.")
    if host:
        lines.append("Host: " + ", ".join(f"{k} {v}" for k, v in host.items()))
    if render.get("render_workers"):
        lines.append(f"Render workers: {render['render_workers']}, chunks: {render.get('chunks')}, frames: {render.get('frames')}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("timings")
    ap.add_argument("--markdown")
    a = ap.parse_args(argv)
    text = report(json.load(open(a.timings)))
    if a.markdown:
        open(a.markdown, "w").write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
