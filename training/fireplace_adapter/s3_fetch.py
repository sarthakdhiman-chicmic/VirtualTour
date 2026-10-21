"""READ-ONLY: pick photo-like objects from the S3 index, dedupe by ETag, download distinct photos.

Run with the BACKEND venv (S3 client + settings), after s3_list.py:
  <backend>/.venv/bin/python training/fireplace_adapter/s3_fetch.py

- photo-like: real photos people uploaded (listings/<id>/original.jpg, uploads/.../original/...,
  user uploads, photos/, ...). Skipped: masks, depth, detections, every generated output
  (empty / empty_pass / empty_room / object_removal / instruction_edit / sku_replace / replaced /
  staged / detected), caches of generated images, perceptual-demo outputs, thumbnails.
- uploads/perceptual_demo/* are the known evaluation listings (turkey, downing): left out, and so is
  any object whose bytes match a local known-listing photo (md5), so the evaluation stays held out.
- photos are saved as photos/<etag-or-md5>.<ext>; manifest.jsonl gets one row per distinct photo:
  {hash, key, size, folder, ext}. Only GetObject is used; no URL or credential is ever printed.
Resumable: photos already on disk are skipped.
"""
import hashlib
import json
import os
import re
import sys

BACKEND = "/home/sarthak/Desktop/Articulait/articulait_backend"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "out", "fireplace_adapter")
PHOTOS = os.path.join(OUT, "photos")
KNOWN_DIRS = [os.path.join(ROOT, "out", d, "images") for d in
              ("perceptual_turkey", "perceptual_downing", "perceptual_turkey_mix", "backend_downing")]
KNOWN_DIRS += ["/home/sarthak/Downloads/603 S Downing Street, Denver, CO 80209 _ MLS #6934339 _ Zillow",
               "/home/sarthak/Downloads/1030 Turkey Trot, Greensboro, GA 30642 _ MLS #71882 _ Zillow",
               os.path.join(ROOT, "out", "mixed_turkey_src")]

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")
SKIP = re.compile(r"/masks/|/depth|detected|detections|empty|object_removal|instruction_edit|sku_replace|"
                  r"replaced|staged|/thumbs/|detect_cache/|empty_room_cache/|perceptual_demo|virtual_tour/|certificates/|"
                  r"test_pipeline|debug_|/services/|Screenshot|Babel_", re.I)
# --wide (2026-10-21, user: take every remaining real photo): also screenshots / service images /
# small files / jfif-heic-avif uploads; generated outputs, masks and the eval listings stay out
WIDE_EXT = IMG_EXT + (".jfif", ".heic", ".avif")
WIDE_SKIP = re.compile(r"/masks/|/depth|detected|detections|empty|object_removal|instruction_edit|sku_replace|"
                       r"replaced|staged|/thumbs/|detect_cache/|empty_room_cache/|perceptual_demo|virtual_tour/|"
                       r"certificates/|test_pipeline|debug_", re.I)


def folder_of(key: str) -> str:
    p = key.split("/")
    if p[0] == "uploads" and len(p) > 3:
        return "/".join(p[:3])
    return "/".join(p[:2]) if len(p) > 2 else p[0]


def known_md5() -> set[str]:
    out = set()
    for d in KNOWN_DIRS:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith(IMG_EXT):
                    out.add(hashlib.md5(open(os.path.join(d, f), "rb").read()).hexdigest())
    return out


WIDE = "--wide" in sys.argv


def main() -> None:
    sys.path.insert(0, BACKEND)
    os.chdir(BACKEND)
    from app.core.config import settings
    from app.core.s3_client import S3Client
    os.makedirs(PHOTOS, exist_ok=True)
    rows = [json.loads(line) for line in open(os.path.join(OUT, "s3_index.jsonl"))]
    known = known_md5()
    cand, seen = [], set()
    for r in sorted(rows, key=lambda r: r["key"]):
        k = r["key"]
        if WIDE:
            if not k.lower().endswith(WIDE_EXT) or WIDE_SKIP.search(k):
                continue
        elif not k.lower().endswith(IMG_EXT) or SKIP.search(k) or r["size"] < 20_000:
            continue
        if r["etag"] in seen or r["etag"] in known:
            continue
        seen.add(r["etag"])
        cand.append(r)
    print(f"candidates (distinct ETag, photo-like): {len(cand)}; known-listing photos excluded: {len(known)} md5s",
          flush=True)
    man_path = os.path.join(OUT, "manifest.jsonl")
    done = {json.loads(line)["hash"] for line in open(man_path)} if os.path.exists(man_path) else set()
    s3 = S3Client()
    bucket = settings.S3_BUCKET_ASSETS
    todo = [r for r in cand if r["etag"].replace("-", "_") not in done]

    def get(r):                                         # GetObject only (read-only)
        try:
            return r, s3.download_file(r["key"], bucket=bucket), None
        except Exception as e:                          # noqa: BLE001 - log the type only, never the message
            return r, None, type(e).__name__

    from concurrent.futures import ThreadPoolExecutor
    n_new = n_fail = 0
    with open(man_path, "a") as man, ThreadPoolExecutor(8) as pool:
        for i, (r, data, err) in enumerate(pool.map(get, todo)):
            h = r["etag"].replace("-", "_")                 # multipart ETags carry "-N"
            if data is None:
                n_fail += 1
                print(f"fail {h} {err}", flush=True)
                continue
            ext = os.path.splitext(r["key"])[1].lower()
            md5 = hashlib.md5(data).hexdigest()
            if md5 in known:
                continue
            with open(os.path.join(PHOTOS, h + ext), "wb") as fh:
                fh.write(data)
            man.write(json.dumps({"hash": h, "key": r["key"], "size": r["size"], "folder": folder_of(r["key"]),
                                  "ext": ext, "md5": md5}) + "\n")
            man.flush()
            n_new += 1
            if n_new % 100 == 0:
                print(f"{i + 1}/{len(todo)} downloaded {n_new}", flush=True)
    print(f"done: new {n_new}, failed {n_fail}, total in manifest {len(done)}", flush=True)


if __name__ == "__main__":
    main()
