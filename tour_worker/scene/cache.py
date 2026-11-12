"""Scene files on local disk, keyed by sha256: fetched once, checked, reused by later renders (and,
in Phase 3, by scene_prep). A file whose hash does not match the plan's is never used."""
from __future__ import annotations

import hashlib
import os
import tempfile

from tour_worker.scene import SceneInvalid

HASH_CHUNK = 8 << 20


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(HASH_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def ensure_scene(uri: str, sha256: str, fmt: str, storage, cache_dir: str, fetch_uri: str | None = None) -> str:
    """The local path of the scene file with this hash: from the cache, else fetched (fetch_uri:
    e.g. a presigned URL for uri) and checked. Raises SceneInvalid when the file's hash differs."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{sha256}.{fmt}")
    if os.path.exists(path):
        return path
    fd, tmp = tempfile.mkstemp(prefix=".fetch_", suffix=f".{fmt}", dir=cache_dir)
    os.close(fd)
    try:
        storage.get(fetch_uri or uri, tmp)
        got = sha256_file(tmp)
        if got != sha256:
            raise SceneInvalid(f"scene file hash {got[:12]}... does not match the plan's {sha256[:12]}...")
        os.replace(tmp, path)                       # atomic: a half-written file is never cached
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path
