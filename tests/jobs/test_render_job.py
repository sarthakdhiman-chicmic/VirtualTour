"""tour_worker.jobs.render.run_tour: the music track is fetched through image_urls like photos and
masks (a presigned URL when the worker cannot read the bucket), else as written."""
import os

import pytest

import tour_worker.jobs.render as render_job
from tests.perceptual_fixtures import _plan, _region
from tour_worker.contract.plan import compute_plan_hash

MUSIC_URI = "s3://music-bucket/library/track.mp3"
SIGNED = "https://music-bucket.s3.amazonaws.com/library/track.mp3?X-Amz-Signature=abc"


class Storage:
    def __init__(self, fail=False):
        self.gets, self.puts, self.fail = [], [], fail

    def get(self, uri, dest):
        self.gets.append((uri, dest))
        if self.fail:
            raise RuntimeError("fetch failed")
        with open(dest, "wb") as f:
            f.write(b"ID3")

    def put(self, path, uri):
        self.puts.append(uri)


def _music_plan():
    p = _plan([_region()], version="1.5")
    p["music"] = {"track_id": "t", "uri": MUSIC_URI, "sha256": "0" * 64, "end_s": 1.0, "synced": True}
    p["provenance"]["plan_hash"] = compute_plan_hash(p)
    return p


@pytest.fixture
def no_render(monkeypatch, tmp_path):
    seen = {}

    def fake_render(plan, workdir, warnings=None, music_path=None, workers=None):
        seen["music_path"] = music_path
        out = os.path.join(workdir, "out.mp4")
        open(out, "wb").close()
        return out

    monkeypatch.setattr(render_job, "fetch_and_prepare", lambda *a, **k: [])
    monkeypatch.setattr(render_job, "render_direct", fake_render)
    monkeypatch.setattr(render_job, "verify", lambda *a, **k: None)
    return seen


@pytest.mark.parametrize("image_urls, fetched", [
    ({MUSIC_URI: SIGNED}, SIGNED),       # mapped: the fetch URL
    ({}, MUSIC_URI),                     # not mapped: the URI as written
])
def test_music_uri_is_resolved_through_image_urls(no_render, tmp_path, image_urls, fetched):
    storage = Storage()
    res = render_job.run_tour({"plan": _music_plan(), "output_prefix": "s3://b/o/",
                               "image_urls": image_urls}, storage, str(tmp_path))
    assert res["status"] == "completed", res
    assert [u for u, _ in storage.gets] == [fetched]
    # the local file is named after the plan's URI, never the signed URL's query string
    assert os.path.basename(storage.gets[0][1]) == "track.mp3"
    assert no_render["music_path"] == storage.gets[0][1]
    assert "music_fetch_failed" not in res["warnings"]


def test_a_failed_music_fetch_renders_without_music(no_render, tmp_path):
    res = render_job.run_tour({"plan": _music_plan(), "output_prefix": "s3://b/o/",
                               "image_urls": {MUSIC_URI: SIGNED}}, Storage(fail=True), str(tmp_path))
    assert res["status"] == "completed"
    assert "music_fetch_failed" in res["warnings"] and no_render["music_path"] is None
