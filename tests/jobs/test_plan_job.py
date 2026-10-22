"""tour_worker.jobs.plan: tour_plan = fetch photos -> scoring -> build_shot_plan -> parse_plan."""
import copy
import os

import cv2
import numpy as np
import pytest

from tour_worker.contract.jobs import JobFailure, TourPlanResult
from tour_worker.contract.plan import compute_plan_hash, parse_plan
from tour_worker.jobs.router import run_job, validate_request

ROOMS = ["exterior", "living_room", "living_room", "kitchen", "kitchen", "dining_room",
         "bedroom", "primary_bathroom"]


def _photo(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur((rng.random((480, 720, 3)) * 255).astype(np.uint8), (0, 0), 3)
    for _ in range(6):
        x, y = int(rng.integers(0, 660)), int(rng.integers(0, 440))
        cv2.rectangle(img, (x, y), (x + 50, y + 30), tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
        cv2.line(img, (int(rng.integers(0, 720)), 0), (int(rng.integers(0, 720)), 479), (20, 20, 20), 3)
    return cv2.imencode(".jpg", img)[1].tobytes()


PHOTOS = {f"s3://bucket/listings/L1/p{i:02d}.jpg": _photo(i) for i in range(len(ROOMS))}


class FakeStorage:
    """Serves PHOTOS by URI; records what was fetched. Never writes."""

    def __init__(self, photos=None, broken=()):
        self.photos = dict(PHOTOS if photos is None else photos)
        self.broken = set(broken)
        self.got = []

    def get(self, uri, dest):
        self.got.append(uri)
        if uri in self.broken:
            raise RuntimeError("HTTP fetch failed for " + uri)
        with open(dest, "wb") as f:
            f.write(self.photos[uri])

    def put(self, path, uri):
        raise AssertionError("tour_plan writes nothing")


def _request(**kw):
    req = {"job_type": "tour_plan", "job_id": "j1", "listing_id": "L1",
           "photos": [{"photo_id": f"img_{i}", "uri": uri, "room_type": rt}
                      for i, (uri, rt) in enumerate(zip(PHOTOS, ROOMS))]}
    req.update(kw)
    return req


def test_plans_a_tour_and_validates_it_with_the_renderers_checks():
    st = FakeStorage()
    res = run_job(_request(), storage=st)
    TourPlanResult.model_validate(res)
    plan = res["plan"]
    parse_plan(plan)
    assert res["plan_hash"] == plan["provenance"]["plan_hash"] == compute_plan_hash(plan)
    assert plan["listing_id"] == "L1" and plan["plan_version"] == "1.3"
    assert (plan["output"]["width"], plan["output"]["height"]) == (1920, 1080)
    # the plan keeps the canonical URIs; the photo size is the decoded size
    assert {s["image_uri"] for s in plan["shots"]} <= set(PHOTOS)
    assert all(s["source"] == {"width": 720, "height": 480} for s in plan["shots"])
    assert res["scoring"]["photos"] == len(ROOMS) and res["scoring"]["sequence"]
    assert res["warnings"] == [] and sorted(st.got) == sorted(PHOTOS)


def test_same_plan_as_scoring_and_planner_called_directly():
    """The job adds nothing of its own to the plan: the backend's former call, reproduced."""
    from tour_worker.planner.config import PlannerConfig
    from tour_worker.planner.planner import build_shot_plan
    from tour_worker.scoring.pipeline import score_photos
    inputs = [(f"img_{i}", data, rt) for i, (data, rt) in enumerate(zip(PHOTOS.values(), ROOMS))]
    assets = {f"img_{i}": {"uri": uri, "width": 720, "height": 480} for i, uri in enumerate(PHOTOS)}
    cfg = PlannerConfig(output_width=1080, output_height=1920, max_total_s=60.0)
    direct = build_shot_plan(score_photos(inputs), "L1", assets, cfg=cfg).model_dump(mode="json")
    res = run_job(_request(options={"output_width": 1080, "output_height": 1920, "max_total_s": 60}),
                  storage=FakeStorage())
    assert res["plan"] == direct


def test_image_urls_are_fetched_instead_and_stay_out_of_the_plan():
    signed = {uri: f"https://signed.example/{i}?X-Amz-Signature=sig" for i, uri in enumerate(PHOTOS)}
    photos = {signed[u]: d for u, d in PHOTOS.items()}
    st = FakeStorage(photos)
    res = run_job(_request(image_urls=signed), storage=st)
    assert res["status"] == "completed" and sorted(st.got) == sorted(signed.values())
    assert "signed.example" not in str(res["plan"])


def test_segments_and_music_reach_the_planner():
    region = {"object_id": "seg_000", "class_name": "curtain", "confidence": 0.99,
              "bbox": [200, 60, 420, 400], "space": [720, 480], "mask_url": "s3://b/seg/img_1/seg_000.png"}
    segments = {f"img_{i}": [dict(region, mask_url=f"s3://b/seg/img_{i}/seg_000.png")]
                for i in range(len(ROOMS))}
    res = run_job(_request(segments=segments), storage=FakeStorage())
    assert res["status"] == "completed" and res["plan"]["plan_version"] == "1.4"
    masks = [r["mask_uri"] for s in res["plan"]["shots"] for r in s.get("motion_regions") or []]
    assert masks and all(m.startswith("s3://b/seg/") for m in masks)

    beat, dur = 0.5, 120.0                                 # a steady 120 bpm track
    cands = [{"time": round(0.25 + i * beat, 4), "bar_start": i % 4 == 0, "kind": "beat",
              "value": 0.5 if i % 32 == 0 else 0.0, "change": 0.0, "what_happens": "beat"}
             for i in range(int((dur - 0.3) / beat))]
    analysis = {"analysis_version": "1", "duration": dur, "music_start": 0.0, "music_end": dur - 0.5,
                "tempo_bpm": 120.0, "has_beat": True, "beat_period_s": beat, "beats": [],
                "candidates": cands}
    track = {"track_id": "t1", "s3_uri": "s3://b/music/t1.mp3", "sha256": "ab" * 32, "analysis": analysis}
    res = run_job(_request(music_track=track), storage=FakeStorage())
    assert res["status"] == "completed", res
    assert res["plan"]["plan_version"] == "1.5" and res["plan"]["music"]["uri"] == "s3://b/music/t1.mp3"


def test_a_photo_that_cannot_be_fetched_or_read_is_left_out_with_a_warning():
    uris = list(PHOTOS)
    photos = dict(PHOTOS)
    photos[uris[2]] = b"not an image"
    res = run_job(_request(), storage=FakeStorage(photos, broken={uris[1]}))
    assert res["status"] == "completed"
    assert res["warnings"][0].startswith("photo_fetch_failed:img_1:")
    assert res["warnings"][1] == "photo_unreadable:img_2"
    assert {s["asset_id"] for s in res["plan"]["shots"]}.isdisjoint({"img_1", "img_2"})


def test_no_usable_photo_is_image_fetch_failed():
    res = run_job(_request(), storage=FakeStorage(broken=set(PHOTOS)))
    JobFailure.model_validate(res)
    assert res["error"]["code"] == "image_fetch_failed"


def test_a_planner_crash_is_planning_failed(monkeypatch):
    import tour_worker.planner.planner as planner

    def boom(*a, **k):
        raise ZeroDivisionError("division by zero")

    monkeypatch.setattr(planner, "build_shot_plan", boom)
    res = run_job(_request(), storage=FakeStorage())
    assert res["error"]["code"] == "planning_failed" and "ZeroDivisionError" in res["error"]["message"]


def test_a_plan_the_validator_rejects_keeps_the_validators_code(monkeypatch):
    import tour_worker.jobs.plan as plan_job

    def reject(plan):
        raise ValueError("unsupported_camera_type: s01 uses warp_drive")

    monkeypatch.setattr(plan_job, "parse_plan", reject)
    res = run_job(_request(), storage=FakeStorage())
    assert res["error"]["code"] == "unsupported_camera_type"


@pytest.mark.parametrize("change", [
    {"listing_id": ""},
    {"photos": []},
    {"photos": [{"photo_id": "a", "uri": "s3://b/a.jpg"}, {"photo_id": "a", "uri": "s3://b/b.jpg"}]},
    {"photos": [{"photo_id": "a"}]},
    {"options": {"fps": 60}},                              # options are explicit: no unknown keys
    {"options": {"output_width": 0}},
    {"music_track": {"track_id": "t"}},
])
def test_bad_requests_are_invalid_request_before_any_fetch(change):
    st = FakeStorage()
    req = _request(**change)
    res = run_job(req, storage=st)
    assert res["error"]["code"] == "invalid_request" and st.got == []
    assert validate_request(req) == res


def test_the_runpod_handler_runs_tour_plan(tmp_path, monkeypatch):
    """Both front doors share the router: handler.py here, the HTTP service in tests/service."""
    from handler import handler
    photos = []
    for i, (uri, rt) in enumerate(zip(PHOTOS, ROOMS)):
        p = tmp_path / os.path.basename(uri)
        p.write_bytes(PHOTOS[uri])
        photos.append({"photo_id": f"img_{i}", "uri": "file://" + str(p), "room_type": rt})
    res = handler({"input": _request(photos=photos)})
    assert res["status"] == "completed", res
    expected = run_job(_request(), storage=FakeStorage())["plan"]
    ours = copy.deepcopy(res["plan"])
    # the same plan but for the photo URIs (and so the hash)
    for a, b in zip(ours["shots"], expected["shots"]):
        assert os.path.basename(a.pop("image_uri")) == os.path.basename(b.pop("image_uri"))
    ours["provenance"].pop("plan_hash"); expected["provenance"].pop("plan_hash")
    assert ours == expected
