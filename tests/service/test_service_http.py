"""tour_worker.service.http: RunPod-shaped endpoints, auth, validation, a real tiny render."""
import hashlib
import json
import logging
import os
import shutil
import threading
import time

import pytest
from fastapi.testclient import TestClient

from tour_worker.config import REPO_ROOT, Settings
from tour_worker.service.http import create_app
from tour_worker.service.jobs import JobStore

KEY = "test-key-123"
AUTH = {"Authorization": f"Bearer {KEY}"}
SEGMENT = {"job_type": "tour_segment", "images": [{"photo_id": "p", "uri": "s3://b/p.jpg"}],
           "output_prefix": "s3://b/s/"}
SIGNED = "https://bucket.s3.amazonaws.com/a.jpg?X-Amz-Credential=AKIA&X-Amz-Signature=SECRETSIG"


def _example(name):
    return json.load(open(os.path.join(REPO_ROOT, "examples", name)))["input"]


@pytest.fixture
def make_client():
    made = []

    def make(runner=None, settings=None, **store_kw):
        settings = settings or Settings(api_key=KEY)
        store = JobStore(runner=runner, **store_kw) if (runner or store_kw) else None
        client = TestClient(create_app(settings, store=store))
        client.__enter__()
        made.append(client)
        return client
    yield make
    for c in made:
        c.__exit__(None, None, None)


def _poll(client, job_id, timeout=60):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        body = client.get(f"/status/{job_id}", headers=AUTH).json()
        if body["status"] in ("COMPLETED", "FAILED", "CANCELLED"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish: {body}")


# ---- auth ----------------------------------------------------------------------------------------

def test_the_app_refuses_to_start_without_a_key():
    with pytest.raises(RuntimeError, match="WORKER_API_KEY"):
        create_app(Settings())
    create_app(Settings(allow_no_auth=True))                             # the local-dev opt-out


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": KEY},
                                     {"Authorization": f"Basic {KEY}"}])
def test_every_endpoint_but_health_needs_the_key(make_client, headers):
    c = make_client(runner=lambda inp: {"status": "completed"})
    for method, path in [("POST", "/run"), ("POST", "/runsync"), ("GET", "/status/x"), ("POST", "/cancel/x")]:
        r = c.request(method, path, headers=headers, json={"input": SEGMENT})
        assert r.status_code == 401, path
        assert r.json() == {"status": "failed", "error": {"code": "unauthorized",
                                                          "message": "send Authorization: Bearer <WORKER_API_KEY>"}}
    assert c.get("/health").status_code == 200


def test_no_auth_mode_accepts_requests(make_client):
    c = make_client(settings=Settings(allow_no_auth=True), runner=lambda inp: {"status": "completed"})
    assert c.post("/run", json={"input": SEGMENT}).status_code == 200


# ---- lifecycle -----------------------------------------------------------------------------------

def test_run_then_status_with_the_router_patched(make_client, monkeypatch):
    from tour_worker.jobs import router
    seen = []

    def fake(job):
        seen.append(job)
        return {"status": "completed", "results": {}, "scenes": {}, "warnings": [], "model": "m", "seconds": 0.1}
    monkeypatch.setattr(router, "run_job", fake)
    c = make_client(settings=Settings(api_key=KEY))                      # the default store and runner
    r = c.post("/run", headers=AUTH, json={"input": SEGMENT})
    assert r.status_code == 200
    job_id = r.json()["id"]
    assert r.json() == {"id": job_id, "status": "IN_QUEUE"} and len(job_id) == 32
    body = _poll(c, job_id)
    assert body["status"] == "COMPLETED" and set(body) == {"id", "status", "delayTime", "executionTime", "output"}
    assert body["output"]["model"] == "m" and seen == [SEGMENT]
    assert isinstance(body["delayTime"], int) and isinstance(body["executionTime"], int)


def test_a_failed_job(make_client):
    err = {"code": "image_fetch_failed", "message": "image_fetch_failed: s01"}
    c = make_client(runner=lambda inp: {"status": "failed", "error": err})
    body = _poll(c, c.post("/run", headers=AUTH, json={"input": SEGMENT}).json()["id"])
    assert body["status"] == "FAILED" and body["error"] == err and "output" not in body


def test_runsync_waits_and_times_out(make_client):
    gate = threading.Event()
    c = make_client(settings=Settings(api_key=KEY, runsync_timeout_s=0.3),
                    runner=lambda inp: gate.wait(10) and {"status": "completed"})
    t0 = time.monotonic()
    r = c.post("/runsync", headers=AUTH, json={"input": SEGMENT})
    assert r.status_code == 200 and r.json()["status"] in ("IN_QUEUE", "IN_PROGRESS")
    assert time.monotonic() - t0 < 5
    gate.set()
    assert _poll(c, r.json()["id"])["status"] == "COMPLETED"
    r = c.post("/runsync", headers=AUTH, json={"input": SEGMENT})             # quick job: the result
    assert r.json()["status"] == "COMPLETED" and r.json()["output"] == {"status": "completed"}


def test_cancel_a_queued_job(make_client):
    gate = threading.Event()
    c = make_client(runner=lambda inp: gate.wait(10) and {"status": "completed"}, concurrency=1)
    first = c.post("/run", headers=AUTH, json={"input": SEGMENT}).json()["id"]
    second = c.post("/run", headers=AUTH, json={"input": SEGMENT}).json()["id"]
    assert c.post(f"/cancel/{second}", headers=AUTH).json() == {"id": second, "status": "CANCELLED"}
    assert c.get("/health").json()["jobs"]["cancelled"] == 1
    gate.set()
    assert _poll(c, first)["status"] == "COMPLETED"
    assert c.get(f"/status/{second}", headers=AUTH).json()["status"] == "CANCELLED"
    r = c.post(f"/cancel/{first}", headers=AUTH)                         # finished: unchanged
    assert r.json() == {"id": first, "status": "COMPLETED"}


def test_unknown_jobs_are_404_in_the_failure_shape(make_client):
    c = make_client(runner=lambda inp: {})
    for r in (c.get("/status/nope", headers=AUTH), c.post("/cancel/nope", headers=AUTH)):
        assert r.status_code == 404 and r.json()["error"]["code"] == "job_not_found"
    assert c.get("/nowhere").json()["error"]["code"] == "not_found"


def test_health(make_client):
    c = make_client(runner=lambda inp: {"status": "completed"}, concurrency=2)
    _poll(c, c.post("/run", headers=AUTH, json={"input": SEGMENT}).json()["id"])
    h = c.get("/health").json()
    assert h["jobs"] == {"inQueue": 0, "inProgress": 0, "completed": 1, "failed": 0, "cancelled": 0}
    assert h["workers"] == {"running": 0, "idle": 2, "concurrency": 2}
    assert h["jobTypes"] == ["tour_render", "tour_segment", "tour_plan"]
    assert h["segmentModel"] == "shi-labs/oneformer_coco_swin_large@3a263017ca"
    assert "version" in h and "workerId" in h


# ---- validation ----------------------------------------------------------------------------------

def _bad_plan():
    req = _example("tour_render.json")
    req["plan"]["total_duration_s"] += 1                               # no longer matches plan_hash
    return req


@pytest.mark.parametrize("body, code", [
    ({"input": {"job_type": "virtual_tour"}}, "invalid_job_type"),
    ({"input": {"job_type": "tour_render", "plan": {}}}, "invalid_request"),   # no output_prefix
    ({"input": {**SEGMENT, "images": []}}, "invalid_request"),
    ({"input": {"job_type": "tour_plan", "listing_id": "L", "photos": []}}, "invalid_request"),
    ({"input": "a string"}, "invalid_request"),
    ({"input": None}, "invalid_request"),
    ({}, "invalid_request"),                                                  # no input
    ({"input": SEGMENT, "webhook": "ftp://x"}, "invalid_request"),
    (None, "invalid_plan"),                                                   # _bad_plan()
])
def test_requests_that_cannot_run_are_400_and_never_queued(make_client, body, code):
    calls = []
    c = make_client(runner=lambda inp: calls.append(inp) or {"status": "completed"})
    body = body if body is not None else {"input": _bad_plan()}
    for path in ("/run", "/runsync"):
        r = c.post(path, headers=AUTH, json=body)
        assert r.status_code == 400, r.text
        out = r.json()
        assert set(out) == {"status", "error"} and out["status"] == "failed"
        assert out["error"]["code"] == code and out["error"]["message"]
    assert calls == [] and c.get("/health").json()["jobs"]["inQueue"] == 0


def test_a_body_that_is_not_json_is_400_and_not_echoed(make_client):
    c = make_client(runner=lambda inp: {})
    r = c.post("/run", headers={**AUTH, "Content-Type": "application/json"}, content=b'{"input": ' + SIGNED.encode())
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_request"
    assert "SECRETSIG" not in r.text


def test_the_examples_are_accepted(make_client):
    c = make_client(runner=lambda inp: {"status": "completed"})
    for name in ("tour_render.json", "tour_segment.json", "tour_plan.json"):
        assert c.post("/run", headers=AUTH, json={"input": _example(name)}).status_code == 200


# ---- logs ----------------------------------------------------------------------------------------

def test_logs_never_hold_the_key_or_signed_urls(make_client, caplog):
    caplog.set_level(logging.DEBUG)
    msg = f"image_fetch_failed: could not fetch {SIGNED}"
    c = make_client(runner=lambda inp: {"status": "failed", "error": {"code": "image_fetch_failed", "message": msg}})
    req = {**_example("tour_render.json"), "job_id": "caller-1", "image_urls": {"a.jpg": SIGNED}}
    r = c.post("/run", headers=AUTH, json={"input": req, "webhook": "http://127.0.0.1:9/hook?token=HOOKTOKEN"})
    body = _poll(c, r.json()["id"])
    assert body["error"]["message"] == msg                # the caller gets its own error unchanged
    time.sleep(0.2)
    assert "image_fetch_failed" in caplog.text and r.json()["id"] in caplog.text
    for secret in ("SECRETSIG", "AKIA", "HOOKTOKEN", KEY):
        assert secret not in caplog.text


# ---- a real render -------------------------------------------------------------------------------

@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_a_real_tiny_render_through_the_service(make_client, monkeypatch, tmp_path):
    from PIL import Image
    monkeypatch.setenv("WORKER_STORAGE", "local")
    monkeypatch.setenv("IMAGES_DIR", str(tmp_path))
    Image.new("RGB", (1920, 1080), color="white").save(tmp_path / "img.jpg")
    cam = {"type": "ken_burns", "easing": "ease_in_out", "start": {"cx": 0.5, "cy": 0.5, "zoom": 1.1},
           "end": {"cx": 0.5, "cy": 0.5, "zoom": 1.0}}
    plan = {"plan_version": "1.2", "listing_id": "L1", "mode": "cinematic", "preset": "P1",
            "output": {"width": 320, "height": 180, "fps": 30, "codec": "h264"}, "total_duration_s": 1.0,
            "shots": [{"shot_id": "s1", "order": 1, "asset_id": "a", "image_uri": "s3://bucket/img.jpg",
                       "source": {"width": 1920, "height": 1080}, "room_key": "x", "room_type": "y",
                       "narrative_stage": "z", "role": "hero", "duration_s": 1.0, "camera": cam,
                       "transition_in": {"type": "cut", "duration_s": 0.0}}],
            "provenance": {"scoring_config_version": "1", "planner_config_version": "1"}}
    canon = json.dumps(plan, separators=(",", ":"), sort_keys=True, ensure_ascii=False)
    plan["provenance"]["plan_hash"] = hashlib.sha256(canon.encode()).hexdigest()
    c = make_client(settings=Settings(api_key=KEY))
    r = c.post("/run", headers=AUTH, json={"input": {"job_type": "tour_render", "plan": plan,
                                                     "output_prefix": "s3://bucket/out/"}})
    body = _poll(c, r.json()["id"], timeout=120)
    assert body["status"] == "COMPLETED", body
    out = body["output"]
    assert out["video_uri"] == "s3://bucket/out/master.mp4" and (out["width"], out["height"]) == (320, 180)
    assert "error" not in out and os.path.getsize(tmp_path / "master.mp4") > 0


# ---- a real plan ---------------------------------------------------------------------------------

def test_a_real_tour_plan_through_the_service(make_client, tmp_path):
    """tour_plan behind POST /runsync: photos from file:// URIs, the plan back as JSON."""
    import cv2
    import numpy as np
    from tour_worker.contract.jobs import TourPlanResult
    rooms = ["exterior", "living_room", "living_room", "kitchen", "kitchen", "bedroom"]
    photos = []
    for i, rt in enumerate(rooms):
        rng = np.random.default_rng(i)
        img = cv2.GaussianBlur((rng.random((360, 540, 3)) * 255).astype(np.uint8), (0, 0), 3)
        cv2.line(img, (int(rng.integers(0, 540)), 0), (int(rng.integers(0, 540)), 359), (20, 20, 20), 3)
        cv2.imwrite(str(tmp_path / f"p{i}.jpg"), img)
        photos.append({"photo_id": f"img_{i}", "uri": f"file://{tmp_path}/p{i}.jpg", "room_type": rt})
    c = make_client(settings=Settings(api_key=KEY))
    body = c.post("/runsync", headers=AUTH, json={"input": {
        "job_type": "tour_plan", "listing_id": "L1", "photos": photos}}).json()
    assert body["status"] == "COMPLETED", body
    TourPlanResult.model_validate(body["output"])
    assert body["output"]["plan"]["shots"]
