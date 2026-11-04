"""tour_worker.service.jobs: lifecycle, cancel, persistence, restart, TTL, webhook."""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from tour_worker.service.jobs import (CANCELLED, COMPLETED, FAILED, IN_PROGRESS, IN_QUEUE, JobStore,
                                      post_webhook, split_result)

RENDER_OK = {"status": "completed", "video_uri": "s3://b/o/master.mp4", "error": None}
RENDER_FAILED = {"status": "failed", "error": {"code": "image_fetch_failed", "message": "image_fetch_failed: x"}}


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def _wait_for(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def stores():
    made = []

    def make(**kw):
        s = JobStore(**kw)
        made.append(s)
        return s
    yield make
    for s in made:
        s.shutdown(wait=False)


def test_split_result_reads_a_result_the_way_runpod_does():
    assert split_result(RENDER_OK) == (COMPLETED, {"status": "completed", "video_uri": "s3://b/o/master.mp4"}, None)
    assert split_result(RENDER_FAILED) == (FAILED, None, RENDER_FAILED["error"])
    assert split_result({"error": "boom"}) == (FAILED, None, {"code": "unknown", "message": "boom"})


def test_lifecycle_with_runpod_fields(stores):
    clock = Clock()
    gate = threading.Event()

    def runner(inp):
        clock.t += 2.5                      # the job "takes" 2.5 s
        gate.wait(5)
        return RENDER_OK
    store = stores(runner=runner, clock=clock)
    job = store.submit({"job_type": "tour_render"})
    assert _wait_for(lambda: store.status(job.id)["status"] == IN_PROGRESS)
    gate.set()
    body = store.wait(job.id, 5)
    assert body == {"id": job.id, "status": COMPLETED, "delayTime": 0, "executionTime": 2500,
                    "output": {"status": "completed", "video_uri": "s3://b/o/master.mp4"}}
    assert store.get(job.id).input is None                     # the request is dropped once it ran


def test_a_failed_job_and_a_crashing_runner(stores):
    store = stores(runner=lambda inp: RENDER_FAILED)
    body = store.wait(store.submit({}).id, 5)
    assert body["status"] == FAILED and body["error"] == RENDER_FAILED["error"] and "output" not in body

    def crash(inp):
        raise ValueError("bug")
    store = stores(runner=crash)
    body = store.wait(store.submit({}).id, 5)
    assert body["status"] == FAILED and body["error"] == {"code": "unknown", "message": "bug"}


def test_only_a_queued_job_can_be_cancelled(stores):
    gate, calls = threading.Event(), []

    def runner(inp):
        calls.append(inp)
        gate.wait(5)
        return RENDER_OK
    store = stores(runner=runner, concurrency=1)
    first, second = store.submit({"n": 1}), store.submit({"n": 2})
    assert _wait_for(lambda: store.status(first.id)["status"] == IN_PROGRESS)
    assert store.status(second.id)["status"] == IN_QUEUE
    assert store.cancel(second.id) == {"id": second.id, "status": CANCELLED}
    assert store.cancel(first.id) == {"id": first.id, "status": IN_PROGRESS}    # running: left alone
    gate.set()
    assert store.wait(first.id, 5)["status"] == COMPLETED
    time.sleep(0.1)
    assert calls == [{"n": 1}] and store.status(second.id)["status"] == CANCELLED
    assert store.cancel("nope") is None


def test_wait_returns_the_current_status_on_timeout(stores):
    gate = threading.Event()
    store = stores(runner=lambda inp: gate.wait(5) and RENDER_OK)
    job = store.submit({})
    t0 = time.monotonic()
    assert store.wait(job.id, 0.2)["status"] in (IN_QUEUE, IN_PROGRESS)
    assert time.monotonic() - t0 < 2
    gate.set()
    assert store.wait(job.id, 5)["status"] == COMPLETED


def test_concurrency(stores):
    gate, running = threading.Event(), []
    store = stores(runner=lambda inp: running.append(1) or gate.wait(5) and RENDER_OK, concurrency=2)
    jobs = [store.submit({}) for _ in range(3)]
    assert _wait_for(lambda: store.counts()[IN_PROGRESS] == 2)
    assert store.counts()[IN_QUEUE] == 1
    gate.set()
    assert all(store.wait(j.id, 5)["status"] == COMPLETED for j in jobs)


def test_records_survive_a_restart_without_the_request(tmp_path, stores):
    secret = "https://bucket.s3.amazonaws.com/a.jpg?X-Amz-Signature=SECRET"
    store = stores(runner=lambda inp: RENDER_OK, jobs_dir=str(tmp_path))
    job = store.submit({"job_type": "tour_render", "image_urls": {"a.jpg": secret}}, webhook="https://h/?t=TOKEN")
    done = store.wait(job.id, 5)
    on_disk = (tmp_path / f"{job.id}.json").read_text()
    assert "SECRET" not in on_disk and "TOKEN" not in on_disk
    again = stores(runner=lambda inp: RENDER_OK, jobs_dir=str(tmp_path))
    assert again.status(job.id) == done


def test_a_restart_fails_the_jobs_it_had_not_finished(tmp_path, stores):
    gate = threading.Event()
    store = stores(runner=lambda inp: gate.wait(5) and RENDER_OK, jobs_dir=str(tmp_path), concurrency=1)
    running, queued = store.submit({}), store.submit({})
    assert _wait_for(lambda: store.status(running.id)["status"] == IN_PROGRESS)
    (tmp_path / "junk.json").write_text("{not json")                    # skipped, not fatal
    restarted = stores(runner=lambda inp: RENDER_OK, jobs_dir=str(tmp_path))
    for job_id in (running.id, queued.id):
        body = restarted.status(job_id)
        assert body["status"] == FAILED and body["error"]["code"] == "worker_restarted"
        assert json.loads((tmp_path / f"{job_id}.json").read_text())["status"] == FAILED
    gate.set()


def test_finished_jobs_are_forgotten_after_the_ttl(tmp_path, stores):
    clock = Clock()
    gate = threading.Event()
    store = stores(runner=lambda inp: gate.wait(5) and RENDER_OK, jobs_dir=str(tmp_path), ttl_s=60, clock=clock)
    done, busy = store.submit({}), store.submit({})
    gate.set()
    store.wait(done.id, 5), store.wait(busy.id, 5)
    clock.t += 59
    assert store.cleanup() == 0
    clock.t += 2
    assert store.cleanup() == 2
    assert store.status(done.id) is None and not os.listdir(tmp_path)


def test_unfinished_jobs_are_never_expired(stores):
    clock, gate = Clock(), threading.Event()
    store = stores(runner=lambda inp: gate.wait(5) and RENDER_OK, ttl_s=1, clock=clock)
    job = store.submit({})
    clock.t += 1000
    assert store.cleanup() == 0 and store.status(job.id) is not None
    gate.set()


class _Hook(BaseHTTPRequestHandler):
    received: list = []
    fail_first = 0
    answer = None              # a fixed (status, headers) instead of 200

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if _Hook.answer:
            _Hook.received.append(body)
            self.send_response(_Hook.answer[0])
            for k, v in _Hook.answer[1].items():
                self.send_header(k, v)
        elif _Hook.fail_first > 0:
            _Hook.fail_first -= 1
            self.send_response(500)
        else:
            _Hook.received.append(body)
            self.send_response(200)
        self.end_headers()

    def log_message(self, *a):
        pass


@pytest.fixture
def hook_server():
    _Hook.received, _Hook.fail_first, _Hook.answer = [], 0, None
    srv = HTTPServer(("127.0.0.1", 0), _Hook)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/hook?token=abc"
    srv.shutdown()


def test_the_webhook_gets_the_final_status(hook_server, stores):
    store = stores(runner=lambda inp: RENDER_OK)
    job = store.submit({}, webhook=hook_server)
    body = store.wait(job.id, 5)
    assert _wait_for(lambda: _Hook.received)
    assert _Hook.received == [body]


def test_the_webhook_is_retried_and_never_fails_the_job(hook_server, stores):
    _Hook.fail_first = 1
    assert post_webhook(hook_server, {"id": "x"}, backoff_s=0.01) is True
    assert _Hook.received == [{"id": "x"}]
    assert post_webhook("http://127.0.0.1:9/nothing", {"id": "y"}, attempts=2, timeout_s=0.5, backoff_s=0.01) is False
    store = stores(runner=lambda inp: RENDER_OK)
    job = store.submit({}, webhook="http://127.0.0.1:9/nothing")
    assert store.wait(job.id, 5)["status"] == COMPLETED


def test_a_4xx_is_not_retried_and_a_redirect_is_not_followed(hook_server):
    _Hook.answer = (404, {})
    assert post_webhook(hook_server, {"id": "x"}, backoff_s=0.01) is False
    assert len(_Hook.received) == 1                              # one try: a 4xx is final
    _Hook.received, _Hook.answer = [], (302, {"Location": "http://169.254.169.254/latest/meta-data/"})
    assert post_webhook(hook_server, {"id": "y"}, backoff_s=0.01) is False
    assert len(_Hook.received) == 1


@pytest.mark.parametrize("url", ["http://169.254.169.254/latest/meta-data/", "http://[fe80::1]/x",
                                 "http://0.0.0.0:9/x"])
def test_a_webhook_never_reaches_link_local_or_reserved_addresses(url, monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", lambda *a, **k: pytest.fail("posted"))
    assert post_webhook(url, {"id": "x"}, backoff_s=0.01) is False


def test_webhook_hosts_limits_where_a_webhook_may_point():
    from tour_worker.service.jobs import webhook_refusal
    assert webhook_refusal("HTTPS://Backend.example.com/hook") is None          # scheme: any case
    assert webhook_refusal("ftp://x/") and webhook_refusal("https:///no-host")
    hosts = ("backend.example.com",)
    assert webhook_refusal("https://backend.example.com/h", hosts) is None
    assert "WEBHOOK_HOSTS" in webhook_refusal("https://evil.example.net/h", hosts)


def test_a_result_that_is_not_json_fails_the_job_instead_of_hanging_it(stores, tmp_path):
    """Before: json.dump raised inside the lock - waiters slept out their timeout, no webhook, and
    the record stayed IN_PROGRESS."""
    import numpy as np
    store = stores(runner=lambda inp: {"status": "completed", "duration_s": np.float32(1.5), "s": {1}},
                   jobs_dir=str(tmp_path))
    nan = stores(runner=lambda inp: {"status": "completed", "duration_s": float("nan")})
    assert nan.wait(nan.submit({}).id, 5)["error"]["code"] == "unknown"     # /status sends strict JSON
    t0 = time.monotonic()
    body = store.wait(store.submit({}).id, 5)
    assert time.monotonic() - t0 < 2
    assert body["status"] == FAILED and body["error"]["code"] == "unknown"
    assert "not JSON" in body["error"]["message"]
    assert json.load(open(next(tmp_path.glob("*.json"))))["status"] == FAILED
    assert not list(tmp_path.glob("*.tmp"))
