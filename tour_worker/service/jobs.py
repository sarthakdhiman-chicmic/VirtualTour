"""Job store and executor of the HTTP service: queue a request, run it on a thread, keep its status.

States are RunPod's: IN_QUEUE -> IN_PROGRESS -> COMPLETED | FAILED, or IN_QUEUE -> CANCELLED.
A job runs `runner(input)` (default tour_worker.jobs.router.run_job) on one of
`JOB_CONCURRENCY` threads. Its result is split the way RunPod's SDK splits a handler's dict: the
"error" key is taken out; if it is set the job is FAILED with that error ({"code", "message"}),
otherwise COMPLETED with the rest as `output`.

Only a queued job can be cancelled: a running render is many ffmpeg / worker processes with
uploads at the end, and there is no safe point to stop it, so cancel leaves it running.

Records are kept in memory and, with JOBS_DIR, as one JSON file per job (written atomically).
Neither keeps the request (it holds presigned URLs) or the webhook. On start, a job a previous
process left IN_QUEUE or IN_PROGRESS is FAILED with code `worker_restarted`: its request is gone
and the caller resubmits. Finished jobs are forgotten JOB_TTL_S after they finished."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from tour_worker.errors import failure, from_exception
from tour_worker.service.redact import redact_text, redact_url

log = logging.getLogger("tour_worker.service")

IN_QUEUE, IN_PROGRESS = "IN_QUEUE", "IN_PROGRESS"
COMPLETED, FAILED, CANCELLED = "COMPLETED", "FAILED", "CANCELLED"
TERMINAL = (COMPLETED, FAILED, CANCELLED)
STATES = (IN_QUEUE, IN_PROGRESS) + TERMINAL

WEBHOOK_TIMEOUT_S = 10.0
WEBHOOK_ATTEMPTS = 3


def _default_runner(job_input: dict) -> dict:
    from tour_worker.jobs import router          # looked up per call: tests patch router.run_job
    return router.run_job(job_input)


@dataclass
class Job:
    id: str
    status: str
    created_at: float
    job_type: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    output: Any = None
    error: dict | None = None
    input: dict | None = field(default=None, repr=False)    # memory only, dropped once it ran
    webhook: str | None = field(default=None, repr=False)   # memory only

    def status_body(self) -> dict:
        """GET /status/{id}: RunPod's fields. delayTime / executionTime are milliseconds."""
        body: dict[str, Any] = {"id": self.id, "status": self.status}
        if self.started_at is not None:
            body["delayTime"] = int(round((self.started_at - self.created_at) * 1000))
            if self.finished_at is not None:
                body["executionTime"] = int(round((self.finished_at - self.started_at) * 1000))
        if self.status == COMPLETED:
            body["output"] = self.output
        if self.error is not None:
            body["error"] = self.error
        return body

    def record(self) -> dict:
        """What JOBS_DIR keeps: everything but the request and the webhook."""
        return {"id": self.id, "status": self.status, "job_type": self.job_type,
                "created_at": self.created_at, "started_at": self.started_at,
                "finished_at": self.finished_at, "output": self.output, "error": self.error}

    @classmethod
    def from_record(cls, r: dict) -> "Job":
        return cls(id=r["id"], status=r["status"], created_at=float(r["created_at"]), job_type=r.get("job_type"),
                   started_at=r.get("started_at"), finished_at=r.get("finished_at"),
                   output=r.get("output"), error=r.get("error"))


def split_result(result: Any) -> tuple[str, Any, dict | None]:
    """(status, output, error) of a runner's result, as RunPod's SDK reads a handler's dict."""
    if not isinstance(result, dict):
        return COMPLETED, result, None
    output = dict(result)
    error = output.pop("error", None)
    if error:
        if not isinstance(error, dict):
            error = {"code": "unknown", "message": str(error)}
        return FAILED, None, error
    return COMPLETED, output, None


class JobStore:
    def __init__(self, runner: Callable[[dict], dict] | None = None, concurrency: int = 1,
                 jobs_dir: str | None = None, ttl_s: float = 86400.0,
                 clock: Callable[[], float] = time.time):
        self.runner = runner or _default_runner
        self.concurrency = max(1, int(concurrency))
        self.jobs_dir = jobs_dir
        self.ttl_s = ttl_s
        self.clock = clock
        self._jobs: dict[str, Job] = {}
        self._cond = threading.Condition()
        self._pool = ThreadPoolExecutor(self.concurrency, thread_name_prefix="tour-job")
        self._stop = threading.Event()
        self._janitor: threading.Thread | None = None
        if jobs_dir:
            os.makedirs(jobs_dir, exist_ok=True)
            self._load()

    # ---- persistence -------------------------------------------------------------------------

    def _path(self, job_id: str) -> str:
        return os.path.join(self.jobs_dir, f"{job_id}.json")

    def _persist(self, job: Job) -> None:
        if not self.jobs_dir:
            return
        tmp = self._path(job.id) + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(job.record(), f)
            os.replace(tmp, self._path(job.id))
        except OSError as e:                     # a full disk must not fail the job
            log.error("job %s: could not save its record: %s", job.id, e)

    def _unpersist(self, job_id: str) -> None:
        if self.jobs_dir:
            try:
                os.remove(self._path(job_id))
            except FileNotFoundError:
                pass

    def _load(self) -> None:
        now = self.clock()
        for name in sorted(os.listdir(self.jobs_dir)):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(self.jobs_dir, name)) as f:
                    job = Job.from_record(json.load(f))
            except (OSError, ValueError, KeyError, TypeError) as e:
                log.warning("skipping unreadable job record %s: %s", name, e)
                continue
            if job.status not in TERMINAL:     # this process never ran it: its request is gone
                was = job.status
                job.status, job.finished_at, job.output = FAILED, now, None
                job.error = failure("worker_restarted",
                                    f"the worker restarted while the job was {was}; submit it again")["error"]
                self._persist(job)
            self._jobs[job.id] = job
        self.cleanup()

    # ---- lifecycle ---------------------------------------------------------------------------

    def start_janitor(self, interval_s: float | None = None) -> None:
        """Forget expired jobs in the background (every minute, or ttl/2 when shorter)."""
        interval = interval_s or max(1.0, min(60.0, self.ttl_s / 2))

        def loop():
            while not self._stop.wait(interval):
                self.cleanup()
        self._janitor = threading.Thread(target=loop, name="tour-job-janitor", daemon=True)
        self._janitor.start()

    def shutdown(self, wait: bool = False) -> None:
        self._stop.set()
        self._pool.shutdown(wait=wait, cancel_futures=True)

    # ---- jobs --------------------------------------------------------------------------------

    def submit(self, job_input: dict, webhook: str | None = None) -> Job:
        job = Job(id=uuid.uuid4().hex, status=IN_QUEUE, created_at=self.clock(),
                  job_type=job_input.get("job_type") if isinstance(job_input, dict) else None,
                  input=job_input, webhook=webhook)
        with self._cond:
            self._jobs[job.id] = job
            self._persist(job)
        log.info("job %s queued: %s (caller job_id %s)", job.id, job.job_type,
                 redact_text(str(job_input.get("job_id"))) if isinstance(job_input, dict) else None)
        self._pool.submit(self._execute, job.id)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._cond:
            return self._jobs.get(job_id)

    def status(self, job_id: str) -> dict | None:
        with self._cond:
            job = self._jobs.get(job_id)
            return job.status_body() if job else None

    def wait(self, job_id: str, timeout_s: float) -> dict | None:
        """The job's status once it is finished, or after timeout_s, whichever comes first."""
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while True:
                job = self._jobs.get(job_id)
                if job is None or job.status in TERMINAL:
                    return job.status_body() if job else None
                left = deadline - time.monotonic()
                if left <= 0:
                    return job.status_body()
                self._cond.wait(left)

    def cancel(self, job_id: str) -> dict | None:
        """Cancel a queued job. Anything else is left as it is; the body says its status."""
        with self._cond:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.status == IN_QUEUE:
                job.status, job.finished_at, job.input = CANCELLED, self.clock(), None
                self._persist(job)
                self._cond.notify_all()
                log.info("job %s cancelled", job.id)
                webhook = job.webhook
            else:
                webhook = None
            body = job.status_body()
        if webhook:
            self._notify(webhook, body)
        return {"id": body["id"], "status": body["status"]}

    def counts(self) -> dict[str, int]:
        with self._cond:
            c = {s: 0 for s in STATES}
            for job in self._jobs.values():
                c[job.status] += 1
            return c

    def cleanup(self) -> int:
        """Forget finished jobs older than the TTL. Returns how many went."""
        cutoff = self.clock() - self.ttl_s
        with self._cond:
            old = [j.id for j in self._jobs.values()
                   if j.status in TERMINAL and j.finished_at is not None and j.finished_at < cutoff]
            for job_id in old:
                del self._jobs[job_id]
                self._unpersist(job_id)
        if old:
            log.info("forgot %d finished job(s) past the %ss TTL", len(old), self.ttl_s)
        return len(old)

    def _execute(self, job_id: str) -> None:
        with self._cond:
            job = self._jobs.get(job_id)
            if job is None or job.status != IN_QUEUE:          # cancelled (or forgotten) meanwhile
                return
            job.status, job.started_at = IN_PROGRESS, self.clock()
            job_input = job.input
            self._persist(job)
            self._cond.notify_all()
        log.info("job %s started", job_id)
        try:
            result = self.runner(job_input)
        except BaseException as e:      # the router returns failures; this is a bug or an abort
            result = from_exception(e)
        status, output, error = split_result(result)
        with self._cond:
            job.status, job.output, job.error = status, output, error
            job.finished_at, job.input = self.clock(), None
            self._persist(job)
            self._cond.notify_all()
            body, webhook = job.status_body(), job.webhook
        if error:
            log.warning("job %s failed: %s: %s", job_id, error.get("code"),
                        redact_text(error.get("message", "")))
        else:
            log.info("job %s completed in %.1fs", job_id, job.finished_at - job.started_at)
        if webhook:
            self._notify(webhook, body)

    def _notify(self, url: str, body: dict) -> None:
        """POST the job's final status to its webhook, on a thread: best effort, a few tries."""
        threading.Thread(target=post_webhook, args=(url, body), name="tour-job-webhook", daemon=True).start()


def post_webhook(url: str, body: dict, attempts: int = WEBHOOK_ATTEMPTS,
                 timeout_s: float = WEBHOOK_TIMEOUT_S, backoff_s: float = 1.0) -> bool:
    """POST body as JSON to url; True once a 2xx comes back. Never raises."""
    data = json.dumps(body).encode()
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, data=data, method="POST",
                                         headers={"Content-Type": "application/json",
                                                  "User-Agent": "tour-worker"})
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                if 200 <= resp.status < 300:
                    return True
                err = f"HTTP {resp.status}"
        except Exception as e:                      # noqa: BLE001 - best effort by design
            err = redact_text(str(e))
        log.warning("job %s: webhook %s attempt %d/%d failed: %s", body.get("id"), redact_url(url),
                    attempt, attempts, err)
        if attempt < attempts:
            time.sleep(backoff_s * attempt)
    return False
