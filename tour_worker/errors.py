"""One failure shape for every job: {"status": "failed", "error": {"code": ..., "message": ...}}.

Code that fails a job raises WorkerError(code, message). It is a RuntimeError whose text is
"<code>: <message>", as the worker's errors always were, so existing `except RuntimeError` paths
and messages are unchanged. The job runners turn any exception into failure(...)."""
from __future__ import annotations


# Every code a job can fail with (README.md, "Jobs and their JSON"); anything else is "unknown"
CODES = frozenset({
    "invalid_job_type", "invalid_request", "invalid_plan", "unsupported_camera_type",
    "unsupported_easing", "image_fetch_failed", "planning_failed", "prewarp", "encode_failed",
    "verification_failed", "unknown",
})


class WorkerError(RuntimeError):
    def __init__(self, code: str, message: str = ""):
        self.code, self.message = code, message
        super().__init__(f"{code}: {message}" if message else code)


def failure(code: str, message: str) -> dict:
    return {"status": "failed", "error": {"code": code, "message": message}}


def from_exception(e: BaseException) -> dict:
    """The failure result for an exception: WorkerError keeps its code; another RuntimeError
    "code: text" gives code "code" when that is one of CODES; anything else is "unknown" (a
    library's own text never becomes a code). The message is the full text."""
    msg = str(e)
    if isinstance(e, WorkerError):
        return failure(e.code, msg)
    prefix = msg.split(":")[0].strip()
    if isinstance(e, RuntimeError) and prefix in CODES:
        return failure(prefix, msg)
    return failure("unknown", msg)
