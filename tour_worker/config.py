"""Worker settings: read from the environment ONCE (get_settings()), nowhere else.

Every setting, its environment variable and default (also in README.md):

  WORKER_STORAGE       ""         "local": dev mode - every image / mask / output URI, s3:// and
                                  https:// included, is served from IMAGES_DIR by basename
                                  (tour_render and tour_plan; tour_segment always uses S3 / HTTPS).
                                  Anything else: by URI scheme (s3://, https://, file:// or a path)
  IMAGES_DIR           "/tmp"     the folder WORKER_STORAGE=local reads from and writes to
  PREWARP_WORKERS      (auto)     processes drawing + encoding frames; auto = min(8, CPUs - 1),
                                  capped by available memory (600 MB each)
  MODELS_DIR           <repo>/models   where model weights live (one folder per model)
  SEGMENT_MODEL        "coco"     OneFormer checkpoint for tour_segment: "coco" or "ade20k"
                                  (tour_worker.vision.segment.MODELS)
  ONEFORMER_DIR        (unset)    that checkpoint's folder; default MODELS_DIR/<its dir name>
  COCO_FIRE_SEARCH     "0"        "1": look for lit fires by colour with the COCO model (off: it
                                  failed validation, 4/11 fires found)
  FIREPLACE_MODEL      "ade20k"   a second OneFormer pass that finds fireplaces (COCO has no such
                                  class); lit fires are looked for inside them. "" = off
  FIREPLACE_DIR        (unset)    that checkpoint's folder; default MODELS_DIR/<its dir name>

The HTTP service (python -m tour_worker.service; handler.py, the RunPod entry, ignores these):

  WORKER_API_KEY       (unset)    the key callers send as "Authorization: Bearer <key>" (every
                                  endpoint but /health). Unset: the service refuses to start ...
  WORKER_ALLOW_NO_AUTH "0"        ... unless this is "1" (local development only: no auth at all)
  JOB_CONCURRENCY      1          jobs run at once (a render already uses every CPU it is given)
  JOBS_DIR             (unset)    folder to keep job records in (one JSON per job), so statuses
                                  survive a restart; unset: in memory only
  JOB_TTL_S            86400      how long a finished job's status is kept (seconds)
  RUNSYNC_TIMEOUT_S    300        how long POST /runsync waits before returning the job's status
  HOST                 127.0.0.1  address the service listens on (the Docker images set 0.0.0.0)
  PORT                 8000       its port

AWS credentials and region are not settings of the worker: boto3 reads them itself
(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION, or an instance role).
"""
from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field
from typing import Mapping

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass(frozen=True)
class Settings:
    storage: str = "remote"                 # "remote" | "local"
    images_dir: str = "/tmp"
    render_workers: int | None = None       # None: automatic
    models_dir: str = os.path.join(REPO_ROOT, "models")
    segment_model: str = "coco"
    oneformer_dir: str | None = None
    coco_fire_search: bool = False
    fireplace_model: str = "ade20k"         # "" = no fireplace pass
    fireplace_dir: str | None = None
    # the HTTP service (tour_worker.service)
    api_key: str | None = field(default=None, repr=False)     # never in a repr / log
    allow_no_auth: bool = False
    job_concurrency: int = 1
    jobs_dir: str | None = None
    job_ttl_s: float = 86400.0
    runsync_timeout_s: float = 300.0
    host: str = "127.0.0.1"
    port: int = 8000

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env
        workers = env.get("PREWARP_WORKERS", "").strip()
        return cls(
            storage="local" if env.get("WORKER_STORAGE") == "local" else "remote",
            images_dir=env.get("IMAGES_DIR", "/tmp"),
            render_workers=max(1, int(workers)) if workers else None,
            models_dir=env.get("MODELS_DIR") or os.path.join(REPO_ROOT, "models"),
            segment_model=env.get("SEGMENT_MODEL", "coco"),
            oneformer_dir=env.get("ONEFORMER_DIR") or None,
            coco_fire_search=env.get("COCO_FIRE_SEARCH", "") == "1",
            fireplace_model=env.get("FIREPLACE_MODEL", "ade20k").strip(),
            fireplace_dir=env.get("FIREPLACE_DIR") or None,
            api_key=env.get("WORKER_API_KEY") or None,
            allow_no_auth=env.get("WORKER_ALLOW_NO_AUTH", "") == "1",
            job_concurrency=max(1, int(env.get("JOB_CONCURRENCY") or 1)),
            jobs_dir=env.get("JOBS_DIR") or None,
            job_ttl_s=max(0.0, float(env.get("JOB_TTL_S") or 86400)),
            runsync_timeout_s=max(0.0, float(env.get("RUNSYNC_TIMEOUT_S") or 300)),
            host=env.get("HOST") or "127.0.0.1",
            port=int(env.get("PORT") or 8000),
        )

    def oneformer_path(self) -> str:
        """The folder of the configured OneFormer checkpoint."""
        from tour_worker.vision.segment import MODELS
        return self.oneformer_dir or os.path.join(self.models_dir, MODELS[self.segment_model]["dir"])


    def fireplace_path(self) -> str | None:
        """The folder of the fireplace pass's checkpoint (None: the pass is off)."""
        if not self.fireplace_model:
            return None
        from tour_worker.vision.segment import MODELS
        return self.fireplace_dir or os.path.join(self.models_dir, MODELS[self.fireplace_model]["dir"])


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The process's settings, read from the environment on first use."""
    return Settings.from_env()


def reset_settings() -> None:
    """Forget the cached settings (tests that change the environment)."""
    get_settings.cache_clear()
