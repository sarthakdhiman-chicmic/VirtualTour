"""Worker settings: read from the environment ONCE (get_settings()), nowhere else.

Every setting, its environment variable and default (also in README.md):

  WORKER_STORAGE       ""         "local": dev mode - every image / mask / output URI, s3:// and
                                  https:// included, is served from IMAGES_DIR by basename
                                  (tour_render only; tour_segment always uses S3 / HTTPS).
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

AWS credentials and region are not settings of the worker: boto3 reads them itself
(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION, or an instance role).
"""
from __future__ import annotations

import functools
import os
from dataclasses import dataclass
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
        )

    def oneformer_path(self) -> str:
        """The folder of the configured OneFormer checkpoint."""
        from tour_worker.vision.segment import MODELS
        return self.oneformer_dir or os.path.join(self.models_dir, MODELS[self.segment_model]["dir"])


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The process's settings, read from the environment on first use."""
    return Settings.from_env()


def reset_settings() -> None:
    """Forget the cached settings (tests that change the environment)."""
    get_settings.cache_clear()
