"""tour_worker.config: one Settings object from the environment."""
import os

from tour_worker.config import REPO_ROOT, Settings, get_settings


def test_defaults():
    s = Settings.from_env({})
    assert s == Settings(storage="remote", images_dir="/tmp", render_workers=None,
                         models_dir=os.path.join(REPO_ROOT, "models"), segment_model="oneformer_coco",
                         segment_dir=None)
    assert s.segment_path() == os.path.join(REPO_ROOT, "models", "oneformer_coco_swin_large")


def test_from_env():
    s = Settings.from_env({"WORKER_STORAGE": "local", "IMAGES_DIR": "/imgs", "PREWARP_WORKERS": "3",
                           "MODELS_DIR": "/m", "SEGMENT_MODEL": "oneformer_coco"})
    assert (s.storage, s.images_dir, s.render_workers, s.segment_model) == ("local", "/imgs", 3, "oneformer_coco")
    assert s.segment_path() == "/m/oneformer_coco_swin_large"
    assert Settings.from_env({"SEGMENT_DIR": "/w"}).segment_path() == "/w"


def test_read_once(monkeypatch):
    monkeypatch.setenv("IMAGES_DIR", "/first")
    assert get_settings().images_dir == "/first"
    monkeypatch.setenv("IMAGES_DIR", "/second")
    assert get_settings().images_dir == "/first"          # cached: env is read once per process


def test_service_settings():
    s = Settings.from_env({})
    assert (s.api_key, s.allow_no_auth, s.job_concurrency, s.jobs_dir, s.job_ttl_s, s.runsync_timeout_s,
            s.host, s.port) == (None, False, 1, None, 86400.0, 300.0, "127.0.0.1", 8000)
    s = Settings.from_env({"WORKER_API_KEY": "k3y", "WORKER_ALLOW_NO_AUTH": "1", "JOB_CONCURRENCY": "2",
                           "JOBS_DIR": "/j", "JOB_TTL_S": "60", "RUNSYNC_TIMEOUT_S": "5", "HOST": "0.0.0.0",
                           "PORT": "9000"})
    assert (s.api_key, s.allow_no_auth, s.job_concurrency, s.jobs_dir, s.job_ttl_s, s.runsync_timeout_s,
            s.host, s.port) == ("k3y", True, 2, "/j", 60.0, 5.0, "0.0.0.0", 9000)
    assert "k3y" not in repr(s)                              # the key never reaches a log
