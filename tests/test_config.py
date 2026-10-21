"""tour_worker.config: one Settings object from the environment."""
import os

from tour_worker.config import REPO_ROOT, Settings, get_settings


def test_defaults():
    s = Settings.from_env({})
    assert s == Settings(storage="remote", images_dir="/tmp", render_workers=None,
                         models_dir=os.path.join(REPO_ROOT, "models"), segment_model="coco",
                         oneformer_dir=None, coco_fire_search=False)
    assert s.oneformer_path() == os.path.join(REPO_ROOT, "models", "oneformer_coco_swin_large")


def test_from_env():
    s = Settings.from_env({"WORKER_STORAGE": "local", "IMAGES_DIR": "/imgs", "PREWARP_WORKERS": "3",
                           "MODELS_DIR": "/m", "SEGMENT_MODEL": "ade20k", "COCO_FIRE_SEARCH": "1"})
    assert (s.storage, s.images_dir, s.render_workers, s.segment_model, s.coco_fire_search) == \
        ("local", "/imgs", 3, "ade20k", True)
    assert s.oneformer_path() == "/m/oneformer_ade20k_swin_large"
    assert Settings.from_env({"ONEFORMER_DIR": "/w"}).oneformer_path() == "/w"


def test_read_once(monkeypatch):
    monkeypatch.setenv("IMAGES_DIR", "/first")
    assert get_settings().images_dir == "/first"
    monkeypatch.setenv("IMAGES_DIR", "/second")
    assert get_settings().images_dir == "/first"          # cached: env is read once per process
