"""tour_worker.jobs.segment: tour_segment writes masks and returns JSON."""
import os
import tempfile

import cv2
import numpy as np

from tests.perceptual_fixtures import _photo


def test_run_segment_writes_masks_and_returns_json_only():
    from tour_worker.jobs.segment import run_segment

    class FakeStorage:
        def __init__(self):
            self.put_calls = []

        def get(self, uri, dest):
            cv2.imwrite(dest + ".png", _photo())
            os.replace(dest + ".png", dest)

        def put(self, path, uri):
            self.put_calls.append(uri)

    def fake_segmenter(img):
        m = np.zeros(img.shape[:2], np.uint8)
        m[50:300, 200:400] = 255
        return [{"object_id": "seg_000", "class_name": "curtain", "confidence": 0.99,
                 "bbox": [200, 50, 400, 300], "mask": m, "space": [img.shape[1], img.shape[0]]}]

    st = FakeStorage()
    with tempfile.TemporaryDirectory() as td:
        res = run_segment({"images": [{"photo_id": "img_0", "uri": "s3://b/a.jpg"}],
                           "output_prefix": "s3://b/seg/job1"}, st, td, segmenter=fake_segmenter)
    assert res["status"] == "completed"
    r = res["results"]["img_0"][0]
    assert "mask" not in r and r["mask_url"] == "s3://b/seg/job1/img_0/seg_000.png"
    assert st.put_calls == ["s3://b/seg/job1/img_0/seg_000.png"]
    from tour_worker.contract.jobs import TourSegmentResult
    TourSegmentResult.model_validate(res)
    assert res["model"] == "shi-labs/oneformer_coco_swin_large@3a263017ca"


def test_the_fireplace_pass_adds_lit_fires_and_never_fails_the_photo():
    """2026-10-22: COCO has no fireplace class; a second OneFormer (ADE20K) finds fireplaces and
    the lit fires inside them are added to the photo's regions."""
    from tour_worker.jobs.segment import run_segment

    class FakeStorage:
        def get(self, uri, dest):
            cv2.imwrite(dest + ".png", _photo())
            os.replace(dest + ".png", dest)

        def put(self, path, uri):
            pass

    def coco(img):
        m = np.zeros(img.shape[:2], np.uint8)
        m[50:300, 200:400] = 255
        return [{"object_id": "seg_000", "class_name": "curtain", "confidence": 0.99,
                 "bbox": [200, 50, 400, 300], "mask": m, "space": [img.shape[1], img.shape[0]]}]

    class FakeFireplaces:
        def __init__(self, fail=False):
            self.fail = fail

        def fires(self, img):
            if self.fail:
                raise RuntimeError("model down")
            m = np.zeros(img.shape[:2], np.uint8)
            m[300:360, 420:500] = 255
            return [{"object_id": "fire_000", "class_name": "fire", "confidence": 0.98,
                     "bbox": [420, 300, 500, 360], "mask": m, "space": [img.shape[1], img.shape[0]]}]

    req = {"images": [{"photo_id": "img_0", "uri": "s3://b/a.jpg"}], "output_prefix": "s3://b/seg/j/"}
    with tempfile.TemporaryDirectory() as td:
        res = run_segment(req, FakeStorage(), td, segmenter=coco, fireplace_segmenter=FakeFireplaces())
    assert sorted(r["class_name"] for r in res["results"]["img_0"]) == ["curtain", "fire"]
    with tempfile.TemporaryDirectory() as td:
        res = run_segment(req, FakeStorage(), td, segmenter=coco, fireplace_segmenter=FakeFireplaces(True))
    assert [r["class_name"] for r in res["results"]["img_0"]] == ["curtain"]
    assert any(w.startswith("fireplace_pass_failed:img_0") for w in res["warnings"])


def test_fireplace_pass_setting():
    from tour_worker.config import Settings
    assert Settings.from_env({}).fireplace_model == "ade20k"
    assert Settings.from_env({"FIREPLACE_MODEL": ""}).fireplace_path() is None


def test_a_photo_the_caller_holds_at_this_version_is_not_segmented_again():
    """The backend caches results per photo (by its sha256) under segment_version: a photo sent
    with the current known_version comes back in `cached`, untouched; a stale version is
    segmented again. With nothing to do the models never load."""
    from tour_worker.config import Settings
    from tour_worker.jobs import segment as job

    class FakeStorage:
        def get(self, uri, dest):
            cv2.imwrite(dest + ".png", _photo())
            os.replace(dest + ".png", dest)

        def put(self, path, uri):
            pass

    calls = []

    def seg(img):
        calls.append(1)
        m = np.zeros(img.shape[:2], np.uint8)
        m[50:300, 200:400] = 255
        return [{"object_id": "seg_000", "class_name": "curtain", "confidence": 0.99,
                 "bbox": [200, 50, 400, 300], "mask": m, "space": [img.shape[1], img.shape[0]]}]

    settings = Settings.from_env({})
    v = job.segment_version(settings)
    assert v.startswith(f"seg{job.SEGMENT_VERSION}|shi-labs/oneformer_coco_swin_large@")
    req = {"images": [{"photo_id": "img_0", "uri": "s3://b/a.jpg", "known_version": v},
                      {"photo_id": "img_1", "uri": "s3://b/b.jpg", "known_version": "seg0|old"},
                      {"photo_id": "img_2", "uri": "s3://b/c.jpg"}], "output_prefix": "s3://b/seg/j/"}
    with tempfile.TemporaryDirectory() as td:
        res = job.run_segment(req, FakeStorage(), td, segmenter=seg, settings=settings)
    assert res["warnings"] == []
    assert res["cached"] == ["img_0"] and res["segment_version"] == v
    assert sorted(res["results"]) == ["img_1", "img_2"] and len(calls) == 2
    from tour_worker.contract.jobs import TourSegmentResult
    TourSegmentResult.model_validate(res)

    def boom(_settings):
        raise AssertionError("models loaded with nothing to segment")

    old = job.get_segmenter, job.get_fireplace_segmenter
    job.get_segmenter = job.get_fireplace_segmenter = boom
    try:
        with tempfile.TemporaryDirectory() as td:
            res = job.run_segment({"images": [{"photo_id": "img_0", "uri": "s3://b/a.jpg", "known_version": v}],
                                   "output_prefix": "s3://b/seg/j/"}, FakeStorage(), td, settings=settings)
    finally:
        job.get_segmenter, job.get_fireplace_segmenter = old
    assert res["cached"] == ["img_0"] and res["results"] == {}
