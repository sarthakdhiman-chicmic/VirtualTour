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
