"""Near-duplicate photos (scoring/duplicates.py): the same picture twice is shown once."""
import cv2
import numpy as np

from tour_worker.scoring.duplicates import find_duplicates, thumb
from tour_worker.scoring.pipeline import score_photos


def _photo(seed, size=(640, 960)):
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur((rng.random((*size, 3)) * 255).astype(np.uint8), (0, 0), 6)
    for _ in range(8):
        x, y = int(rng.integers(0, size[1] - 100)), int(rng.integers(0, size[0] - 80))
        cv2.rectangle(img, (x, y), (x + 100, y + 80), tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
    return img


def test_the_same_picture_twice_is_found_and_the_better_one_kept():
    a = _photo(1)
    retouched = cv2.convertScaleAbs(a, alpha=1.08, beta=6)              # brighter export of the same frame
    smaller = cv2.resize(a, (480, 320), interpolation=cv2.INTER_AREA)   # another size
    thumbs = {"a": thumb(a), "b": thumb(retouched), "c": thumb(smaller), "d": thumb(_photo(2))}
    assert find_duplicates(thumbs, {"a": 0.6, "b": 0.7, "c": 0.5, "d": 0.4}) == {"a": "b", "c": "b"}


def test_scoring_leaves_a_duplicate_out_with_a_warning():
    a, b = _photo(3), _photo(4)
    enc = lambda im: cv2.imencode(".jpg", im)[1].tobytes()   # noqa: E731
    report = score_photos([("p1", enc(a), "dining_room"), ("p2", enc(a), "dining_room"), ("p3", enc(b), "kitchen")])
    ids = {p.photo_id for p in report.photos}
    assert len(ids) == 2 and "p3" in ids
    gone = ({"p1", "p2"} - ids).pop()
    assert report.warnings == [f"duplicate_photo:{gone}:{({'p1', 'p2'} - {gone}).pop()}"]
