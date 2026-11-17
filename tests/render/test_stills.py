"""Instagram feed stills (render/stills.py, SOW EX-03): one JPEG a room, 4:5 at 1080 x 1350, from late
in the room's lead shot - the same bytes every time."""
import hashlib
import os
import tempfile

import cv2

from tests.render.test_direct import _plan as two_shots, _workdir as two_shot_images
from tour_worker.render.direct import render_direct
from tour_worker.render.stills import FEED_SIZE, feed_stills, still_times


def test_one_feed_still_a_room_cropped_to_four_by_five_and_deterministic():
    plan = two_shots()
    times = still_times(plan)
    assert [r for r, _, _ in times] == list(dict.fromkeys(s.room_key for s in plan.shots))
    t0 = plan.shots[0].duration_s
    assert times[0][2] < t0 < times[-1][2] or len(times) == 1        # inside each room's own shot
    digests = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as d:
            two_shot_images(d)
            video = render_direct(plan, d, workers=1)
            made = feed_stills(video, plan, os.path.join(d, "stills"))
            assert len(made) == len(times)
            for st in made:
                img = cv2.imread(st["path"])
                assert (img.shape[1], img.shape[0]) == FEED_SIZE
            digests.append([hashlib.md5(open(st["path"], "rb").read()).hexdigest() for st in made])
    assert digests[0] == digests[1]
