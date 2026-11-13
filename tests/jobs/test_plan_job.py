"""tour_worker.jobs.plan: tour_plan = fetch photos -> scoring -> build_shot_plan -> parse_plan."""
import copy
import os

import cv2
import numpy as np
import pytest

from tour_worker.contract.jobs import JobFailure, TourPlanResult
from tour_worker.contract.plan import compute_plan_hash, parse_plan
from tour_worker.jobs.router import run_job, validate_request

ROOMS = ["exterior", "living_room", "living_room", "kitchen", "kitchen", "dining_room",
         "bedroom", "primary_bathroom"]


def _photo(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur((rng.random((480, 720, 3)) * 255).astype(np.uint8), (0, 0), 3)
    for _ in range(6):
        x, y = int(rng.integers(0, 660)), int(rng.integers(0, 440))
        cv2.rectangle(img, (x, y), (x + 50, y + 30), tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
        cv2.line(img, (int(rng.integers(0, 720)), 0), (int(rng.integers(0, 720)), 479), (20, 20, 20), 3)
    return cv2.imencode(".jpg", img)[1].tobytes()


PHOTOS = {f"s3://bucket/listings/L1/p{i:02d}.jpg": _photo(i) for i in range(len(ROOMS))}


class FakeStorage:
    """Serves PHOTOS by URI; records what was fetched. Never writes."""

    def __init__(self, photos=None, broken=()):
        self.photos = dict(PHOTOS if photos is None else photos)
        self.broken = set(broken)
        self.got = []

    def get(self, uri, dest):
        self.got.append(uri)
        if uri in self.broken:
            raise RuntimeError("HTTP fetch failed for " + uri)
        with open(dest, "wb") as f:
            f.write(self.photos[uri])

    def put(self, path, uri):
        raise AssertionError("tour_plan writes nothing")


def _request(**kw):
    req = {"job_type": "tour_plan", "job_id": "j1", "listing_id": "L1",
           "photos": [{"photo_id": f"img_{i}", "uri": uri, "room_type": rt}
                      for i, (uri, rt) in enumerate(zip(PHOTOS, ROOMS))]}
    req.update(kw)
    return req


def test_plans_a_tour_and_validates_it_with_the_renderers_checks():
    st = FakeStorage()
    res = run_job(_request(), storage=st)
    TourPlanResult.model_validate(res)
    plan = res["plan"]
    parse_plan(plan)
    assert res["plan_hash"] == plan["provenance"]["plan_hash"] == compute_plan_hash(plan)
    assert plan["listing_id"] == "L1" and plan["plan_version"] == "1.3"
    assert (plan["output"]["width"], plan["output"]["height"]) == (1920, 1080)
    # the plan keeps the canonical URIs; the photo size is the decoded size
    assert {s["image_uri"] for s in plan["shots"]} <= set(PHOTOS)
    assert all(s["source"] == {"width": 720, "height": 480} for s in plan["shots"])
    assert res["scoring"]["photos"] == len(ROOMS) and res["scoring"]["sequence"]
    assert res["warnings"] == [] and sorted(st.got) == sorted(PHOTOS)


def test_same_plan_as_scoring_and_planner_called_directly():
    """The job adds nothing of its own to the plan: the backend's former call, reproduced."""
    from tour_worker.planner.config import PlannerConfig
    from tour_worker.planner.planner import build_shot_plan
    from tour_worker.scoring.pipeline import score_photos
    inputs = [(f"img_{i}", data, rt) for i, (data, rt) in enumerate(zip(PHOTOS.values(), ROOMS))]
    assets = {f"img_{i}": {"uri": uri, "width": 720, "height": 480} for i, uri in enumerate(PHOTOS)}
    cfg = PlannerConfig(output_width=1080, output_height=1920, max_total_s=60.0)
    direct = build_shot_plan(score_photos(inputs), "L1", assets, cfg=cfg).model_dump(mode="json")
    res = run_job(_request(options={"output_width": 1080, "output_height": 1920, "max_total_s": 60}),
                  storage=FakeStorage())
    assert res["plan"] == direct


def test_a_file_size_limit_reaches_the_plan_and_is_absent_otherwise():
    """A platform's file-size limit (the backend's Instagram Reels preset: 100 MB) becomes the
    plan's output.max_bytes; without one the output (and so plan_hash) is what it always was."""
    from tour_worker.contract.plan import parse_plan
    res = run_job(_request(options={"output_width": 1080, "output_height": 1920, "max_total_s": 90,
                                    "max_bytes": 100_000_000}), storage=FakeStorage())
    assert res["plan"]["output"]["max_bytes"] == 100_000_000
    assert parse_plan(res["plan"]).output.max_bytes == 100_000_000
    res = run_job(_request(), storage=FakeStorage())
    assert "max_bytes" not in res["plan"]["output"]


def test_the_hevc_copy_reaches_the_plan_and_is_absent_otherwise():
    """options.hevc (the SOW's optional H.265) becomes the plan's output.hevc; without it the
    output (and so plan_hash) is what it always was."""
    from tour_worker.contract.plan import parse_plan
    res = run_job(_request(options={"hevc": True}), storage=FakeStorage())
    assert res["plan"]["output"]["hevc"] is True and parse_plan(res["plan"]).output.hevc
    res = run_job(_request(), storage=FakeStorage())
    assert "hevc" not in res["plan"]["output"] and not parse_plan(res["plan"]).output.hevc


def test_room_labels_caption_each_rooms_first_shot_and_are_absent_otherwise():
    """options.room_labels: the plan carries the labels' settings and each room's first shot its
    display name; staged photos are flagged. Without them, no caption / staged / room_labels key
    appears, so the output and plan_hash are what they always were."""
    photos = _request()["photos"]
    photos[3]["staged"] = True                              # a kitchen photo
    res = run_job(_request(photos=photos, options={"room_labels": {"scale": 0.9, "placement": "top_left"},
                                                   "mls_copy": True}), storage=FakeStorage())
    plan = res["plan"]
    assert plan["output"]["room_labels"] == {"font": "sacramento", "scale": 0.9, "placement": "top_left"}
    assert plan["output"]["mls_copy"] is True
    rooms, captions = [], []
    for shot in plan["shots"]:
        first = not rooms or rooms[-1] != shot["room_key"]
        rooms.append(shot["room_key"])
        assert ("caption" in shot) == first
        if first:
            captions.append(shot["caption"])
    assert "Kitchen" in captions and "Living Room" in captions
    assert all(s.get("staged") is (True if s["asset_id"] == "img_3" else None) for s in plan["shots"])
    parsed = parse_plan(plan)
    assert parsed.output.room_labels.placement == "top_left" and parsed.output.mls_copy
    plain = run_job(_request(), storage=FakeStorage())["plan"]
    assert not {"room_labels", "mls_copy"} & set(plain["output"])
    assert not any({"caption", "staged"} & set(s) for s in plain["shots"])


def test_bad_room_labels_are_invalid_request():
    for bad in ({"font": "comic_sans"}, {"scale": 0.4}, {"placement": "middle"}, {"colour": "red"}):
        res = run_job(_request(options={"room_labels": bad}), storage=FakeStorage())
        assert res["error"]["code"] == "invalid_request", bad


def test_image_urls_are_fetched_instead_and_stay_out_of_the_plan():
    signed = {uri: f"https://signed.example/{i}?X-Amz-Signature=sig" for i, uri in enumerate(PHOTOS)}
    photos = {signed[u]: d for u, d in PHOTOS.items()}
    st = FakeStorage(photos)
    res = run_job(_request(image_urls=signed), storage=st)
    assert res["status"] == "completed" and sorted(st.got) == sorted(signed.values())
    assert "signed.example" not in str(res["plan"])


def test_segments_and_music_reach_the_planner():
    region = {"object_id": "seg_000", "class_name": "curtain", "confidence": 0.99,
              "bbox": [200, 60, 420, 400], "space": [720, 480], "mask_url": "s3://b/seg/img_1/seg_000.png"}
    segments = {f"img_{i}": [dict(region, mask_url=f"s3://b/seg/img_{i}/seg_000.png")]
                for i in range(len(ROOMS))}
    res = run_job(_request(segments=segments), storage=FakeStorage())
    assert res["status"] == "completed" and res["plan"]["plan_version"] == "1.4"
    masks = [r["mask_uri"] for s in res["plan"]["shots"] for r in s.get("motion_regions") or []]
    assert masks and all(m.startswith("s3://b/seg/") for m in masks)

    beat, dur = 0.5, 120.0                                 # a steady 120 bpm track
    cands = [{"time": round(0.25 + i * beat, 4), "bar_start": i % 4 == 0, "kind": "beat",
              "value": 0.5 if i % 32 == 0 else 0.0, "change": 0.0, "what_happens": "beat"}
             for i in range(int((dur - 0.3) / beat))]
    analysis = {"analysis_version": "1", "duration": dur, "music_start": 0.0, "music_end": dur - 0.5,
                "tempo_bpm": 120.0, "has_beat": True, "beat_period_s": beat, "beats": [],
                "candidates": cands}
    track = {"track_id": "t1", "s3_uri": "s3://b/music/t1.mp3", "sha256": "ab" * 32, "analysis": analysis}
    res = run_job(_request(music_track=track), storage=FakeStorage())
    assert res["status"] == "completed", res
    assert res["plan"]["plan_version"] == "1.5" and res["plan"]["music"]["uri"] == "s3://b/music/t1.mp3"


def test_a_photo_that_cannot_be_fetched_or_read_is_left_out_with_a_warning():
    uris = list(PHOTOS)
    photos = dict(PHOTOS)
    photos[uris[2]] = b"not an image"
    res = run_job(_request(), storage=FakeStorage(photos, broken={uris[1]}))
    assert res["status"] == "completed"
    assert res["warnings"][0].startswith("photo_fetch_failed:img_1:")
    assert res["warnings"][1] == "photo_unreadable:img_2"
    assert {s["asset_id"] for s in res["plan"]["shots"]}.isdisjoint({"img_1", "img_2"})


def test_no_usable_photo_is_image_fetch_failed():
    res = run_job(_request(), storage=FakeStorage(broken=set(PHOTOS)))
    JobFailure.model_validate(res)
    assert res["error"]["code"] == "image_fetch_failed"


def test_a_planner_crash_is_planning_failed(monkeypatch):
    import tour_worker.planner.planner as planner

    def boom(*a, **k):
        raise ZeroDivisionError("division by zero")

    monkeypatch.setattr(planner, "build_shot_plan", boom)
    res = run_job(_request(), storage=FakeStorage())
    assert res["error"]["code"] == "planning_failed" and "ZeroDivisionError" in res["error"]["message"]


def test_a_plan_the_validator_rejects_keeps_the_validators_code(monkeypatch):
    import tour_worker.jobs.plan as plan_job

    def reject(plan):
        raise ValueError("unsupported_camera_type: s01 uses warp_drive")

    monkeypatch.setattr(plan_job, "parse_plan", reject)
    res = run_job(_request(), storage=FakeStorage())
    assert res["error"]["code"] == "unsupported_camera_type"


@pytest.mark.parametrize("change", [
    {"listing_id": ""},
    {"photos": []},
    {"photos": [{"photo_id": "a", "uri": "s3://b/a.jpg"}, {"photo_id": "a", "uri": "s3://b/b.jpg"}]},
    {"photos": [{"photo_id": "a"}]},
    {"options": {"fps": 60}},                              # options are explicit: no unknown keys
    {"options": {"output_width": 0}},
    {"music_track": {"track_id": "t"}},
])
def test_bad_requests_are_invalid_request_before_any_fetch(change):
    st = FakeStorage()
    req = _request(**change)
    res = run_job(req, storage=st)
    assert res["error"]["code"] == "invalid_request" and st.got == []
    assert validate_request(req) == res


def test_the_runpod_handler_runs_tour_plan(tmp_path, monkeypatch):
    """Both front doors share the router: handler.py here, the HTTP service in tests/service."""
    from handler import handler
    photos = []
    for i, (uri, rt) in enumerate(zip(PHOTOS, ROOMS)):
        p = tmp_path / os.path.basename(uri)
        p.write_bytes(PHOTOS[uri])
        photos.append({"photo_id": f"img_{i}", "uri": "file://" + str(p), "room_type": rt})
    res = handler({"input": _request(photos=photos)})
    assert res["status"] == "completed", res
    expected = run_job(_request(), storage=FakeStorage())["plan"]
    ours = copy.deepcopy(res["plan"])
    # the same plan but for the photo URIs (and so the hash)
    for a, b in zip(ours["shots"], expected["shots"]):
        assert os.path.basename(a.pop("image_uri")) == os.path.basename(b.pop("image_uri"))
    ours["provenance"].pop("plan_hash"); expected["provenance"].pop("plan_hash")
    assert ours == expected


def test_pacing_reaches_the_planner():
    """options.pacing (Slow / Normal / Fast): a slow tour is longer than a fast one, its shots longer."""
    slow = run_job(_request(options={"pacing": "slow"}), storage=FakeStorage())["plan"]
    fast = run_job(_request(options={"pacing": "fast"}), storage=FakeStorage())["plan"]
    assert slow["total_duration_s"] > fast["total_duration_s"]
    assert min(s["duration_s"] for s in slow["shots"]) > min(s["duration_s"] for s in fast["shots"])
    assert min(s["duration_s"] for s in fast["shots"]) >= 2.5 - 1e-6
    assert run_job(_request(options={"pacing": "turbo"}), storage=FakeStorage())["error"]["code"] == "invalid_request"


def test_branding_reaches_the_plan_and_an_end_card_holds_the_final_shot():
    branding = {"address": "1030 Turkey Trot", "locality": "Greensboro, GA",
                "agent": {"name": "Jordan Avery", "photo_uri": "s3://b/a.jpg"}, "logo_uri": "s3://b/l.png"}
    plan = run_job(_request(options={"branding": branding}), storage=FakeStorage())["plan"]
    assert plan["output"]["branding"]["agent"]["name"] == "Jordan Avery"
    assert plan["shots"][-1]["duration_s"] >= 6.0 - 1e-6
    assert "branding" not in run_job(_request(), storage=FakeStorage())["plan"]["output"]
    bad = run_job(_request(options={"branding": {"logo_uri": "ftp://x"}}), storage=FakeStorage())
    assert bad["status"] == "failed"


def test_presets_set_the_pace_the_kicker_and_outdoor_time_and_the_request_wins():
    """Property Showcase (Normal), Just Listed (Fast, "Just Listed" on the title card), Lifestyle
    (Slow, exteriors 1.5x); an explicit pacing beats the preset's."""
    def plan(**options):
        return run_job(_request(options=options), storage=FakeStorage())["plan"]

    def ext_time(p):
        return sum(s["duration_s"] for s in p["shots"] if s["room_type"] in ("exterior",))

    show, listed, life = plan(preset="property_showcase"), plan(preset="just_listed"), plan(preset="lifestyle")
    assert listed["output"]["branding"]["kicker"] == "Just Listed"
    assert "branding" not in show["output"] and "branding" not in life["output"]
    assert ext_time(life) > ext_time(show)
    assert plan(preset="lifestyle", pacing="fast")["total_duration_s"] < life["total_duration_s"]
    assert "preset_not_defined" in plan(preset="mystery")["warnings"]


def test_manual_order_plays_the_rooms_and_photos_as_the_request_lists_them():
    """photo_order "manual" (drag-to-reorder): rooms in the order of their first photo in the
    request, each room's photos in request order; "narrative" (default) is the planner's order."""
    photos = _request()["photos"]
    reordered = [photos[i] for i in (6, 0, 3, 1, 4, 2, 5, 7)]       # bedroom first, then exterior...
    plan = run_job(_request(photos=reordered, options={"photo_order": "manual"}), storage=FakeStorage())["plan"]
    rooms = []
    for s in plan["shots"]:
        if not rooms or rooms[-1] != s["room_type"]:
            rooms.append(s["room_type"])
    listed = []
    for p in reordered:
        if p["room_type"] not in listed:
            listed.append(p["room_type"])
    assert rooms == [r for r in listed if r in rooms]
    natural = run_job(_request(photos=reordered), storage=FakeStorage())["plan"]
    assert natural["shots"][0]["room_type"] == "exterior"            # the narrative opens outside


def test_a_voiceover_reaches_the_plan_and_is_absent_otherwise():
    plan = run_job(_request(options={"voiceover": {"uri": "s3://b/vo.mp3", "balance": 0.7}}), storage=FakeStorage())["plan"]
    assert plan["voiceover"] == {"uri": "s3://b/vo.mp3", "balance": 0.7}
    assert parse_plan(plan).voiceover.balance == 0.7
    assert "voiceover" not in run_job(_request(), storage=FakeStorage())["plan"]
    bad = run_job(_request(options={"voiceover": {"uri": "s3://b/vo.mp3", "balance": 2}}), storage=FakeStorage())
    assert bad["status"] == "failed"


# ---- a 3D tour: scene + rooms (tour_worker/planner3d) ------------------------------------------------
def _scene_request(tmp_path, **kw):
    """A tour_plan request for the planner3d test home, served from disk."""
    from tests.planner3d.test_planner3d import ROOMS as ROOMS_3D
    from tests.planner3d.test_planner3d import home
    from tests.scene_fixtures import write_splat
    from tour_worker.scene.cache import sha256_file
    path = tmp_path / "home.splat"
    write_splat(path, home())
    files = {"s3://bucket/scenes/home.splat": path.read_bytes()}
    req = {"job_type": "tour_plan", "job_id": "j3d", "listing_id": "L3D",
           "scene": {"uri": "s3://bucket/scenes/home.splat", "sha256": sha256_file(str(path)), "format": "splat"},
           "rooms": ROOMS_3D, "options": {"room_labels": {"font": "sacramento"}}}
    req.update(kw)
    return req, files


def test_a_scene_and_its_rooms_plan_a_3d_tour(tmp_path):
    req, files = _scene_request(tmp_path)
    res = run_job(req, storage=FakeStorage(files))
    assert res["status"] == "completed", res
    TourPlanResult.model_validate(res)
    plan = parse_plan(res["plan"])
    assert plan.mode == "cinematic_3d" and plan.scene.uri == "s3://bucket/scenes/home.splat"
    assert [s.room_key for s in plan.shots][0] == "living" and plan.shots[0].caption == "Living Room"
    assert res["planner_3d"]["sequence"] == ["living", "bed"] and res["scoring"]["photos"] == 0
    assert set(res["timings"]["stages"]) >= {"scene_fetch", "scene_load", "planning"}


def test_a_scene_needs_its_rooms_and_a_request_needs_photos_or_a_scene(tmp_path):
    req, _ = _scene_request(tmp_path)
    assert validate_request({**req, "rooms": None})["error"]["code"] == "invalid_request"
    bare = {k: v for k, v in req.items() if k not in ("scene", "rooms")}
    assert validate_request(bare)["error"]["code"] == "invalid_request"


def test_a_scene_that_fails_falls_back_to_the_photos_or_fails(tmp_path):
    req, files = _scene_request(tmp_path)
    bad = {**req, "scene": {**req["scene"], "sha256": "0" * 64}}       # the file is not what was promised
    res = run_job(bad, storage=FakeStorage(files))
    assert res["status"] == "failed" and res["error"]["code"] == "scene_invalid"
    photos = _request()["photos"]
    res = run_job({**bad, "photos": photos}, storage=FakeStorage({**PHOTOS, **files}))
    assert res["status"] == "completed" and res["plan"]["mode"] != "cinematic_3d"
    assert "scene_fallback:scene_invalid" in res["warnings"]
