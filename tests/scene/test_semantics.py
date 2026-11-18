"""Scene labels (scene/semantics.py, the scene_label job): classes voted onto the splats from
rendered survey views, objects with rooms, motion from the labels. A fake segmenter stands in for
Mask2Former: it reads the synthetic home's colours (a red sofa, a blue bed)."""
import numpy as np
import pytest

from tests.planner3d.test_planner3d import _box, home
from tour_worker.scene.splat_io import SH_C0, SplatModel

pytest.importorskip("torch")

RED, BLUE = (0.9, 0.1, 0.1), (0.1, 0.1, 0.9)


def coloured_home() -> SplatModel:
    m = home()
    rgb = np.full((len(m), 3), 0.6, np.float32)
    sofa = (m.means[:, 0] > 0.95) & (m.means[:, 0] < 3.25) & (m.means[:, 1] > 0.35) & (m.means[:, 1] < 1.35) \
        & (m.means[:, 2] > 0.15) & (m.means[:, 2] < 0.85)
    bed = (m.means[:, 0] > 7.55) & (m.means[:, 0] < 9.65) & (m.means[:, 1] > 2.95) & (m.means[:, 1] < 4.65) \
        & (m.means[:, 2] > 0.15) & (m.means[:, 2] < 0.65)
    rgb[sofa], rgb[bed] = RED, BLUE
    sh = ((rgb - 0.5) / SH_C0)[:, None, :].astype(np.float32)
    return SplatModel(m.means, m.scales, m.quats, m.opacities, sh, 0, "ply", {}), sofa, bed


class ColourSegmenter:
    """Mask2Former's interface (panoptic, id2label): red pixels a sofa, blue a bed, the rest wall."""
    id2label = {0: "wall", 7: "bed ", 23: "sofa"}

    def panoptic(self, bgr):
        b, g, r = (bgr[..., k].astype(int) for k in range(3))
        seg = np.zeros(bgr.shape[:2], np.int32)
        seg[(r > 150) & (b < 90)] = 1
        seg[(b > 150) & (r < 90)] = 2
        return bgr, seg, [{"id": 0, "label_id": 0}, {"id": 1, "label_id": 23}, {"id": 2, "label_id": 7}]


@pytest.fixture(scope="module")
def labelled(monkeypatch_module):
    from tour_worker.planner3d.space import Space
    from tour_worker.render.splat_cpu import CpuSplatRenderer
    from tour_worker.scene import semantics
    monkeypatch_module.setattr(semantics, "VIEW_W", 240)
    monkeypatch_module.setattr(semantics, "VIEW_H", 136)
    m, sofa, bed = coloured_home()
    sp = Space(m)
    sp.set_rooms({"living": np.array([3.0, 3.0]), "bed": np.array([8.0, 1.5])})
    lab, stats, seen = semantics.label_scene(m, sp, CpuSplatRenderer(m), ColourSegmenter(), ["living", "bed"], "ab" * 32)
    return m, sp, lab, stats, seen, sofa, bed


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_the_classes_land_on_the_right_splats(labelled):
    m, sp, lab, stats, seen, sofa, bed = labelled
    # spots spread over the whole open floor (one per SPOT_M2, at least SPOTS_MIN), 8 headings each -
    # the same views however the rooms are found; each view counts for the room it stands in
    from tour_worker.scene.semantics import SPOTS_MIN, VIEW_HEADINGS
    assert stats["views"] >= SPOTS_MIN * VIEW_HEADINGS and stats["views"] % VIEW_HEADINGS == 0 and stats["rooms"] == 2
    top_sofa = sofa & (m.means[:, 2] > 0.75)                 # its top: seen from every spot
    assert lab.mask("sofa")[top_sofa].mean() > 0.8 and lab.mask("bed")[bed & (m.means[:, 2] > 0.55)].mean() > 0.8
    assert lab.mask("sofa")[~sofa].mean() < 0.002 and lab.mask("bed")[~bed].mean() < 0.002
    assert lab.mask("wall")[(m.means[:, 0] < 0.02) & (m.means[:, 2] > 0.5)].mean() > 0.5


def test_objects_are_found_with_their_rooms(labelled):
    _, _, lab, _, _, _, _ = labelled
    objs = {o["class"]: o for o in lab.objects}
    assert set(objs) == {"sofa", "bed"}                      # walls are not objects
    assert objs["sofa"]["room"] == "living" and objs["bed"]["room"] == "bed"
    # votes reach what a view sees (the sofa's back is against the wall): the object lies within the
    # real sofa; the planner takes the exact footprint from the floor plan and the class from here
    cx, cy = objs["sofa"]["footprint"]["centre"]
    assert 1.0 <= cx <= 3.2 and 0.4 <= cy <= 1.35
    assert abs(objs["bed"]["top"] - 0.6) < 0.1


def test_the_labels_file_round_trips_and_is_bound_to_its_scene(labelled, tmp_path):
    from tour_worker.scene.semantics import LabelsInvalid, check_binding, load_labels
    _, _, lab, _, _, _, _ = labelled
    p = str(tmp_path / "l.npz")
    lab.save(p)
    back = load_labels(p)
    assert np.array_equal(back.labels, lab.labels) and back.objects == lab.objects
    assert check_binding(back, "ab" * 32, lab.splats) is None and "another scene" in check_binding(back, "cd" * 32, lab.splats)
    open(p, "wb").write(b"junk")
    with pytest.raises(LabelsInvalid):
        load_labels(p)


def test_a_lit_fireplace_gets_flames_and_an_unlit_one_none():
    """Bright, warm Gaussians inside a labelled fireplace flicker: still at their base, moving most
    at their tips; a fireplace without them (unlit) gets no motion."""
    from tour_worker.scene.motion_label import flame_field
    rng = np.random.default_rng(0)
    n = 3000
    flames = np.c_[rng.uniform(-0.3, 0.3, n), rng.uniform(-0.05, 0.05, n), rng.uniform(0.2, 0.7, n)].astype(np.float32)
    box = _box(-0.6, 0.6, -0.2, 0.2, 1.1).astype(np.float32)          # the fireplace's body (grey)
    means = np.concatenate([flames, box])
    rgb = np.concatenate([np.tile((1.0, 0.7, 0.3), (n, 1)), np.full((len(box), 3), 0.5)]).astype(np.float32)
    sh = ((rgb - 0.5) / SH_C0)[:, None, :]
    k = len(means)
    m = SplatModel(means, np.full((k, 3), 0.01, np.float32), np.tile([[1, 0, 0, 0]], (k, 1)).astype(np.float32),
                   np.full(k, 0.9, np.float32), sh.astype(np.float32), 0, "ply", {})
    fire = {"class": "fireplace", "min": [-0.6, -0.2, 0.0], "max": [0.6, 0.2, 1.1],
            "footprint": {"centre": [0, 0], "long": 1.2, "short": 0.4, "axis_deg": 0.0}}
    f = flame_field(m, means, [fire], (0, 0, 1), np.eye(3), "ab" * 32)
    assert f is not None and f.classes == {"flame": n} and set(f.idx.tolist()) == set(range(n))
    z = means[f.idx, 2]
    assert f.amp[z < 0.22, 0].max() < 0.2 * f.amp[:, 0].max() and f.amp[z > 0.68, 0].min() > 0.7 * f.amp[:, 0].max()
    unlit = SplatModel(m.means, m.scales, m.quats, m.opacities, np.full_like(m.sh, 0.0), 0, "ply", {})
    assert flame_field(unlit, means, [fire], (0, 0, 1), np.eye(3), "ab" * 32) is None


def test_the_scene_label_job_stores_labels_and_reports_objects(tmp_path, monkeypatch):
    from tests.scene_fixtures import write_splat
    from tour_worker.contract.jobs import SceneLabelResult
    from tour_worker.jobs.scene_label import run_scene_label
    from tour_worker.render.splat_cpu import CpuSplatRenderer
    from tour_worker.scene import semantics
    from tour_worker.scene.cache import sha256_file
    monkeypatch.setattr(semantics, "VIEW_W", 240)
    monkeypatch.setattr(semantics, "VIEW_H", 136)
    m, _, _ = coloured_home()
    path = tmp_path / "home.splat"
    write_splat(path, m)
    loaded = __import__("tour_worker.scene.splat_io", fromlist=["load_scene"]).load_scene(str(path), "splat")

    class Disk:
        def __init__(self):
            self.puts = {}

        def get(self, uri, dest):
            open(dest, "wb").write(path.read_bytes())

        def put(self, src, uri):
            self.puts[uri] = open(src, "rb").read()
    st = Disk()
    req = {"job_type": "scene_label", "job_id": "sl1", "output_prefix": "s3://b/scenes/home/",
           "scene": {"uri": "s3://b/home.splat", "sha256": sha256_file(str(path)), "format": "splat"},
           "rooms": [{"room_key": "living", "room_type": "Living Room", "point": [3, 3, 1]},
                     {"room_key": "bed", "room_type": "Bedroom", "point": [8, 1.5, 1]}]}
    (tmp_path / "w").mkdir()
    res = run_scene_label(req, st, str(tmp_path / "w"), segmenter=ColourSegmenter(), renderer=CpuSplatRenderer(loaded))
    SceneLabelResult.model_validate(res)
    assert res["labels_uri"] == "s3://b/scenes/home/scene_labels.npz" and res["labels_uri"] in st.puts
    assert "motion_uri" not in res                                       # nothing moves here
    from tour_worker.scene.semantics import SPOTS_MIN, VIEW_HEADINGS
    assert {o["class"] for o in res["objects"]} == {"sofa", "bed"} and res["stats"]["views"] >= SPOTS_MIN * VIEW_HEADINGS


def test_with_no_rooms_given_the_job_finds_and_names_them(tmp_path, monkeypatch):
    """No input but the scene (user 2026-11-16): the rooms come from the floor plan and their
    types from what they hold - the sofa's room is the living room, the bed's the primary bedroom -
    returned as tour_plan takes them."""
    from tests.scene_fixtures import write_splat
    from tour_worker.contract.jobs import SceneLabelResult
    from tour_worker.jobs.scene_label import run_scene_label
    from tour_worker.render.splat_cpu import CpuSplatRenderer
    from tour_worker.scene import semantics
    from tour_worker.scene.cache import sha256_file
    from tour_worker.scene.splat_io import load_scene
    monkeypatch.setattr(semantics, "VIEW_W", 240)
    monkeypatch.setattr(semantics, "VIEW_H", 136)
    m, _, _ = coloured_home()
    path = tmp_path / "home.splat"
    write_splat(path, m)

    class Disk:
        def get(self, uri, dest):
            open(dest, "wb").write(path.read_bytes())

        def put(self, src, uri):
            pass
    req = {"job_type": "scene_label", "job_id": "sl2", "output_prefix": "s3://b/scenes/home/",
           "scene": {"uri": "s3://b/home.splat", "sha256": sha256_file(str(path)), "format": "splat"}}
    (tmp_path / "w").mkdir()
    res = run_scene_label(req, Disk(), str(tmp_path / "w"), segmenter=ColourSegmenter(),
                          renderer=CpuSplatRenderer(load_scene(str(path), "splat")))
    SceneLabelResult.model_validate(res)
    rooms = {r["room_type"]: r for r in res["rooms"]}
    assert set(rooms) == {"living_room", "primary_bedroom"}
    assert rooms["living_room"]["point"][0] < 6.0 < rooms["primary_bedroom"]["point"][0]     # the wall at x = 6


def test_a_wall_behind_a_curtain_takes_no_curtain_votes_with_a_depth_image():
    """With the renderer's depth image, a wall 10 cm behind a curtain is hidden (the point depth
    buffer's 3 cm + 4% let it through, and it took the curtain's votes)."""
    from tour_worker.scene.camera import intrinsics
    from tour_worker.scene.semantics import INDEX, Voter
    W, H = 64, 48
    K = intrinsics(60.0, W, H)
    xs, ys = np.meshgrid(np.linspace(-0.3, 0.3, 12), np.linspace(-0.2, 0.2, 10))
    curtain = np.c_[xs.ravel(), ys.ravel(), np.full(xs.size, 2.0)]
    wall = np.c_[xs.ravel(), ys.ravel(), np.full(xs.size, 2.1)]
    pts = np.vstack([curtain, wall]).astype(np.float32)
    # a sheer curtain: below the point buffer's opacity, so it never hid the wall there
    opac = np.r_[np.full(len(curtain), 0.3), np.full(len(wall), 0.9)].astype(np.float32)
    classes = np.full((H, W), INDEX["curtain"], np.uint8)
    view = np.eye(4)
    v_img, v_buf = Voter(pts, opac), Voter(pts, opac)
    for _ in range(3):
        v_img.add(view, K, W, H, classes, depth=np.full((H, W), 2.0, np.float32))
        v_buf.add(view, K, W, H, classes)
    lab_img, lab_buf = v_img.labels(), v_buf.labels()
    n = len(curtain)
    assert (lab_img[:n] == INDEX["curtain"]).all() and (lab_img[n:] == 0).all()
    assert (lab_buf[n:] == INDEX["curtain"]).all()        # the old buffer's leak, kept as the fallback


def test_what_moves_needs_more_agreement():
    """At 4 of 9 views agreeing, a sofa Gaussian is labelled and a curtain one is not (MOVING):
    a wrong curtain label puts motion on a wall."""
    from tour_worker.scene.semantics import INDEX, Voter
    v = Voter(np.zeros((2, 3), np.float32), np.ones(2, np.float32))
    v.seen[:] = 9
    v.hits[0, INDEX["sofa"]] = 4                 # 4 of 9: over MIN_SHARE, under MOVING_SHARE
    v.hits[1, INDEX["curtain"]] = 4
    lab = v.labels()
    assert lab[0] == INDEX["sofa"] and lab[1] == 0
    v.hits[1, INDEX["curtain"]] = 5
    assert v.labels()[1] == INDEX["curtain"]


def test_one_thing_named_differently_from_view_to_view_is_still_labelled():
    """Seen 8 times as sofa x3, armchair x3, chair x2: eight votes for seating, labelled sofa (the
    first of its most-voted classes) - per class it was three of eight, under MIN_SHARE, unlabelled."""
    from tour_worker.scene.semantics import INDEX, Voter
    v = Voter(np.zeros((3, 3), np.float32), np.ones(3, np.float32))
    v.seen[:] = 8
    v.hits[0, INDEX["sofa"]], v.hits[0, INDEX["armchair"]], v.hits[0, INDEX["chair"]] = 3, 3, 2
    v.hits[1, INDEX["armchair"]], v.hits[1, INDEX["sofa"]] = 4, 1            # its most-voted class
    v.hits[2, INDEX["sofa"]], v.hits[2, INDEX["bed"]] = 2, 2                 # different things: no group
    lab = v.labels(chunk=2)                                                   # (across chunks too)
    assert lab[0] == INDEX["sofa"] and lab[1] == INDEX["armchair"] and lab[2] == 0


def test_a_thick_wall_layer_is_seen_and_the_wall_behind_a_curtain_is_not():
    """The depth image's allowance grows with each Gaussian's size: a wall's noisy layer, splats up
    to 8 cm deep (each 4 cm across), is seen; a wall 10 cm behind a curtain is still hidden."""
    from tour_worker.scene.camera import intrinsics
    from tour_worker.scene.semantics import INDEX, Voter
    W, H = 64, 48
    K = intrinsics(60.0, W, H)
    xs, ys = np.meshgrid(np.linspace(-0.3, 0.3, 12), np.linspace(-0.2, 0.2, 10))
    n = xs.size
    layer = np.c_[xs.ravel(), ys.ravel(), 2.0 + np.linspace(0.0, 0.08, n)]           # 8 cm deep
    behind = np.c_[xs.ravel(), ys.ravel(), np.full(n, 2.1)]                          # 10 cm back
    pts = np.vstack([layer, behind]).astype(np.float32)
    scales = np.r_[np.full((n, 3), 0.04), np.full((n, 3), 0.03)].astype(np.float32)
    classes = np.full((H, W), INDEX["wall"], np.uint8)
    depth = np.full((H, W), 2.0, np.float32)
    flat = Voter(pts, np.ones(len(pts), np.float32), depth_tol=(0.02, 0.02, 0.0))
    sized = Voter(pts, np.ones(len(pts), np.float32), scales=scales)
    for v in (flat, sized):
        for _ in range(3):
            v.add(np.eye(4), K, W, H, classes, depth=depth)
    seen_flat, seen_sized = flat.seen_img_np, sized.seen_img_np      # (at the depth image's surface)
    assert (seen_flat[:n] == 0).any()                    # the flat allowance loses the back of the layer
    assert (seen_sized[:n] > 0).all()                    # sized: the whole layer
    assert (seen_sized[n:] == 0).all()                   # and still not the wall behind


def test_no_vote_group_spans_two_room_types_evidence():
    """A group never mixes what tells one room type from another (planner3d/discover.py): counters
    stay kitchen, a desk / bookcase / monitor stay study, a TV stays living room."""
    from tour_worker.planner3d.discover import COOKING, KITCHEN, STUDY
    from tour_worker.scene.semantics import GROUPS
    kinds = {"kitchen": set(KITCHEN) | set(COOKING), "study": set(STUDY) | {"desk", "computer", "monitor"},
             "dining": {"table"}, "living": {"sofa", "television receiver", "fireplace"}, "bed": {"bed"}}
    for g in GROUPS:
        hit = {k for k, cs in kinds.items() if cs & set(g)}
        assert len(hit) <= 1, (g, hit)
