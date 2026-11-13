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
    assert stats["views"] == 2 * 2 * 8 and stats["rooms"] == 2
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
    assert {o["class"] for o in res["objects"]} == {"sofa", "bed"} and res["stats"]["views"] == 32
