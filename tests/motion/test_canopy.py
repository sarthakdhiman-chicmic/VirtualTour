"""tour_worker.motion.canopy: leaves move, trunks / sky / solids / hedges stay still, sway on lines."""

import numpy as np

from tests.perceptual_fixtures import SH, SW, _Spec
from tour_worker.motion.region import Region


def test_trunks_and_open_sky_stay_still_in_trees():
    from tour_worker.motion.canopy import leaf_weight
    rng = np.random.default_rng(1)
    img = np.zeros((300, 200, 3), np.uint8)
    img[:] = (230, 180, 120)                                         # sky (BGR: blue)
    crown = (np.clip(rng.normal(0, 40, (150, 200)), -80, 80)[..., None] + [40, 140, 60]).clip(0, 255)
    img[:150] = crown.astype(np.uint8)                               # textured green leaves
    img[150:, 85:115] = (40, 70, 110)                                # smooth brown trunk
    mask = np.zeros((300, 200), np.float32)
    mask[:150] = 1.0
    mask[150:, 60:140] = 1.0                                         # trunk + sky beside it
    w, solid = leaf_weight(img, mask)
    assert w[20:130, 20:180].mean() > 0.6                            # crown moves
    assert w[170:, 90:110].max() < 0.1                               # trunk does not
    assert w[170:, 62:80].max() < 0.1                                # nor the sky in the mask
    assert w[mask == 0].max() == 0


def test_solid_objects_in_a_tree_stay_rigid_and_hedges_still():
    """Review 2026-10-16: lantern frames inside a tree mask bent; hedges boiled."""
    from tour_worker.motion.canopy import leaf_weight
    rng = np.random.default_rng(2)
    H, W = 600, 400
    img = (np.clip(rng.normal(0, 40, (H, W)), -80, 80)[..., None] + [40, 140, 60]).clip(0, 255).astype(np.uint8)
    img[150:350, 180:186] = (35, 35, 38)                             # a dark grey lantern rod
    mask = np.zeros((H, W), np.float32)
    mask[:450] = 1.0                                                 # the tree (crown above)
    mask[520:560, 50:350] = 1.0                                      # a separate low hedge row
    reach = 8.0
    w, solid = leaf_weight(img, mask, reach)
    assert solid[150:350, 180:186].mean() > 0.75                     # the rod is solid
    # a leaf at distance d from the rod moves at most w x reach: it must never reach the rod
    for d in range(1, 3 * int(reach)):
        assert w[160:340, 185 + d].max() * reach < d
    assert w[60:200, 300:380].mean() > 0.5                           # the crown away from it moves
    assert w[520:560].max() == 0.0                                   # the hedge row is still


def test_trees_sway_sideways_together_not_in_circles():
    """The earlier field (independent x / y noise at branch scale) read as "a circular warp around
    the trees". The crown now sways sideways as one: mostly horizontal, neighbouring parts moving
    the same way, the sway travelling across the tree."""
    r = Region(_Spec(effect="canopy_flutter", anchor="none", amp=8.0, bbox=[0.2, 0.1, 0.6, 0.6], freq=1.1),
               np.ones((SH, SW), np.float32), SW, SH)
    u, v = np.meshgrid(np.linspace(0.25 * SW, 0.55 * SW, 12), np.linspace(0.15 * SH, 0.55 * SH, 10))
    u, v = u.ravel(), v.ravel()
    dxs, dys, same = [], [], []
    for f in range(0, 150, 3):
        dx, dy = r.displacement(u, v, f / 30)
        dxs.append(dx)
        dys.append(dy)
        big = np.abs(dx) > 0.3 * np.abs(dx).max()
        if big.sum() > 10:
            same.append(max((dx[big] > 0).mean(), (dx[big] < 0).mean()))
    dxs, dys = np.array(dxs), np.array(dys)
    # mostly sideways; since 2026-10-21 some branch clusters swing at an angle (the variety), so
    # vertical motion is a real share - but each point still moves on a line (test below)
    assert np.abs(dys).mean() < 0.8 * np.abs(dxs).mean()
    assert np.mean(same) > 0.8                                           # parts move together
    assert np.abs(dxs).max() <= 1.6 * 8.0


def test_a_distant_tree_line_moves_but_a_hedge_stays_still():
    from tour_worker.motion.canopy import leaf_weight
    rng = np.random.default_rng(3)
    img = np.zeros((1000, 1000, 3), np.uint8)
    img[:] = (230, 180, 120)
    leaves = (np.clip(rng.normal(0, 40, (1000, 1000)), -80, 80)[..., None] + [40, 140, 60]).clip(0, 255)
    mask = np.zeros((1000, 1000), np.float32)
    mask[450:520, 100:900] = 1.0              # a thin tree line at the horizon (far)
    mask[860:950, 100:900] = 1.0              # a hedge in front of the camera
    img[mask > 0] = leaves[mask > 0].astype(np.uint8)
    w, _ = leaf_weight(img, mask)
    assert w[455:490, 200:800].mean() > 0.4   # the far line moves (not held as a hedge)
    assert w[860:950].max() == 0.0            # the hedge stays still


def test_tree_tops_sway_most_and_the_whole_crown_reverses_together():
    """A tree bends from its base (the top moves most) and, review 2026-10-22, reverses all at
    once: parts reversing at different moments read as rotation."""
    from tour_worker.motion.canopy import tree_columns
    mask = np.zeros((SH, SW), bool)
    mask[int(0.1 * SH):int(0.6 * SH), int(0.2 * SW):int(0.6 * SW)] = True
    r = Region(_Spec(effect="canopy_flutter", anchor="none", amp=8.0, bbox=[0.2, 0.1, 0.6, 0.6], freq=1.1),
               mask.astype(np.float32), SW, SH)
    r.col_bottom, r.col_span = tree_columns(mask)
    x = np.full(2, 0.4 * SW)
    top_bottom = np.array([0.15 * SH, 0.55 * SH])
    amp = np.zeros(2)
    for f in range(0, 300, 2):
        amp = np.maximum(amp, np.abs(r.displacement(x, top_bottom, f / 30)[0]))
    assert amp[0] > 1.8 * amp[1]                                     # the top sways most
    r.canopy_amps = lambda: (r.amp, 0.0)                             # the sway alone (no leaf jitter)
    u = np.linspace(0.25 * SW, 0.55 * SW, 40)
    v = np.full(40, 0.2 * SH)
    series = np.array([r.displacement(u, v, f / 30)[0] for f in range(0, 300, 2)])
    assert np.corrcoef(series.T)[0, -1] > 0.98                       # one timing across the crown
    assert (np.abs(np.diff(series, axis=1)) / (u[1] - u[0])).max() < 0.2   # no tearing / shear


def test_touching_crowns_split_into_trees_that_blend_without_tearing():
    """2026-10-22: COCO's "tree" is one mask for a whole tree line; the skyline's dips split it
    into trees, each with its own timing, blended over a soft band."""
    from tour_worker.motion.canopy import build_instances, tree_instances
    from tour_worker.motion.wind import Wind
    rng = np.random.default_rng(5)
    img = (np.clip(rng.normal(0, 30, (SH, SW)), -60, 60)[..., None] + [40, 140, 60]).clip(0, 255).astype(np.uint8)
    mask = np.zeros((SH, SW), np.float32)
    yy, xx = np.mgrid[0:SH, 0:SW]
    for cx in (300, 520):                                           # two round crowns, touching
        mask[((xx - cx) / 130.0) ** 2 + ((yy - 260) / 170.0) ** 2 <= 1.0] = 1.0
    mask[260:430, 300:520] = 1.0
    parts = tree_instances(img, mask)
    assert len(parts) == 2
    total = sum(w[mask[y0:y0 + w.shape[0], x0:x0 + w.shape[1]] > 0].sum() for w, (y0, x0) in parts)
    assert abs(total / mask.sum() - 1.0) < 1e-3                     # weights share the mask
    one = np.zeros((SH, SW), np.float32)
    one[((xx - 400) / 150.0) ** 2 + ((yy - 260) / 180.0) ** 2 <= 1.0] = 1.0
    assert tree_instances(img, one)[0][0] is None                   # a single crown stays one tree
    r = Region(_Spec(effect="canopy_flutter", anchor="none", amp=8.0, bbox=[0.15, 0.1, 0.7, 0.7], freq=1.1),
               mask, SW, SH)
    r.wind = Wind(7, SW)
    r.instances = build_instances(r, img, mask, r.wind)
    assert len(r.instances) == 2
    r.canopy_amps = lambda: (r.amp, 0.0)
    u = np.arange(200, 620, 4, dtype=float)
    v = np.full(len(u), 200.0)
    series = np.array([r.displacement(u, v, f / 30)[0] for f in range(0, 450, 3)])
    left, right = series[:, np.argmin(np.abs(u - 260))], series[:, np.argmin(np.abs(u - 560))]
    assert np.corrcoef(left, right)[0, 1] < 0.9                     # two trees: their own timing
    assert (np.abs(np.diff(series, axis=1)) / 4.0).max() < 0.25     # the band between: no tear


def test_the_shot_wind_is_shared_delayed_across_the_frame_and_never_overdrives():
    from tour_worker.motion.wind import CALM, Wind
    a, b = Wind(11, 1920), Wind(11, 1920)
    ts = np.arange(0, 60, 0.1)
    assert [a.envelope(t) for t in ts] == [b.envelope(t) for t in ts]      # deterministic
    env = np.array([a.envelope(t) for t in ts])
    assert env.min() >= CALM - 1e-9 and env.max() <= 1.0 + 1e-9           # never above the plan
    assert env.max() - env.min() > 0.15                                    # gusts are felt
    up, down = (0.0, 1920.0) if a.direction > 0 else (1920.0, 0.0)
    assert a.delay(up) == 0.0 and a.delay(down) > 1.5                      # crosses the frame


def test_tree_points_swing_on_a_line_not_in_circles():
    """Review 2026-10-21: "the trees look like they are doing a circular motion". A point's path
    over time must be a line (its 2-D spread nearly one-dimensional), not a loop."""
    from tour_worker.motion.canopy import tree_columns
    mask = np.zeros((SH, SW), bool)
    mask[64:384, 192:576] = True
    r = Region(_Spec(effect="canopy_flutter", anchor="none", amp=8.0, bbox=[0.2, 0.1, 0.6, 0.6], freq=1.1),
               mask.astype(np.float32), SW, SH)
    r.col_bottom, r.col_span = tree_columns(mask)
    r.amp = 8.0
    ab, al = r.canopy_amps()
    rng = np.random.default_rng(0)
    u = rng.uniform(0.25 * SW, 0.55 * SW, 30)
    v = rng.uniform(0.12 * SH, 0.5 * SH, 30)
    path = np.array([np.stack(r.displacement(u, v, f / 30)) for f in range(0, 600, 2)])  # (T, 2, N)
    for k in range(len(u)):
        p = path[:, :, k] - path[:, :, k].mean(axis=0)
        ev = np.linalg.eigvalsh(np.cov(p.T))
        # the minor axis is only the tiny leaf jitter: well under a tenth of the swing
        assert np.sqrt(ev[0]) < 0.1 * np.sqrt(ev[1]) + 0.5 * al, (k, ev)
