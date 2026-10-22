"""Perceptual motion effects (plan 1.4): displacement fields inside object masks.

Each region moves the photo's pixels in source space; prewarp composes that with the camera warp
in a single resample (no extra blur). Displacement is scaled by a feathered mask (0 at the mask
edge, 1 inside), so the boundary never tears, and by an anchor profile, so a curtain moves from
the rod down and a plant from the pot up. Everything is smooth in time (sines + gradient noise)
and seeded, so renders stay deterministic.

  fabric_sway   anchored at the top; a slow wave travels down the fabric, folds move slightly
                differently, a little low-frequency noise keeps it from looking mechanical
  foliage_sway  anchored at the bottom; a slow sway grows toward the top, plus fine leaf flutter
  water_ripple  (superseded by calm_water) two layers of drifting noise; reads as static
  calm_water    a moving wave height field on the water plane, in perspective (thin dense bands
                far away, wider near the camera); the reflection is displaced by the surface slope,
                mostly sideways, so reflected edges sway in horizontal bands; brightness follows
                the slope smoothly (+-4%, no thresholds: no flicker). amplitude_px = peak sideways
                displacement near the camera
  canopy_flutter  trees (review 2026-10-19: the earlier independent x / y branch noise read as "a
                circular warp around the trees"): the crown sways sideways as one, the sway
                travelling across it like wind (neighbouring parts move together, the far side
                a moment later), gusts swelling and fading, branch masses lagging a little, leaf
                shimmer on top; the whole foliage mask moves (trunks and solids held).
                amplitude_px = peak displacement

Strengths were set by what can be SEEN, not by numbers: a time-slice (one image row stacked over
time) must show wiggling edges, not straight stripes (tests/test_perceptual.py).
"""
from __future__ import annotations

import math
import os

import cv2
import numpy as np

from modules.perceptual.noise import GradientNoise3D

FEATHER_FRAC = 0.04            # feather width as a fraction of the region's shorter side
FEATHER_PX = (4.0, 16.0)
# a mask must belong to its detection: most of it inside the (slightly grown) bbox, and filling a
# fair part of it. Stored detect masks can be out of step with the detection list (seen: a curtain
# detection whose mask file was a side table), and moving the wrong object is worse than no motion.
BOX_MARGIN_FRAC = 0.03
MIN_INSIDE_SHARE = 0.8
MIN_BOX_FILL = 0.15


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def load_feather(mask_path: str, src_w: int, src_h: int, bbox_px) -> tuple[np.ndarray | None, str]:
    """The mask scaled to the source photo, checked against its bbox, clipped to it and feathered
    inward: (float32 (src_h, src_w) in [0, 1], "") or (None, reason)."""
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None, "unreadable"
    if m.shape != (src_h, src_w):
        # the mask is the detected image's size; it must be the same photo (same aspect)
        if abs(m.shape[1] / m.shape[0] - src_w / src_h) > 0.02:
            return None, "aspect"
        m = cv2.resize(m, (src_w, src_h), interpolation=cv2.INTER_LINEAR)
    binary = (m >= 128).astype(np.uint8)
    total = int(binary.sum())
    if not total:
        return None, "empty"
    bw, bh = bbox_px[2] - bbox_px[0], bbox_px[3] - bbox_px[1]
    mx, my = BOX_MARGIN_FRAC * src_w, BOX_MARGIN_FRAC * src_h
    x0, y0 = max(0, int(bbox_px[0] - mx)), max(0, int(bbox_px[1] - my))
    x1, y1 = min(src_w, int(np.ceil(bbox_px[2] + mx))), min(src_h, int(np.ceil(bbox_px[3] + my)))
    inside = int(binary[y0:y1, x0:x1].sum())
    if inside < MIN_INSIDE_SHARE * total or inside < MIN_BOX_FILL * max(1.0, bw * bh):
        return None, "mismatch"
    clipped = np.zeros_like(binary)
    clipped[y0:y1, x0:x1] = binary[y0:y1, x0:x1]
    binary = clipped
    feather = min(max(FEATHER_FRAC * min(bw, bh), FEATHER_PX[0]), FEATHER_PX[1])
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    # the mask's outermost pixel ring stays at exactly 0, so bilinear sampling just outside the
    # mask (between pixel centres) never picks up a non-zero weight
    return _smoothstep((dist - 1.0) / feather).astype(np.float32), ""


# calm_water wave trains on the water plane: (direction from "toward the camera" deg, wavelength,
# speed in plane units per s, relative amplitude). Plane units: x ~ du / (v - v_h), z ~ 1 / (v - v_h).
CALM_TRAINS = ((-12, 0.090, 0.050, 1.00), (18, 0.060, 0.042, 0.70), (-35, 0.038, 0.034, 0.45),
               (50, 0.024, 0.028, 0.25))
_SX_MAX = sum(a * abs(math.sin(math.radians(d))) * 2 * math.pi / lam for d, lam, _, a in CALM_TRAINS)
_SZ_MAX = sum(a * abs(math.cos(math.radians(d))) * 2 * math.pi / lam for d, lam, _, a in CALM_TRAINS)
CALM_BRIGHTNESS = 0.04
FIRE_GLOW = 0.08            # the warm light's flicker on the hearth and walls around a fire
FIRE_PLUME = 0.45           # flames may stretch above their tips by this share of their height
# 2026-10-21 (user: "too much flame motion will warp the mantles"): the flames move only inside the
# fire's own box - the dark cavity connected to them, at most FIREBOX_SIDE of their width beside
# them and FIRE_PLUME of their height above - and their strength follows their own size
FIREBOX_SIDE = 0.25
STRUCT_EDGE = 0.25          # gradient (x 255) above which an edge is hard: structure, not flame
STRUCT_FAINT_EDGE = 0.06    # a faint edge counts as structure when it runs straight for ...
STRUCT_LONG = 0.35          # ... this share of the fire box's width (or height)
STRUCT_LIT_V = 0.28         # in a fire's box, non-flame pixels at least this bright are held ...
STRUCT_ABOVE = 0.15         # ... above the fire (rows above its top + this share of its height)
FIRE_AMP_OF_HEIGHT = 0.15   # worker cap: peak flame displacement <= this x the flames' height
SWAY_HZ = 0.22              # trees: the crown's sway (the plan's freq_hz drives the leaf shimmer)
BRANCH_HZ = 0.55            # each branch cluster's own, quicker sway, at its own phase
LEAF_SHIMMER = 0.03         # leaves turning in the wind: +-3% light at leaf-cluster scale
GUST_PERIOD_S = 6.5         # a gust front crosses each tree this often
SWAY_ACROSS = 0.35          # of a sway cycle across a tree's width: the wind's travel
# the approved prototype's slope -> light weights (tanh(0.0009 * raw slopes)), on normalised slopes
_SHADE_Z, _SHADE_X = 0.0009 * 0.6 * _SZ_MAX, 0.0009 * 0.4 * _SX_MAX


def sample_bilinear(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """img at the points (u, v), bilinear, 0 outside. cv2.remap caps output sides at 32767, so the
    points are laid out as rows of 4096."""
    n = u.size
    if n == 0:
        return np.zeros(u.shape, np.float32)
    w = 4096
    rows = -(-n // w)
    pad = rows * w - n
    mu = np.pad(u.astype(np.float32).ravel(), (0, pad)).reshape(rows, w)
    mv = np.pad(v.astype(np.float32).ravel(), (0, pad)).reshape(rows, w)
    out = cv2.remap(img, mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out.ravel()[:n].reshape(u.shape)


class Region:
    def __init__(self, spec, feather: np.ndarray, src_w: int, src_h: int):
        self.spec = spec
        self.effect, self.anchor = spec.effect, spec.anchor
        self.amp, self.freq = float(spec.amplitude_px), float(spec.freq_hz)
        x1, y1, x2, y2 = spec.bbox
        self.x0, self.y0 = x1 * src_w, y1 * src_h
        self.bw, self.bh = max(1.0, (x2 - x1) * src_w), max(1.0, (y2 - y1) * src_h)
        self.feather = feather
        self.src_w = src_w
        rng = np.random.default_rng(spec.seed & 0xFFFFFFFF)
        self.phase, self.phase2 = rng.uniform(0, 2 * math.pi, 2)
        if getattr(spec, "phase", None) is not None:   # planner-set: curtains vs plants opposite
            self.phase = float(spec.phase)
        self.noise_a = GradientNoise3D(spec.seed)
        self.noise_b = GradientNoise3D(spec.seed ^ 0x5F3759DF)
        # calm_water plane: a horizon above the far edge of the water
        self.v_bot = self.y0 + self.bh
        self.v_h = self.y0 - 0.9 * self.bh
        self.u_mid = self.x0 + self.bw / 2
        self.ref = self.v_bot - self.v_h

    def calm_irregular(self, u, v, t):
        """The slow (80-140 source px) variation of wave strength across the surface."""
        d = np.maximum(v - self.v_h, 1.0)
        return 1.0 + 0.35 * self.noise_a((u - self.u_mid) / d * 9.0, self.ref / d * 2.5, 0.25 * t)

    def _calm_slopes(self, u, v, t, irregular=None):
        d = np.maximum(v - self.v_h, 1.0)
        x, z = (u - self.u_mid) / d, self.ref / d
        sx, sz = np.zeros_like(u), np.zeros_like(u)
        for k, (ang, lam, spd, amp) in enumerate(CALM_TRAINS):
            a = math.radians(ang)
            kx, kz = math.sin(a) * 2 * math.pi / lam, math.cos(a) * 2 * math.pi / lam
            c = amp * np.cos(kx * x + kz * z - 2 * math.pi * spd / lam * t + 1.3 * k + self.phase)
            sx += c * kx
            sz += c * kz
        if irregular is None:
            irregular = 1.0 + 0.35 * self.noise_a(x * 9.0, z * 2.5, 0.25 * t)
        near = np.clip((v - self.v_h) / self.ref, 0.0, 1.0)
        return sx / _SX_MAX * irregular, sz / _SZ_MAX * irregular, near

    def leaf_px(self) -> float:
        return max(8.0, 0.012 * self.src_w)                  # leaf clusters

    def branch_px(self) -> float:
        return max(40.0, 0.3 * min(self.bw, self.bh))        # branch masses

    def canopy_amps(self) -> tuple[float, float]:
        """(branch sway, leaf flutter) peaks: the leaves never move more than a quarter of a
        cluster (beyond that foliage boils), the branches carry the rest of the strength."""
        # 2026-10-19: a quarter cluster stretched leaves by up to 0.46 px per px (visible pixel
        # shift); the leaves' life now comes mostly from the light shimmer (LEAF_SHIMMER)
        al = min(0.1 * self.leaf_px(), 0.12 * self.amp)
        return max(0.0, self.amp - al), al

    def step_error(self, step: int, screen_per_src: float) -> float:
        """Predicted bilinear interpolation error (screen px) of a coarse field grid."""
        if self.effect == "canopy_flutter":
            ab, al = self.canopy_amps()
            return screen_per_src * (ab * (step / (self.branch_px() * screen_per_src)) ** 2
                                     + al * (step / (self.leaf_px() * screen_per_src)) ** 2)
        feat = self.feature_px() * screen_per_src
        return self.amp * screen_per_src * (step / feat) ** 2 if feat > 0 else float("inf")

    def feature_px(self) -> float:
        """Size (source px) of the effect's main spatial detail: how fine a field grid it needs."""
        if self.effect == "fabric_sway":
            return min(max(40.0, 0.12 * self.bw), 0.5 * self.bw, 0.5 * self.bh)       # folds
        if self.effect == "canopy_flutter":
            return max(8.0, 0.012 * self.src_w)                                      # leaf clusters
        if self.effect == "water_ripple":
            return min(max(30.0, 0.08 * self.bw), max(10.0, 0.03 * self.bh)) * 0.5
        if self.effect == "foliage_sway":
            return min(0.35 * self.bw, 0.35 * self.bh)
        if self.effect == "fire_flicker":
            return self.flame_px()                                                   # flame tongues
        return 0.0                                                                   # calm_water: exact

    # ---- fire_flicker (modules/perceptual/fire.py): this region's feather is the glow (1 on the
    # flames, fading out around them); self.flame (set by load_regions) is the flames alone.
    def flame_px(self) -> float:
        return max(3.0, 0.18 * getattr(self, "flame_h", self.bh))

    def _fire(self, u, v, t):
        """Fire (reviews 2026-10-21): the v1 licking tongues read best; "only one kind of motion"
        was the fault. Layers: tongues licking up and travelling up the flames; the fire surging
        (growing / shrinking) over a second or two; a slight lean with the draft; a fast small
        flutter at the flame edges; brightness flicker rising through the flames, a quick
        sparkle, the fire's pulse, and a soft glow on its surroundings. Structure inside the
        fire's box (walls, glass frame, grate, glass reflections) is held still (load_regions)."""
        fw = sample_bilinear(self.flame, u, v)
        plume = getattr(self, "plume", None)
        pw = sample_bilinear(plume, u, v) if plume is not None else fw
        fh, fb = self.flame_h, self.flame_bottom
        s = np.clip((fb - v) / max(fh, 1.0), 0.0, 1.0) ** 0.8          # still at the base (logs)
        lam = self.flame_px()
        rise = 0.9 * fh * t                                             # tongues travel upward
        A = self.amp
        surge = 0.75 + 0.35 * float(self.noise_b(1.7, 0.2, 0.6 * t))    # the fire grows / shrinks
        lean = 0.15 * math.sin(2 * math.pi * 0.31 * t + self.phase)     # the draft
        n1 = self.noise_a(u / lam, (v + rise) / lam, 1.3 * t)
        n2 = self.noise_b(u / lam + 7.0, (v + rise) / lam, 1.7 * t)
        ff = 0.5 * lam                                                  # fast flutter, finer
        n3 = self.noise_a(u / ff + 11.0, (v + 1.4 * rise) / ff, 4.0 * t)
        n4 = self.noise_b(u / ff + 23.0, (v + 1.4 * rise) / ff, 4.0 * t)
        dx = A * (0.4 * n1 + lean + 0.15 * n4) * s * pw
        dy = -A * (0.5 * surge * (1.0 + n2) + 0.2 * n3) * s * pw         # lick up, surge, flutter
        whole = float(self.noise_b(0.3, 0.7, 2.4 * t))                  # the fire's overall flicker
        local = self.noise_a(u / lam + 3.0, (v + 1.6 * rise) / lam, 3.0 * t)
        sparkle = self.noise_b(u / ff + 5.0, (v + 2.0 * rise) / ff, 6.0 * t)
        # light rides on the moving tongues; the whole fire blinking reads as "a light flickering"
        b = (1.0 + fw * (0.22 * local + 0.04 * whole + 0.06 * sparkle + 0.05 * (surge - 0.75))
             + (1.0 - fw) * FIRE_GLOW * whole)
        return dx, dy, b

    def brightness(self, u, v, t):
        """Multiplicative brightness inside the region, or None (displacement-only effects)."""
        return self.fields(u, v, t)[2]

    def fields(self, u, v, t, irregular=None):
        """(dx, dy, brightness or None) at source points; calm_water's waves are computed once
        (irregular: calm_irregular at these points when already known)."""
        if self.effect == "fire_flicker":
            return self._fire(u, v, t)
        if self.effect == "canopy_flutter":
            return self._canopy(u, v, t)
        if self.effect != "calm_water":
            dx, dy = self.displacement(u, v, t)
            return dx, dy, None
        sx, sz, near = self._calm_slopes(u, v, t, irregular)
        return (self.amp * near * sx / 1.35, 0.5 * self.amp * near * sz / 1.35,
                1.0 + CALM_BRIGHTNESS * np.tanh(_SHADE_Z * sz + _SHADE_X * sx))

    def _canopy(self, u, v, t):
        """Trees (dx, dy, light). History: 2026-10-16 leaf jitter read as jelly; 2026-10-19 x/y
        branch noise read as a circular warp, then a coherent sideways sway with bend, branch
        phases and leaf shimmer read better but "a simple omni-directional sway". 2026-10-21
        ("varied motions ... like they would in a jungle"), layers:
          - the crown's sway, travelling across the tree; small trees sway quicker than big ones
          - branch clusters each swinging along their OWN line (mostly sideways, some angled up
            to ~55 deg) at their own phase, blending two rates across the crown; every point
            moves back and forth on its line, never in a loop (2026-10-21: loops read as
            "circular motion")
          - gust fronts sweeping across the canopy every few seconds: branches push harder and
            the leaves flip lighter as a front passes
          - rustling patches of leaves, on and off in different places
        Direction, phase and rate fields change slowly across the crown, so neighbouring leaves
        never shear apart (stretch <= ~0.25 px per px, tests)."""
        lb, ll = self.branch_px(), self.leaf_px()
        ab, al = self.canopy_amps()
        f = self.freq
        span = max(self.bw, 1.0)
        bend = self.bend(u, v)
        # stiffness: a tree a third of the photo high sways at SWAY_HZ; smaller ones quicker
        hz = SWAY_HZ * min(1.8, max(0.75, (0.33 * self.src_w / max(self.bh, 1.0)) ** 0.4))
        ph = 2 * math.pi * hz * t - 2 * math.pi * SWAY_ACROSS * (u - self.x0) / span + self.phase
        sway = 0.8 * np.sin(ph) + 0.2 * np.sin(1.63 * ph + self.phase2)
        gust = 0.6 + 0.4 * np.clip(0.5 + self.noise_b(u / (0.8 * span) - 0.1 * t, v / (0.8 * span),
                                                      0.07 * t), 0, 1)
        # a gust front sweeping across the crown (period GUST_PERIOD_S, off the crown between)
        p = (t / GUST_PERIOD_S + self.phase / (2 * math.pi)) % 1.0
        x_front = self.x0 + (2.2 * p - 0.6) * span
        front = np.exp(-(((u - x_front) / (0.3 * span)) ** 2))
        # branch clusters: own phase, two rates blended across the crown - SIDEWAYS only. Review
        # 2026-10-22: angled swings made the crown breathe ("something pushing from behind,
        # bulging out and in"): vertical motion that grows toward the top stretches and squashes
        # the crown; a tree bends sideways from its trunk. Gust fronts now only brighten the
        # leaves (light) - as displacement they squeezed the crown sideways as they passed.
        # neighbouring clusters stay close in step (phase changes slowly and by at most ~half a
        # turn across the crown): clusters swinging against each other squeeze the leaves
        # between them - the same breathing, sideways
        cs = 3.0 * lb
        bph = 0.5 * math.pi * self.noise_a(u / cs + 60.0, v / cs, 0.0)
        mix = np.clip(0.5 + self.noise_b(u / cs + 30.0, v / cs, 0.0), 0.0, 1.0)
        osc = ((1.0 - mix) * np.sin(2 * math.pi * BRANCH_HZ * t + bph)
               + mix * np.sin(2 * math.pi * 1.45 * BRANCH_HZ * t + 1.7 * bph))
        # rustling patches (on / off slowly), leaves flutter there
        rustle = np.clip(1.6 * (self.noise_a(u / (2.0 * lb) + 140.0, v / (2.0 * lb), 0.12 * t) + 0.1), 0.0, 1.0)
        swing = ab * bend * gust * (0.75 * sway + 0.25 * osc)
        jitter = al * (0.4 + 0.8 * rustle) * self.noise_b(u / ll, v / ll, f * t)   # leaves, tiny
        dx = swing + jitter
        dy = 0.4 * jitter
        light = 1.0 + LEAF_SHIMMER * (0.5 + 0.7 * rustle + 1.0 * front) * self.noise_b(
            u / ll + 80.0, v / ll, 1.6 * f * t)
        return dx, dy, light

    def bend(self, u, v):
        """Trees: 0.3 at the crown's base .. 1 at its top, per column (col_bottom / col_span set
        by load_regions; without them, 1)."""
        cb = getattr(self, "col_bottom", None)
        if cb is None:
            return np.ones_like(u)
        i = np.clip(u.astype(np.int64), 0, cb.shape[0] - 1)
        hh = np.clip((cb[i] - v) / np.maximum(self.col_span[i], 1.0), 0.0, 1.0)
        return 0.3 + 0.7 * hh ** 1.5

    def anchor_weight(self, v):
        if self.anchor == "none":           # water: the whole surface moves
            return np.ones_like(v), np.ones_like(v)
        if self.anchor == "top":            # hangs from a rod: still at the top, freest at the hem
            s = np.clip((v - self.y0) / self.bh, 0.0, 1.0)
            return s ** 1.3, s
        s = np.clip((self.y0 + self.bh - v) / self.bh, 0.0, 1.0)   # rooted: still at the base
        return s ** 1.5, s

    def displacement(self, u, v, t):
        """(dx, dy) in source px at source points (u, v), time t seconds."""
        a, s = self.anchor_weight(v)
        A, f = self.amp, self.freq
        if self.effect == "fabric_sway":
            ph = 2 * math.pi * f * t - 2 * math.pi * 0.6 * s + self.phase
            fold = 1.0 + 0.35 * np.sin(2 * math.pi * (u - self.x0) / max(40.0, 0.12 * self.bw) + self.phase2)
            n = self.noise_a(u / (0.5 * self.bw), v / (0.5 * self.bh), 0.8 * f * t)
            dx = A * a * (0.75 * np.sin(ph) * fold + 0.35 * n)
            dy = 0.12 * A * a * np.cos(ph)
            return dx, dy
        if self.effect == "calm_water":
            sx, sz, near = self._calm_slopes(u, v, t)
            return A * near * sx / 1.35, 0.5 * A * near * sz / 1.35
        if self.effect == "canopy_flutter":
            dx, dy, _ = self._canopy(u, v, t)
            return dx, dy
        if self.effect == "water_ripple":
            return min(max(30.0, 0.08 * self.bw), max(10.0, 0.03 * self.bh)) * 0.5
        if self.effect == "foliage_sway":
            return min(0.35 * self.bw, 0.35 * self.bh)
        if self.effect == "fire_flicker":
            return self.flame_px()                                                   # flame tongues
        return 0.0                                                                   # calm_water: exact

    # ---- fire_flicker (modules/perceptual/fire.py): this region's feather is the glow (1 on the
    # flames, fading out around them); self.flame (set by load_regions) is the flames alone.
    def flame_px(self) -> float:
        return max(3.0, 0.18 * getattr(self, "flame_h", self.bh))

    def _fire(self, u, v, t):
        """Fire (reviews 2026-10-21): the v1 licking tongues read best; "only one kind of motion"
        was the fault. Layers: tongues licking up and travelling up the flames; the fire surging
        (growing / shrinking) over a second or two; a slight lean with the draft; a fast small
        flutter at the flame edges; brightness flicker rising through the flames, a quick
        sparkle, the fire's pulse, and a soft glow on its surroundings. Structure inside the
        fire's box (walls, glass frame, grate, glass reflections) is held still (load_regions)."""
        fw = sample_bilinear(self.flame, u, v)
        plume = getattr(self, "plume", None)
        pw = sample_bilinear(plume, u, v) if plume is not None else fw
        fh, fb = self.flame_h, self.flame_bottom
        s = np.clip((fb - v) / max(fh, 1.0), 0.0, 1.0) ** 0.8          # still at the base (logs)
        lam = self.flame_px()
        rise = 0.9 * fh * t                                             # tongues travel upward
        A = self.amp
        surge = 0.75 + 0.35 * float(self.noise_b(1.7, 0.2, 0.6 * t))    # the fire grows / shrinks
        lean = 0.15 * math.sin(2 * math.pi * 0.31 * t + self.phase)     # the draft
        n1 = self.noise_a(u / lam, (v + rise) / lam, 1.3 * t)
        n2 = self.noise_b(u / lam + 7.0, (v + rise) / lam, 1.7 * t)
        ff = 0.5 * lam                                                  # fast flutter, finer
        n3 = self.noise_a(u / ff + 11.0, (v + 1.4 * rise) / ff, 4.0 * t)
        n4 = self.noise_b(u / ff + 23.0, (v + 1.4 * rise) / ff, 4.0 * t)
        dx = A * (0.4 * n1 + lean + 0.15 * n4) * s * pw
        dy = -A * (0.5 * surge * (1.0 + n2) + 0.2 * n3) * s * pw         # lick up, surge, flutter
        whole = float(self.noise_b(0.3, 0.7, 2.4 * t))                  # the fire's overall flicker
        local = self.noise_a(u / lam + 3.0, (v + 1.6 * rise) / lam, 3.0 * t)
        sparkle = self.noise_b(u / ff + 5.0, (v + 2.0 * rise) / ff, 6.0 * t)
        b = (1.0 + fw * (0.28 * local + 0.10 * whole + 0.08 * sparkle + 0.08 * (surge - 0.75))
             + (1.0 - fw) * FIRE_GLOW * whole)
        return dx, dy, b

    def brightness(self, u, v, t):
        """Multiplicative brightness inside the region, or None (displacement-only effects)."""
        return self.fields(u, v, t)[2]

    def fields(self, u, v, t, irregular=None):
        """(dx, dy, brightness or None) at source points; calm_water's waves are computed once
        (irregular: calm_irregular at these points when already known)."""
        if self.effect == "fire_flicker":
            return self._fire(u, v, t)
        if self.effect == "canopy_flutter":
            return self._canopy(u, v, t)
        if self.effect != "calm_water":
            dx, dy = self.displacement(u, v, t)
            return dx, dy, None
        sx, sz, near = self._calm_slopes(u, v, t, irregular)
        return (self.amp * near * sx / 1.35, 0.5 * self.amp * near * sz / 1.35,
                1.0 + CALM_BRIGHTNESS * np.tanh(_SHADE_Z * sz + _SHADE_X * sx))

    def bend(self, u, v):
        """Trees: 0.3 at the crown's base .. 1 at its top, per column (col_bottom / col_span set
        by load_regions; without them, 1)."""
        cb = getattr(self, "col_bottom", None)
        if cb is None:
            return np.ones_like(u)
        i = np.clip(u.astype(np.int64), 0, cb.shape[0] - 1)
        hh = np.clip((cb[i] - v) / np.maximum(self.col_span[i], 1.0), 0.0, 1.0)
        return 0.3 + 0.7 * hh ** 1.5

    def anchor_weight(self, v):
        if self.anchor == "none":           # water: the whole surface moves
            return np.ones_like(v), np.ones_like(v)
        if self.anchor == "top":            # hangs from a rod: still at the top, freest at the hem
            s = np.clip((v - self.y0) / self.bh, 0.0, 1.0)
            return s ** 1.3, s
        s = np.clip((self.y0 + self.bh - v) / self.bh, 0.0, 1.0)   # rooted: still at the base
        return s ** 1.5, s

    def displacement(self, u, v, t):
        """(dx, dy) in source px at source points (u, v), time t seconds."""
        a, s = self.anchor_weight(v)
        A, f = self.amp, self.freq
        if self.effect == "fabric_sway":
            ph = 2 * math.pi * f * t - 2 * math.pi * 0.6 * s + self.phase
            fold = 1.0 + 0.35 * np.sin(2 * math.pi * (u - self.x0) / max(40.0, 0.12 * self.bw) + self.phase2)
            n = self.noise_a(u / (0.5 * self.bw), v / (0.5 * self.bh), 0.8 * f * t)
            dx = A * a * (0.75 * np.sin(ph) * fold + 0.35 * n)
            dy = 0.12 * A * a * np.cos(ph)
            return dx, dy
        if self.effect == "calm_water":
            sx, sz, near = self._calm_slopes(u, v, t)
            return A * near * sx / 1.35, 0.5 * A * near * sz / 1.35
        if self.effect == "canopy_flutter":
            # review 2026-10-16: leaf jitter at 10 px read as jelly; 2026-10-19: branch-scale
            # noise in x AND y read as a circular warp. Wind bends the crown SIDEWAYS as one:
            # a sway travelling across the tree, gusts swelling and fading, branch masses
            # lagging a little behind it, a slight dip as it bends, leaf shimmer on top.
            # 2026-10-19 ("can look more natural"): a tree bends from its base, so the top of the
            # crown sways most (bend); each branch cluster also sways on its own, a little quicker
            # and at its own phase (a static phase field), so the crown never moves as one sheet;
            # the leaves shimmer in light (fields(): LEAF_SHIMMER)
            lb, ll = self.branch_px(), self.leaf_px()
            ab, al = self.canopy_amps()
            span = max(self.bw, 1.0)
            ph = (2 * math.pi * SWAY_HZ * t - 2 * math.pi * SWAY_ACROSS * (u - self.x0) / span
                  + self.phase)
            sway = 0.8 * np.sin(ph) + 0.2 * np.sin(1.63 * ph + self.phase2)
            gust = 0.6 + 0.4 * np.clip(0.5 + self.noise_b(u / (0.8 * span) - 0.1 * t, v / (0.8 * span),
                                                          0.07 * t), 0, 1)
            bend = self.bend(u, v)
            # branch phase: changes slowly across the crown (a branch cluster ~1.5 lb across);
            # faster phase changes shear neighbouring leaves apart
            bph = math.pi * self.noise_a(u / (1.5 * lb) + 60.0, v / (1.5 * lb), 0.0)
            branch = np.sin(2 * math.pi * BRANCH_HZ * t + bph)
            dx = gust * ab * bend * (0.7 * sway + 0.3 * branch) + al * self.noise_b(u / ll, v / ll, f * t)
            dy = (gust * ab * bend * (0.06 * sway * sway + 0.08 * np.cos(2 * math.pi * BRANCH_HZ * t + bph))
                  + 0.6 * al * self.noise_b(u / ll + 40.0, v / ll, f * t))
            return dx, dy
        if self.effect == "water_ripple":
            sx, sy = max(30.0, 0.08 * self.bw), max(10.0, 0.03 * self.bh)   # wide, flat ripples
            n1 = self.noise_a(u / sx + 0.35 * f * t, v / sy, 0.5 * f * t)
            n2 = self.noise_b(u / (0.5 * sx) - 0.5 * f * t, v / (0.5 * sy), 0.9 * f * t)
            dx = A * a * (0.65 * n1 + 0.35 * n2)
            dy = 0.35 * A * a * self.noise_b(u / sx + 20.0, v / sy, 0.7 * f * t)
            return dx, dy
        # foliage_sway
        sway = np.sin(2 * math.pi * f * t + self.phase + 0.5 * s)
        n1 = self.noise_a(u / (0.35 * self.bw), v / (0.35 * self.bh), 0.6 * f * t)
        flutter = self.noise_b(u / 18.0, v / 18.0, 2.2 * f * t)
        dx = A * a * (0.7 * sway + 0.3 * n1 + 0.25 * flutter)
        dy = A * a * 0.15 * self.noise_b(u / (0.35 * self.bw) + 50.0, v / (0.35 * self.bh), 0.6 * f * t)
        return dx, dy


def _leafiness(crop: np.ndarray, soften: bool = True) -> np.ndarray:
    """0..1 from the photo: foliage hue (yellow-green to green-cyan) with some saturation and
    light, scaled by fine texture (leaves are busy; trunks and sky are smooth), softened 2 px
    (unsoftened for finding thin solid objects: softening lets green bleed over a lantern rod)."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, sat, val = hsv[..., 0], hsv[..., 1] / 255, hsv[..., 2] / 255
    w = (np.clip(1 - np.abs(h - 55) / 40, 0, 1) ** 0.5            # OpenCV hue: green = 60
         * np.clip((sat - 0.12) / 0.15, 0, 1) * np.clip((val - 0.06) / 0.08, 0, 1))
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tex = cv2.GaussianBlur(np.abs(cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 1.0), cv2.CV_32F)), (0, 0), 3)
    tex_w = np.clip(tex / (np.percentile(tex, 70) + 1e-6), 0, 1)
    w = (w * (0.4 + 0.6 * tex_w)).astype(np.float32)
    return np.clip(1.5 * (cv2.GaussianBlur(w, (0, 0), 2.0) if soften else w), 0, 1)


def _big_parts(mask: np.ndarray, min_extent: float, by_height: bool = False) -> np.ndarray:
    """The connected parts of a binary mask at least min_extent px across (or tall)."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    size = (st[1:, cv2.CC_STAT_HEIGHT] if by_height
            else np.maximum(st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT]))
    keep[1:] = size >= min_extent
    return keep[lab]


def _guard(solid: np.ndarray, reach: float) -> np.ndarray:
    """0 on and near solid pixels, rising to 1 at `reach` px: a pixel there that moves by up to
    its own distance never samples (drags) the solid object."""
    if not solid.any():
        return np.ones(solid.shape, np.float32)
    dist = cv2.distanceTransform((~solid).astype(np.uint8), cv2.DIST_L2, 5)
    x = np.clip((dist - 1.0) / max(reach, 1.0), 0, 1)
    return (x * x * (3 - 2 * x)).astype(np.float32)


def _crop_box(feather: np.ndarray, margin: int):
    ys, xs = np.nonzero(feather > 0)
    if len(ys) == 0:
        return None
    H, W = feather.shape
    return (max(0, ys.min() - margin), min(H, ys.max() + 1 + margin),
            max(0, xs.min() - margin), min(W, xs.max() + 1 + margin))


HEDGE_BOTTOM = 0.55         # a short mask part ending below this share of the photo height is a
#                             hedge / bush / garden bed (still); a short part wholly in the upper
#                             part of the photo is a distant tree line (moves). 0.72 caught the
#                             hedges in front of a house (review 2026-10-19)
BASE_HOLD = (0.3, 0.12)     # share of each tree column held still at its base (near, far trees)


POT_REACH_BOTTOM = 0.15     # a pot: a non-green part reaching the lowest 15% of a plant's mask,
POT_MAX_SHARE = 0.6         # at most this share of it


def plant_weight(src: np.ndarray, feather: np.ndarray, reach: float) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Potted plants (review 2026-10-21: "we are moving the pots alongside the plants" - the
    segmentation mask covers plant AND pot). The pot (basket, planter, stand) is a non-green part
    of the mask reaching its bottom; it is held still, with the foliage near it held back by the
    motion's reach so nothing drags it. Flowers are not green either, but sit up in the plant:
    they keep moving. Returns (weight, pot mask, the pot's top row or None) - the plant sways
    from the pot's rim, not from the bottom of the pot."""
    box = _crop_box(feather, int(reach) + 20)
    if box is None:
        return feather, np.zeros(feather.shape, bool), None
    y0, y1, x0, x1 = box
    crop, m = src[y0:y1, x0:x1], feather[y0:y1, x0:x1]
    inside = m > 0
    ys = np.nonzero(inside.any(axis=1))[0]
    bottom, top = ys.max(), ys.min()
    nonleaf = (_leafiness(crop, soften=False) < 0.15) & inside
    nonleaf = cv2.morphologyEx(nonleaf.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(nonleaf, connectivity=8)
    low = bottom - POT_REACH_BOTTOM * (bottom - top + 1)
    pot = np.zeros(inside.shape, bool)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_TOP] + st[i, cv2.CC_STAT_HEIGHT] - 1 >= low and \
                max(st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]) >= max(12.0, 2.0 * reach):
            pot |= lab == i
    # a pot is a minority of the mask in its lower part; a non-green part that is most of the
    # mask or reaches its top is the plant itself (dried stems, coloured foliage, a dark photo)
    if pot.any():
        rows = np.nonzero(pot.any(axis=1))[0]
        if pot.sum() > POT_MAX_SHARE * inside.sum() or rows.min() < top + 0.3 * (bottom - top + 1):
            pot[:] = False
    if pot.any():                                  # the pot is solid below its rim
        rows = np.nonzero(pot.any(axis=1))[0]
        pot_top = rows.min()
        pot[pot_top:] |= inside[pot_top:] & (cv2.dilate(pot.astype(np.uint8), np.ones((1, 15), np.uint8)) > 0)[pot_top:]
    w = m * _guard(pot, 1.5 * reach + 2.0)
    out = np.zeros_like(feather)
    out[y0:y1, x0:x1] = w
    full = np.zeros(feather.shape, bool)
    full[y0:y1, x0:x1] = pot
    return out, full, (float(y0 + np.nonzero(pot.any(axis=1))[0].min()) if pot.any() else None)


def tree_columns(inside: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per photo column: the tree mask's bottom row and height, smoothed across ~2% of the width
    (float32): where each part of a crown bends from."""
    W = inside.shape[1]
    any_col = inside.any(axis=0)
    top = np.where(any_col, inside.argmax(axis=0), 0).astype(np.float32)
    bottom = np.where(any_col, inside.shape[0] - 1 - inside[::-1].argmax(axis=0), 0).astype(np.float32)
    k = max(3, int(0.02 * W) | 1)
    smooth = lambda a: cv2.GaussianBlur(a[None, :], (k, 1), 0)[0]    # noqa: E731
    span = np.where(any_col, bottom - top, 0.0).astype(np.float32)
    return smooth(bottom).astype(np.float32), smooth(span).astype(np.float32)


def leaf_weight(src: np.ndarray, feather: np.ndarray, reach: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Tree regions (canopy_flutter), reviews 2026-10-16 / 2026-10-19. Returns (weight, solid):
    - leaves move (the leaf measure: foliage hue, texture); trunks, branches and sky in the mask
      do not;
    - SOLID objects stay rigid: lanterns, poles, railings, trunks and walls inside or next to the
      mask - non-leaf, non-sky parts at least max(30, 4 x reach) px across (smaller non-leaf
      bits are gaps between leaves). Foliage within `reach` (the motion distance) of a solid
      object is held, so no moving pixel samples it and nothing bends;
    - hedges and bushes stay still (user decision): short mask parts (under 12% of the photo
      high) standing low in the photo (HEDGE_BOTTOM); short parts higher up are distant tree
      lines and move; the base of each tree column is held (BASE_HOLD: trunks, undergrowth)."""
    box = _crop_box(feather, int(reach) + 40)
    if box is None:
        return feather, np.zeros(feather.shape, bool)
    y0, y1, x0, x1 = box
    crop, m = src[y0:y1, x0:x1], feather[y0:y1, x0:x1]
    leaf = _leafiness(crop)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, sat, val = hsv[..., 0], hsv[..., 1] / 255, hsv[..., 2] / 255
    sky = (val > 0.55) & ((((h > 85) & (h < 130)) & (sat > 0.08)) | ((sat < 0.15) & (val > 0.75)))
    cand = cv2.morphologyEx(((_leafiness(crop, soften=False) < 0.15) & ~sky).astype(np.uint8),
                            cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    solid = _big_parts(cand, max(30.0, 4.0 * reach))
    # (2026-10-19, reverted the same day: treating a large busy non-green part as foliage, and a
    # 0.75 weight floor for the whole mask, bent lantern frames, poles and thin pine trunks -
    # "the trees are so messed up". The leaf measure decides what moves.)
    w = m * leaf * _guard(solid, 1.5 * reach + 2.0)
    inside = m > 0
    H = src.shape[0]
    hedge_h = 0.12 * H
    n, lab, st, _ = cv2.connectedComponentsWithStats(inside.astype(np.uint8), connectivity=8)
    bottom_px = y0 + st[:, cv2.CC_STAT_TOP] + st[:, cv2.CC_STAT_HEIGHT]
    far_part = (bottom_px < HEDGE_BOTTOM * H)
    hedge_part = (st[:, cv2.CC_STAT_HEIGHT] < hedge_h) & ~far_part
    far_part[0] = hedge_part[0] = False
    w[hedge_part[lab]] = 0.0                                           # bushes, hedges: still
    far = far_part[lab] & inside
    # each column of the tree: still at its base (hedge rows, trunks, undergrowth), moving in the
    # crown; a column only a hedge tall stays still. The column profiles are smoothed across
    # ~2% of the width so neighbouring columns never differ sharply (no shearing streaks).
    rows = np.arange(inside.shape[0])[:, None]
    any_col = inside.any(axis=0)
    top = np.where(any_col, inside.argmax(axis=0), inside.shape[0]).astype(np.float32)
    bottom = np.where(any_col, inside.shape[0] - 1 - inside[::-1].argmax(axis=0), 0).astype(np.float32)
    k = max(3, int(0.02 * src.shape[1]) | 1)
    smooth = lambda a: cv2.GaussianBlur(a[None, :], (k, 1), 0)[0]    # noqa: E731
    span = np.where(any_col, bottom - top, 0.0).astype(np.float32)
    bottom_s, span_s = smooth(bottom), smooth(span)
    x = np.clip((bottom_s[None, :] - rows) / (BASE_HOLD[0] * np.maximum(span_s[None, :], 1.0)), 0, 1)
    xf = np.clip((bottom_s[None, :] - rows) / (BASE_HOLD[1] * np.maximum(span_s[None, :], 1.0)), 0, 1)
    tall = np.clip((span_s - 0.8 * hedge_h) / (0.5 * hedge_h), 0, 1)
    near_w = x * x * (3 - 2 * x) * (tall * tall * (3 - 2 * tall))[None, :]
    far_w = xf * xf * (3 - 2 * xf)                     # a distant tree line: only its foot held
    w *= np.where(far, far_w, near_w).astype(np.float32)
    out = np.zeros_like(feather)
    out[y0:y1, x0:x1] = w
    full_solid = np.zeros(feather.shape, bool)
    full_solid[y0:y1, x0:x1] = solid
    return out, full_solid


def _touching(parts: np.ndarray, edge: np.ndarray, within: float) -> np.ndarray:
    """The connected parts of `parts` that come within `within` px of `edge`."""
    if not parts.any() or not edge.any():
        return np.zeros_like(parts) if not edge.any() else parts
    near = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 5) <= within
    n, lab = cv2.connectedComponents(parts.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[near & parts])] = True
    keep[0] = False
    return keep[lab]


def water_guard(src: np.ndarray, feather: np.ndarray, reach: float) -> tuple[np.ndarray, np.ndarray]:
    """Water regions: dark grey solid things reaching the water's edge (railing bars in front of a
    pool, dock posts) and the water's own edge stay rigid, like solid objects in trees
    (leaf_weight): no moving pixel reaches them. Submerged things (steps, benches, lights) ripple."""
    box = _crop_box(feather, int(reach) + 40)
    if box is None:
        return feather, np.zeros(feather.shape, bool)
    y0, y1, x0, x1 = box
    hsv = cv2.cvtColor(src[y0:y1, x0:x1], cv2.COLOR_BGR2HSV).astype(np.float32)
    dark_grey = ((hsv[..., 2] / 255 < 0.3) & (hsv[..., 1] / 255 < 0.4)).astype(np.uint8)
    solid = _big_parts(cv2.morphologyEx(dark_grey, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)),
                       max(30.0, 4.0 * reach))
    # (no pool-structure rule: COCO's water masks are the water surface only; ADE20K's "pool"
    # covered coping and tiled walls, and rules to carve them out also froze submerged steps -
    # review 2026-10-19, COCO is the primary model again)
    # the water's own edge too: ripples next to the pool's coping must not pull the coping in
    # (Turkey Trot pool, 2026-10-16: 0.3% of moving pixels dragged it, gate limit 0.2%)
    edge = feather[y0:y1, x0:x1] <= 0
    # the ripples' reach per row: calm_water moves a row by amp x near(v) (perspective: far rows
    # barely move), so the held band along the edge narrows with it (review 2026-10-19: a thin
    # distant lake lost most of its surface to a band sized for the near water)
    ys = np.nonzero((feather > 0).any(axis=1))[0]
    bh = float(ys.max() - ys.min() + 1)
    v_h = ys.min() - 0.9 * bh
    near = np.clip((np.arange(y0, y1, dtype=np.float32) - v_h) / (ys.max() + 1 - v_h), 0.0, 1.0)
    row_reach = (1.5 * reach * near + 2.0)[:, None]
    # ...but only structure that reaches the water's edge is held: coping, walls above the
    # waterline and railing bars meet or cross it; submerged steps, benches and lights are
    # surrounded by water and ripple with it (review 2026-10-19: the pool steps were held still)
    solid = _touching(solid, edge, max(8.0, 2.0 * reach))
    out = feather.copy()
    held = solid | edge
    if held.any():
        dist = cv2.distanceTransform((~held).astype(np.uint8), cv2.DIST_L2, 5)
        g = np.clip((dist - 1.0) / np.maximum(row_reach, 1.0), 0, 1)
        out[y0:y1, x0:x1] *= (g * g * (3 - 2 * g)).astype(np.float32)
    full_solid = np.zeros(feather.shape, bool)
    full_solid[y0:y1, x0:x1] = solid
    return out, full_solid


def flame_plume(flame: np.ndarray, height: float) -> np.ndarray:
    """The flames smeared upward by `height` px, fading with height: where the tongues may lick
    above the tips the photo caught (0..1)."""
    out = (flame > 0).astype(np.float32)
    base = out.copy()
    n = 12
    for i in range(1, n + 1):
        sh = int(round(i * height / n))
        if sh <= 0 or sh >= base.shape[0]:
            continue
        up = np.zeros_like(base)
        up[:-sh] = base[sh:]
        out = np.maximum(out, (1.0 - i / (n + 1)) * up)
    return cv2.GaussianBlur(out, (0, 0), 2.0).clip(0, 1).astype(np.float32)


def fire_box(src: np.ndarray, flame: np.ndarray) -> np.ndarray:
    """Where a fire may move (0..1): the flames plus the dark cavity connected to them, within a
    box FIREBOX_SIDE of the flames' width either side and FIRE_PLUME of their height above (down
    to their base). The mantel, surround and hearth stay outside."""
    from modules.perceptual.fire import CAVITY_V
    f = flame > 0
    ys, xs = np.nonzero(f)
    H, W = f.shape
    fh, fw = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    y0, y1 = max(0, int(ys.min() - FIRE_PLUME * fh)), min(H, int(ys.max()) + 2)
    x0, x1 = max(0, int(xs.min() - FIREBOX_SIDE * fw)), min(W, int(xs.max() + FIREBOX_SIDE * fw) + 1)
    v = cv2.cvtColor(src[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)[..., 2].astype(np.float32) / 255
    cand = ((v < CAVITY_V) | f[y0:y1, x0:x1]).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab = cv2.connectedComponents(cand, connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[f[y0:y1, x0:x1]])] = True
    keep[0] = False
    out = np.zeros((H, W), np.float32)
    out[y0:y1, x0:x1] = keep[lab]
    return out


def fire_structure(src: np.ndarray, flame: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Inside a fire's box, what is not fire (bool): everything that is neither flame nor dark
    empty space - firebox walls and back panels (herringbone, brick), logs, glass reflections,
    the frame - plus hard straight edges even when dark (a grate, andirons, frame bars).
    Review 2026-10-21: side views warped the walls; behind glass the reflections slid with the
    flames. The flames themselves (and a few px around them) never count."""
    out = np.zeros(flame.shape, bool)
    ys, xs = np.nonzero(box > 0.5)
    if not len(ys):
        return out
    pad = 6
    y0, y1 = max(0, ys.min() - pad), min(flame.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(flame.shape[1], xs.max() + 1 + pad)
    crop = src[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat, val = hsv[..., 1] / 255, hsv[..., 2] / 255
    g = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.0)
    edge = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    hard = edge > STRUCT_EDGE * 255
    # long straight-ish runs only: walls, frames, bars (flames make short broken edges)
    lines = np.zeros_like(hard, np.uint8)
    k = max(9, int(0.15 * max(1, min(y1 - y0, x1 - x0))))
    for kern in (np.ones((1, k), np.uint8), np.ones((k, 1), np.uint8)):
        lines |= cv2.morphologyEx(hard.astype(np.uint8), cv2.MORPH_OPEN, kern)
    # faint but LONG straight runs too: a screen rod or wire, a door frame's bar - dark on the
    # dark cavity, so low contrast (review 2026-10-21: a wire across a brick fireplace's
    # opening warped). Flames never make long straight lines, so length is the test.
    # Only ABOVE the fire: among the logs, long faint edges are the logs themselves.
    faint = (edge > STRUCT_FAINT_EDGE * 255).astype(np.uint8)
    kh = max(15, int(STRUCT_LONG * (x1 - x0)))
    kv = max(15, int(STRUCT_LONG * (y1 - y0)))
    faint_lines = (cv2.morphologyEx(faint, cv2.MORPH_OPEN, np.ones((1, kh), np.uint8))
                   | cv2.morphologyEx(faint, cv2.MORPH_OPEN, np.ones((kv, 1), np.uint8)))
    # only flames and dark empty space may move: anything lit that is not flame - a back wall's
    # herringbone or brick, logs, glass reflections (grey, often dim), the frame - is held
    lit = val >= STRUCT_LIT_V
    # the flame PIXELS (flame-coloured and bright inside the fire's zone - the zone itself is
    # closed over the gaps between tongues, where the back wall shows through)
    from modules.perceptual.fire import FLAME_HUE, FLAME_SAT, FLAME_VAL
    hue = hsv[..., 0]
    tongues = ((flame[y0:y1, x0:x1] > 0) & (hue >= FLAME_HUE[0]) & (hue <= FLAME_HUE[1])
               & (sat > FLAME_SAT) & (val > FLAME_VAL))
    near_flame = cv2.dilate(tongues.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    # lit non-flame pixels are held only ABOVE the fire (the back wall, glass reflections over
    # the flames): logs and what shows between the tongues may move with the fire (2026-10-21:
    # holding them too left some fires only flickering in light)
    fz = flame[y0:y1, x0:x1] > 0
    fy = np.nonzero(fz.any(axis=1))[0]
    above = np.zeros_like(lit)
    if len(fy):
        above[: max(0, int(fy.min() + STRUCT_ABOVE * (fy.max() - fy.min() + 1)))] = True
    st = ((lines > 0) | ((lit | (faint_lines > 0)) & above & ~fz)) & ~near_flame
    st = cv2.dilate(st.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    out[y0:y1, x0:x1] = st & (box[y0:y1, x0:x1] > 0.5)
    return out


def fire_glow(flame: np.ndarray) -> np.ndarray:
    """1 on the flames, a soft warm halo around them (about the fire's own size), 0 beyond."""
    ys, xs = np.nonzero(flame > 0)
    size = max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)
    sigma = max(4.0, 0.6 * size)
    halo = cv2.GaussianBlur((flame > 0).astype(np.float32), (0, 0), sigma)
    halo = np.clip(halo / max(float(halo.max()), 1e-6) * 2.0, 0, 1)
    halo[halo < 0.01] = 0.0                    # no faint tail across the whole photo
    return np.maximum(flame, halo).astype(np.float32)


def _fire_zone_file(path: str, src_w: int, src_h: int):
    """Flame tongues -> their filled zone (modules/perceptual/fire.flame_zone), written next to
    the mask; returns (zone path, its bbox normalised). Masks from before the zone existed work."""
    from modules.perceptual.fire import flame_zone
    m = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (src_w, src_h), interpolation=cv2.INTER_NEAREST)
    z = flame_zone(m)
    zp = path[:-4] + ".zone.png"
    cv2.imwrite(zp, z)
    ys, xs = np.nonzero(z)
    if not len(ys):
        return path, (0.0, 0.0, 1.0, 1.0)
    return zp, (xs.min() / src_w, ys.min() / src_h, (xs.max() + 1) / src_w, (ys.max() + 1) / src_h)


def load_regions(shot, images_dir: str, src_w: int, src_h: int,
                 src: np.ndarray | None = None) -> tuple[list[Region], list[str]]:
    """Regions whose mask was fetched (images/masks/<shot>_<id>.png) and fits the photo.
    src (the photo, BGR): tree regions are narrowed to their leaves with solid objects and hedges
    held still (leaf_weight); water regions hold solid dark objects still (water_guard)."""
    out, warnings = [], []
    for spec in shot.motion_regions or []:
        path = os.path.join(images_dir, mask_relpath(shot.shot_id, spec.id))
        x1, y1, x2, y2 = spec.bbox
        if spec.effect == "fire_flicker" and os.path.exists(path):
            path, (x1, y1, x2, y2) = _fire_zone_file(path, src_w, src_h)
        feather, reason = (load_feather(path, src_w, src_h, (x1 * src_w, y1 * src_h, x2 * src_w, y2 * src_h))
                           if os.path.exists(path) else (None, "missing"))
        if feather is None:
            warnings.append(f"motion_region_dropped:{shot.shot_id}:{spec.id}:{reason}")
            continue
        solid = cols = pot_top = None
        if src is not None and spec.effect == "canopy_flutter":
            cols = tree_columns(feather > 0)
            feather, solid = leaf_weight(src, feather, float(spec.amplitude_px))
        elif src is not None and spec.effect == "foliage_sway":
            feather, solid, pot_top = plant_weight(src, feather, float(spec.amplitude_px))
        elif src is not None and spec.effect == "calm_water":
            feather, solid = water_guard(src, feather, float(spec.amplitude_px))
        flame = plume = None
        if spec.effect == "fire_flicker":
            ys = np.nonzero((feather > 0).any(axis=1))[0]
            fh = float(ys.max() - ys.min() + 1)
            plume = flame_plume(feather, FIRE_PLUME * fh)
            if src is not None:
                # inside the fire's own box only, held back from its edge and from the structure
                # inside it by the motion's reach, so no moving pixel samples the mantel, the
                # firebox walls, the glass frame or the glass's reflections
                box = fire_box(src, feather)
                amp = min(float(spec.amplitude_px), FIRE_AMP_OF_HEIGHT * fh)
                struct = fire_structure(src, feather, box)
                # the tongues lick UP (sampling from below) and only ~0.55 x amp sideways:
                # what lies above or beside them needs a margin of the sideways reach only
                reach = 0.7 * amp + 2.0
                plume *= _guard((box < 0.5) | struct, reach)
                solid = ((box < 0.5) | struct) & (cv2.dilate((plume > 0).astype(np.uint8),
                                                             np.ones((9, 9), np.uint8)) > 0)
            flame, feather = feather, np.maximum(fire_glow(feather), plume)
        region = Region(spec, feather, src_w, src_h)
        region.solid = solid                       # for the quality gate's drag check
        if cols is not None:
            region.col_bottom, region.col_span = cols
        if pot_top is not None and pot_top > region.y0 + 0.2 * region.bh:
            region.bh = pot_top - region.y0                # sway from the pot's rim
        if flame is not None:
            ys, xs = np.nonzero(flame > 0)
            region.flame, region.flame_h = flame, float(ys.max() - ys.min() + 1)
            region.plume = plume
            region.amp = min(region.amp, FIRE_AMP_OF_HEIGHT * region.flame_h)
            region.flame_bottom = float(ys.max())
            gy, gx = np.nonzero(feather > 1e-3)     # the region spans its glow, not just the flames
            region.x0, region.y0 = float(gx.min()), float(gy.min())
            region.bw, region.bh = float(gx.max() - gx.min() + 1), float(gy.max() - gy.min() + 1)
        out.append(region)
    return out, warnings


def mask_relpath(shot_id: str, region_id: str) -> str:
    return os.path.join("masks", f"{shot_id}_{region_id}.png")


# Fields are smooth, so they are computed on a coarse grid of output pixels (every FIELD_STEP px)
# and interpolated bilinearly (exactly, also over the uneven last cell) to every pixel; the mask
# weight itself stays per pixel. step=1 is the exact per-pixel path. Accuracy against step=1 is
# measured by tools/perceptual_quality.py and held by tests (tests/test_perceptual.py).
# (rows, cols) grid spacing per effect, an upper limit: per frame the grid is also chosen from the
# effect's strength and main detail as seen on screen (adaptive_step: predicted error within
# FIELD_ERR_BUDGET). Steps stay in {1, 2, 4} so windows align to 4.
FIELD_STEP = {"canopy_flutter": (2, 2),     # small leaf clusters: a finer grid
              "calm_water": (1, 1)}         # far waves are a few px apart: no grid can hold them
DEFAULT_FIELD_STEP = (4, 4)
# calm_water's slow strength variation (calm_irregular) on a coarse grid; the waves stay exact
SMOOTH_STEP = {"calm_water": (2, 2)}   # 4 px left 0.35 px max error on far water (Turkey Trot pool)
_INTERP_CACHE: dict = {}


def field_step(effect: str) -> tuple[int, int]:
    return FIELD_STEP.get(effect, DEFAULT_FIELD_STEP)


def _coarse_grid(h: int, w: int, step: tuple[int, int]):
    """Coarse sample rows/cols (always incl. the last pixel) and full-res maps into that grid."""
    key = (h, w, step)
    if key not in _INTERP_CACHE:
        ys, xs = _sample_grid(h, w, *step)
        fx = np.interp(np.arange(w), xs, np.arange(len(xs))).astype(np.float32)
        fy = np.interp(np.arange(h), ys, np.arange(len(ys))).astype(np.float32)
        _INTERP_CACHE[key] = (ys, xs, *np.meshgrid(fx, fy))
    return _INTERP_CACHE[key]


ROI_PROBE = 8      # px: grid used to find where on screen a region can be


def _sample_grid(h: int, w: int, step_y: int, step_x: int):
    return (np.unique(np.r_[np.arange(0, h, step_y), h - 1]), np.unique(np.r_[np.arange(0, w, step_x), w - 1]))


def _cell_px(cu: np.ndarray, cv: np.ndarray) -> float:
    """The largest source-px distance between neighbouring grid points."""
    return max([1.0] + [float(np.abs(np.diff(c, axis=ax)).max()) for c in (cu, cv) for ax in (0, 1)
                        if c.shape[ax] > 1])


def _near_bbox(r: Region, cu: np.ndarray, cv: np.ndarray, m: float) -> np.ndarray:
    return (cu >= r.x0 - m) & (cu <= r.x0 + r.bw + m) & (cv >= r.y0 - m) & (cv <= r.y0 + r.bh + m)


def _screen_box(r: Region, map_x, map_y):
    """((r0, r1, c0, c1) rows/cols of these maps that can show the region, or None; screen px per
    source px, at its smallest)."""
    h, w = map_x.shape
    ys, xs = _sample_grid(h, w, ROI_PROBE, ROI_PROBE)
    pu, pv = map_x[np.ix_(ys, xs)], map_y[np.ix_(ys, xs)]
    # the camera's zoom from the median spacing (earlier regions may already displace the maps;
    # the box margins above keep the maximum)
    spacing = max([1e-6] + [float(np.median(np.abs(np.diff(c, axis=ax)))) for c, ax in ((pu, 1), (pv, 0))
                            if c.shape[ax] > 1])
    return probe_box(r, pu, pv, ys, xs, h, w), ROI_PROBE / spacing


def probe_box(r: Region, pu, pv, ys, xs, h: int, w: int):
    """The screen box from maps sampled at probe rows ys / cols xs (at most ROI_PROBE apart). The
    mask is clipped to the region's bbox, so a pixel that moves maps into the bbox; its nearest
    probe point is within one probe cell, so it is flagged with a 2-cell margin and the box
    grows by one probe."""
    hit = _near_bbox(r, pu, pv, 2.0 * _cell_px(pu, pv))
    if not hit.any():
        return None
    rows, cols = np.nonzero(hit.any(axis=1))[0], np.nonzero(hit.any(axis=0))[0]
    return (max(0, int(ys[rows[0]]) - ROI_PROBE), min(h, int(ys[rows[-1]]) + ROI_PROBE + 1),
            max(0, int(xs[cols[0]]) - ROI_PROBE), min(w, int(xs[cols[-1]]) + ROI_PROBE + 1))


def _coarse_eval(r: Region, fn, n_out: int, map_x, map_y, step: tuple[int, int], box, origin, frame_shape):
    """Evaluate fn(u, v) -> n_out arrays (or None entries) on the frame's coarse grid around box
    (r0, r1, c0, c1, frame px) and interpolate to full res. The box is aligned to grid points so
    every pixel has its 4 neighbours inside. Returns (R, [arrays or None]); R: the aligned box as
    slices of these maps (which start at origin). Points farther than a cell from the region's
    bbox (source px) cannot feed a moving pixel: they are left at 0 (1 for the third output)."""
    oy, ox = origin
    ys, xs, gx, gy = _coarse_grid(*frame_shape, step)
    r0, r1, c0, c1 = box
    i0, i1 = np.searchsorted(ys, r0, "right") - 1, np.searchsorted(ys, r1 - 1, "left")
    j0, j1 = np.searchsorted(xs, c0, "right") - 1, np.searchsorted(xs, c1 - 1, "left")
    cu = map_x[np.ix_(ys[i0:i1 + 1] - oy, xs[j0:j1 + 1] - ox)].astype(np.float64)
    cv = map_y[np.ix_(ys[i0:i1 + 1] - oy, xs[j0:j1 + 1] - ox)].astype(np.float64)
    need = _near_bbox(r, cu, cv, 1.5 * _cell_px(cu, cv))
    vals = fn(cu[need], cv[need]) if need.any() else [None] * n_out
    G = (slice(int(ys[i0]), int(ys[i1]) + 1), slice(int(xs[j0]), int(xs[j1]) + 1))
    lx = np.ascontiguousarray(gx[G] - np.float32(j0))
    ly = np.ascontiguousarray(gy[G] - np.float32(i0))
    out = []
    for k, val in enumerate(vals):
        if val is None:
            out.append(None)
            continue
        c = np.full(cu.shape, 1.0 if k == 2 else 0.0, np.float32)
        c[need] = val
        out.append(cv2.remap(c, lx, ly, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
    if not need.any() and n_out >= 2:
        out[:2] = [np.zeros(lx.shape, np.float32)] * 2
    R = (slice(G[0].start - oy, G[0].stop - oy), slice(G[1].start - ox, G[1].stop - ox))
    return R, out


FIELD_ERR_BUDGET = 0.02   # screen px: predicted interpolation error a coarse grid may add


def adaptive_step(r: Region, configured: tuple[int, int], screen_per_src: float) -> tuple[int, int]:
    """The coarsest grid (4, 2 or exact) whose interpolation error stays small: bilinear error
    grows with the strength and with (step / detail size)^2 - est. strength x (step / detail)^2,
    calibrated on Downing / Turkey Trot (curtains at 7 px with step 4 over 48 px detail: 0.06 px
    p99 - too much; trees at 10 px flutter with step 2 over ~25 px leaves: 0.35 px max)."""
    for st in (4, 2):
        if st <= min(configured) and r.step_error(st, screen_per_src) <= FIELD_ERR_BUDGET:
            return (min(configured[0], st), min(configured[1], st))
    return (1, 1)


def field_align(regions: list[Region]) -> int:
    """A window whose bounds are multiples of this (or the frame's last row/col) contains every
    coarse grid point its regions use."""
    steps = [s for r in regions for s in field_step(r.effect) + SMOOTH_STEP.get(r.effect, ())]
    assert all(s in (1, 2, 4) for s in steps), "field steps must be 1, 2 or 4 (window alignment)"
    return int(np.lcm.reduce([max(1, s) for s in steps] or [1]))


def apply_regions(map_x: np.ndarray, map_y: np.ndarray, regions: list[Region], t: float,
                  step: int | tuple[int, int] | None = None, origin: tuple[int, int] = (0, 0),
                  frame_shape: tuple[int, int] | None = None) -> np.ndarray | None:
    """Shift the source sampling maps (output px -> source px) in place so the content inside
    each mask moves by its displacement: sampling from (u - dx) shows the pixel moved by +dx.
    Returns a per-output-pixel brightness gain when any region changes brightness, else None.
    step: field grid in output px (None: per effect, FIELD_STEP; 1: exact per pixel).
    The maps may be a window of the frame starting at origin (rows, cols) of a frame of
    frame_shape, aligned as field_align() says; results per pixel are the same as full-frame.
    Work is limited to the part of the screen where each region can be (memory traffic, not
    arithmetic, is what limits the speed); per pixel the results are the same as full-frame."""
    gain = None
    frame_shape = frame_shape or map_x.shape
    oy, ox = origin
    for r in regions:
        box, screen_per_src = _screen_box(r, map_x, map_y)
        if box is None:
            continue
        if step is None:
            st = adaptive_step(r, field_step(r.effect), screen_per_src)
        else:
            st = (step, step) if isinstance(step, int) else tuple(step)
        gbox = (box[0] + oy, box[1] + oy, box[2] + ox, box[3] + ox)
        smooth = SMOOTH_STEP.get(r.effect) if step is None else None
        irr = None
        if max(st) > 1:
            R, (fdx, fdy, fb) = _coarse_eval(r, lambda u, v: r.fields(u, v, t), 3, map_x, map_y, st, gbox,
                                             origin, frame_shape)
        elif smooth:
            R, (irr,) = _coarse_eval(r, lambda u, v: (r.calm_irregular(u, v, t),), 1, map_x, map_y, smooth,
                                     gbox, origin, frame_shape)
        else:
            R = (slice(box[0], box[1]), slice(box[2], box[3]))
        mx, my = map_x[R], map_y[R]                     # views: writes land in the full maps
        w = cv2.remap(r.feather, np.ascontiguousarray(mx), np.ascontiguousarray(my), cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        idx = w > 1e-3
        if not idx.any():
            continue
        u, v = mx[idx].astype(np.float64), my[idx].astype(np.float64)
        if max(st) <= 1:
            dx, dy, b = r.fields(u, v, t, None if irr is None else irr[idx].astype(np.float64))
        else:
            dx, dy = fdx[idx].astype(np.float64), fdy[idx].astype(np.float64)
            b = fb[idx].astype(np.float64) if fb is not None else None
        ww = w[idx].astype(np.float64)
        if b is not None:
            if gain is None:
                gain = np.ones(map_x.shape, np.float32)
            g = gain[R]
            g[idx] *= (1.0 + ww * (b - 1.0)).astype(np.float32)
        mx[idx] = (u - ww * dx).astype(np.float32)
        my[idx] = (v - ww * dy).astype(np.float32)
    return gain
