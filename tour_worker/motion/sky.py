"""Sky motion (effect sky_drift, 2026-10-27): clouds that drift and billow (cloud shadows were
tried and left out - user: "lets not do shadows"; the code stays, off by default).

Unlike the other effects the sky cannot be moved in place: clouds drifting behind a roofline or a
tree need sky the photo never showed. So the sky is its own layer:
  alpha    where the photo shows sky: OneFormer's sky checked pixel by pixel (blue sky, white /
           grey cloud), so leaves, roof edges and thin things stay put. The model's mask is coarse
           and takes in twigs, wires, a weathervane or half a gutter whole (review 2026-10-27:
           "the thinner objects are selected and then moved with the clouds"): anything darker
           than the clear blue sky there that hangs off the foreground is foreground
  plate    the sky continued behind everything else (inpainted: at full resolution just behind
           thin things, at low resolution further in - only a few px of it ever shows, at the
           skyline, as the clouds drift)
  edge     how much of each pixel just outside the sky is sky (a twig's or a vane's soft edge)
  frame(t) the photo plus the change in the sky - the plate warped (drift + billow) minus the
           plate - through alpha, and through edge at the edges: a thing in front keeps its own
           colour and only the sky's part of it changes (else a blue outline stays round a vane
           as a cloud passes behind it); cloud shadows dim the ground and the house below the
           skyline
Everything is a smooth function of t (seeded gradient noise), so frames are deterministic.

Effects (any combination; the tour uses drift + billow):
  drift    the clouds slide with the wind (the shot's wind direction, as the trees): 2% of the
           photo width per second at the top, less towards the horizon (user's pick of 1 / 2 / 3%)
  billow   slow, large-scale churn: clouds subtly change shape
  shadows  soft cloud shadows pass over the ground and the house (off)

In a tour the sky is a layer under everything else (tour_worker/render/frames.py ShotWarp): each
frame's photo is first this composite, then the other regions (trees over the moving sky) and
the camera act on it.
"""
from __future__ import annotations

import cv2
import numpy as np

from tour_worker.motion.noise import GradientNoise3D

EDGE_BAND_PX = 24          # the refined edge: this far either side of OneFormer's sky boundary
SKY_DE = 10.0              # outside the model's sky: within this colour distance (Lab) of the sky there ...
SKY_TEXTURE = 3.0          # ... and this smooth (std of lightness, 5x5): sky between branches, not slate
DARKER_DL = 12.0           # darker than the clear blue sky there by this much (Lab L) ...
NEAR_PX = 48               # ... within this of the foreground and hanging off it: foreground
THIN_PX, THIN_DL = 9, 10.0  # the sky's own colour leaves out detail darker than its 9 px surround
PLATE_INSET_PX = 3         # the plate keeps photo pixels only this far inside the sky (no mixed edge)
FINE_PX = 8                # hidden sky this close to visible sky is inpainted at full resolution
FRINGE_PX = 2              # non-sky pixels this close to the sky can be part sky (a soft edge)
DRIFT_FRAC = 0.02          # drift: the top clouds move this share of the photo width per second
#                            (0.4%/s is real-time speed - invisible in a 4 s shot, review 2026-10-27:
#                            "the cloud movements are not apparent"; ~5x is a gentle time-lapse)
DRIFT_HORIZON = 0.35       # clouds at the horizon move this share of the top clouds' speed
BILLOW_FRAC = 0.0028       # billow: peak displacement, share of the photo width (~4 px at 1536)
BILLOW_SCALE = 0.22        # billow: feature size, share of the width (whole cloud banks churn)
BILLOW_HZ = 0.12
# cloud shadows (review 2026-10-27: "i cant say that i can see cloud shadows" at 10%, 45%-wide
# vague patches): as real ones read - strong on sunlit surfaces, faint on what is already shaded,
# a visible soft edge, sweeping over the ground faster near the camera, a little cooler (lit by
# the blue sky only)
SHADOW_DEPTH = 0.22        # a fully sunlit surface under a cloud shadow: this much darker
SHADOW_SCALE = 0.30        # patch size, share of the width
SHADOW_EDGE = 0.22         # noise band of the soft edge (smaller = crisper)
SHADOW_COVER = 0.0         # noise threshold: about half the ground in shadow at any time
SHADOW_SPEED = 1.5         # x the top clouds' drift, at the skyline ...
SHADOW_NEAR = 2.0          # ... and this many times more at the bottom of the frame (perspective)
SHADOW_TINT = (0.85, 1.0, 1.12)   # B, G, R share of the darkening: shade is a touch blue
GRID = 16                  # noise fields are evaluated every GRID px and upsampled (they are smooth)


def refine_alpha(bgr: np.ndarray, coarse: np.ndarray) -> np.ndarray:
    """Sky coverage per pixel (float32 0..1, feathered inwards only: whatever is not sky is 0).

    Every pixel of the model's sky must look like the sky: not green or dark, and within a band
    round its boundary blue or a bright low-saturation cloud too. OUTSIDE the model's mask only
    what IS the sky there joins (its colour and smoothness): a light-grey chimney passed for a
    cloud and blue-grey slate for blue, and drifted off (samples p00, p01, 2026-10-27).

    The mask is coarse and takes in thin things whole - twigs, loose leaves, a wire, a weathervane,
    the lower half of a gutter - which drifted off with the clouds (review 2026-10-27). So anything
    darker than the clear blue sky nearby that hangs off the foreground is foreground: a twig is
    darker than the blue even where a cloud is behind it, and a grey wisp in a cloud is not."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    s, v = hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    b, g, r = (bgr[..., i].astype(np.float32) for i in range(3))
    blue = (b >= g + 6) & (b >= r + 12) & (v > 0.45)
    cloud = (s < 0.16) & (v > 0.72)
    green = (g > b + 8) & (g > r)
    dark = v < 0.35
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]
    tex = np.sqrt(np.maximum(cv2.blur(L * L, (5, 5)) - cv2.blur(L, (5, 5)) ** 2, 0.0))
    thin_dark = cv2.morphologyEx(L, cv2.MORPH_CLOSE, np.ones((THIN_PX, THIN_PX), np.uint8)) - L > THIN_DL
    k = 2 * EDGE_BAND_PX + 1
    m = (coarse > 127).astype(np.uint8)
    inside = m > 0
    zone = cv2.dilate(m, np.ones((k, k), np.uint8)) > 0
    core = cv2.erode(m, np.ones((k, k), np.uint8)) > 0
    # the colour the sky has at each spot (its surest pixels continued over the rest)
    strict = inside & ~green & ~dark & ~thin_dark & (tex < 6.0) & (blue | cloud | core)
    lab_exp = cv2.cvtColor(_field(bgr, strict), cv2.COLOR_BGR2LAB).astype(np.float32)
    de = np.sqrt(((lab - lab_exp) ** 2).sum(axis=2))
    is_sky_there = (de < SKY_DE) & (tex < SKY_TEXTURE) & ~green
    fg = (~inside & ~is_sky_there) | green | dark
    # the darkest the clear blue gets nearby (haze by a cloud is paler; blue-grey metal is not
    # clear blue - it made itself the reference and passed)
    clear_blue = inside & blue & ~green & (tex < 6.0) & (s > 0.25) & (v > 0.6)
    ref = clear_blue if clear_blue.sum() > 2000 else strict
    L_min = cv2.erode(np.where(ref, L, 255.0).astype(np.float32), np.ones((25, 25), np.uint8))
    L_sky = _field(L_min, L_min < 255.0)
    darker = L < L_sky - DARKER_DL
    near_fg = cv2.dilate(fg.astype(np.uint8), np.ones((2 * NEAR_PX + 1, 2 * NEAR_PX + 1), np.uint8)) > 0
    detail = darker & (near_fg | (v < 0.5))
    # only what hangs off the foreground, across gaps of a few px (a vane's rod fades to sky
    # colour in places); a wire dark enough counts on its own
    linked = cv2.dilate((detail | fg).astype(np.uint8), np.ones((5, 5), np.uint8))
    n, labels = cv2.connectedComponents(linked, connectivity=8)
    hangs = np.zeros(n, bool)
    hangs[np.unique(labels[fg])] = True
    detail = (hangs[labels] & detail) | (detail & (v < 0.5))
    detail = cv2.dilate(detail.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0     # and its fringe
    sky_colour = (blue | cloud) & ~darker
    a = (core & ~green & ~dark) | (zone & inside & sky_colour & ~green) | (zone & ~inside & is_sky_there)
    a = cv2.morphologyEx((a & ~detail).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # specks in the trees are not sky, unless they are the model's sky between twigs - the sky's
    # colour there, which a cloud passing behind must fill (lighter flecks of a slate roof pass
    # for blue, but not for the sky's own colour)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(a, connectivity=8)
    area = stats[:, cv2.CC_STAT_AREA]
    in_model = np.bincount(labels.ravel(), weights=inside.ravel().astype(np.float64), minlength=n)
    like_sky = np.bincount(labels.ravel(), weights=(de < SKY_DE).ravel().astype(np.float64), minlength=n)
    keep = (area >= 200) | ((area >= 9) & (in_model >= 0.5 * area) & (like_sky >= 0.5 * area))
    keep[0] = False
    a = keep[labels]
    # a small hole in the sky with nothing green, dark or hanging in it is sky (a wisp, a blue gap)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((~a).astype(np.uint8), connectivity=8)
    solid = np.zeros(n, bool)
    solid[np.unique(labels[green | (v < 0.5) | (L < L_sky - 2 * DARKER_DL) | detail])] = True
    hole = (stats[:, cv2.CC_STAT_AREA] < 40) & ~solid
    hole[0] = False
    a = a | hole[labels]
    a = a.astype(np.float32)
    return cv2.GaussianBlur(a, (0, 0), 1.0) * a


def _field(img: np.ndarray, known: np.ndarray) -> np.ndarray:
    """A smooth field of `img` (colour or single channel) from its `known` pixels only: their mean
    at a quarter resolution, inpainted where none are known, scaled back up."""
    h, w = known.shape
    k = known.astype(np.float32)
    x = img.astype(np.float32) * (k[..., None] if img.ndim == 3 else k)
    num = cv2.resize(x, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    den = cv2.resize(k, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    ok = den > 0.5
    small = np.zeros_like(num)
    small[ok] = num[ok] / (den[ok][:, None] if img.ndim == 3 else den[ok])
    small = cv2.inpaint(np.clip(small, 0, 255).astype(np.uint8), (~ok).astype(np.uint8) * 255, 5,
                        cv2.INPAINT_TELEA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


STRIP_PX = 12              # the hidden-sky fill works in strips this tall (each its own offset)


def fill_hidden_sky(bgr: np.ndarray, alpha: np.ndarray, plate: np.ndarray, reach: int) -> np.ndarray:
    """The plate, where drifting can reveal it, filled with real cloud texture instead of a smooth
    inpainted colour (else a cloudless band trails every roof and tree edge as the clouds move):
    hidden sky within `reach` px sideways of visible sky is taken, strip by strip (STRIP_PX tall,
    so a long diagonal tree edge is many short pieces), from the nearest visible sky at the same
    height beside it - the smallest horizontal offset whose copy is all sky - Poisson-blended in."""
    if reach < 2:
        return plate
    sky = alpha > 0.9
    near = cv2.dilate(sky.astype(np.uint8), np.ones((1, 2 * reach + 1), np.uint8)) > 0
    region = ((alpha < 0.5) & near).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
    h, w = alpha.shape
    out = plate.copy()
    pad = 3
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if area < 20:
            continue
        for sy in range(y, y + bh, STRIP_PX):
            rows = slice(sy, min(y + bh, sy + STRIP_PX))
            piece = lab[rows, x:x + bw] == i
            if not piece.any():
                continue
            cols = np.nonzero(piece.any(axis=0))[0]
            px0, px1 = x + int(cols.min()), x + int(cols.max()) + 1
            pw = px1 - px0
            y0, y1 = max(0, rows.start - pad), min(h, rows.stop + pad)
            x0, x1 = max(0, px0 - pad), min(w, px1 + pad)
            comp = np.zeros((y1 - y0, x1 - x0), bool)
            comp[rows.start - y0:rows.stop - y0, px0 - x0:px1 - x0] = lab[rows, px0:px1] == i
            best_dx = None
            for k in range(0, 6):
                for dx in (pw + 4 + 8 * k, -(pw + 4 + 8 * k)):
                    if x0 + dx < 0 or x1 + dx > w:
                        continue
                    if float(sky[y0:y1, x0 + dx:x1 + dx][comp].mean()) >= 0.97:
                        best_dx = dx
                        break
                if best_dx is not None:
                    break
            if best_dx is None:
                continue
            mask = cv2.dilate(comp.astype(np.uint8), np.ones((3, 3), np.uint8)) * 255
            mask[[0, -1], :] = 0
            mask[:, [0, -1]] = 0
            if not mask.any():
                continue
            try:
                out = cv2.seamlessClone(np.ascontiguousarray(bgr[y0:y1, x0 + best_dx:x1 + best_dx]), out, mask,
                                        ((x0 + x1) // 2, (y0 + y1) // 2), cv2.NORMAL_CLONE)
            except cv2.error:
                continue
    return out


def sky_plate(bgr: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """The sky continued behind the foreground. Photo pixels are kept only PLATE_INSET_PX inside
    the sky: an edge pixel mixes roof or leaf into the sky and, carried along by the drift, shows
    as a ghost. The rest is inpainted at a quarter resolution, except within FINE_PX of the sky -
    behind a twig, a wire, a gutter's edge - which is inpainted at full resolution from the sky
    round it: in 4 px blocks a small hole drifted into view as a square patch."""
    h, w = alpha.shape
    k = 2 * PLATE_INSET_PX + 1
    sure = cv2.erode((alpha > 0.5).astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    small = cv2.resize(bgr, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    sure_small = cv2.resize(sure.astype(np.uint8), (w // 4, h // 4), interpolation=cv2.INTER_NEAREST)
    sure_small = cv2.erode(sure_small, np.ones((3, 3), np.uint8))
    coarse = cv2.inpaint(small, (1 - sure_small) * 255, 5, cv2.INPAINT_TELEA)
    coarse = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    f = 2 * FINE_PX + 1
    near = (cv2.dilate(sure.astype(np.uint8), np.ones((f, f), np.uint8)) > 0) & ~sure
    fine = cv2.inpaint(np.where(sure[..., None], bgr, coarse), near.astype(np.uint8) * 255, 3,
                       cv2.INPAINT_TELEA)
    return np.where(sure[..., None], bgr, np.where(near[..., None], fine, coarse))


def edge_share(bgr: np.ndarray, alpha: np.ndarray, plate: np.ndarray) -> np.ndarray:
    """How much of each pixel just outside the sky (within FRINGE_PX) is sky, 0..0.85: its
    lightness between the dark thing beside it and the sky behind it (the plate). Only for things
    darker than the sky; a light edge (white trim) stays 0."""
    L = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    L0 = cv2.cvtColor(plate, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    sky = alpha > 0.5
    k = 2 * FRINGE_PX + 1
    fringe = (cv2.dilate(sky.astype(np.uint8), np.ones((k, k), np.uint8)) > 0) & ~sky
    L_obj = cv2.erode(np.where(sky, 255.0, L).astype(np.float32), np.ones((5, 5), np.uint8))
    span = L0 - L_obj
    share = np.clip((L - L_obj) / np.maximum(span, 1.0), 0.0, 0.85)
    return np.where(fringe & (span > DARKER_DL), share, 0.0).astype(np.float32)


class SkyMotion:
    def __init__(self, bgr: np.ndarray, coarse: np.ndarray, seed: int = 7,
                 effects=("drift", "billow"), strength: float = 1.0, wind: float = 1.0,
                 speed: float = DRIFT_FRAC, duration: float = 4.0, billow_px: float | None = None,
                 billow_hz: float = BILLOW_HZ):
        """strength scales billow and shadows (1 = the defaults above); speed: the top clouds'
        drift, share of the photo width per second; wind: +1 left-to-right, -1 right-to-left;
        duration: the shot's length - the drift is centred on its middle (the sky is exactly the
        photo half-way through), so it only reveals half as much hidden sky at either end."""
        self.src = bgr
        self.h, self.w = bgr.shape[:2]
        self.alpha = refine_alpha(bgr, coarse)
        self.effects, self.k, self.wind = set(effects), float(strength), float(wind)
        self.speed, self.duration = float(speed), float(duration)
        self.billow_px = float(billow_px) if billow_px is not None else BILLOW_FRAC * self.w
        self.billow_hz = float(billow_hz)
        reach = int(np.ceil(self.speed * self.w * self.duration / 2)) + 6 if "drift" in self.effects else 0
        self.plate = fill_hidden_sky(bgr, self.alpha, sky_plate(bgr, self.alpha), reach)
        self.edge = edge_share(bgr, self.alpha, self.plate)
        self._share = np.maximum(self.alpha, self.edge)[..., None]
        self._plate_f = self.plate.astype(np.float32)
        self.na, self.nb, self.nc = GradientNoise3D(seed), GradientNoise3D(seed ^ 0x9E3779B9), GradientNoise3D(seed ^ 0x51ED27)
        ys = np.nonzero(self.alpha > 0.5)[0]
        self.top = float(ys.min()) if len(ys) else 0.0
        self.horizon = float(np.percentile(ys, 97)) if len(ys) else self.h * 0.4
        X, Y = np.meshgrid(np.arange(self.w, dtype=np.float32), np.arange(self.h, dtype=np.float32))
        self.X, self.Y = X, Y
        # drift speed by height: nearer (higher) clouds faster
        depth = np.clip((self.horizon - Y[:, :1]) / max(1.0, self.horizon - self.top), 0.0, 1.0)
        self.drift_w = (DRIFT_HORIZON + (1 - DRIFT_HORIZON) * depth).astype(np.float32)
        # the ground below the skyline takes the cloud shadows (fading in just under it), most on
        # sunlit surfaces: what is already in shade barely changes when a cloud passes
        ground = np.clip((Y - self.horizon) / (0.05 * self.h), 0.0, 1.0)
        v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 2].astype(np.float32) / 255.0
        sunlit = np.clip((v - 0.30) / 0.40, 0.0, 1.0)
        self.shadow_w = ((1.0 - self.alpha) * ground * sunlit).astype(np.float32)
        # shadows sweep faster near the camera (the ground plane in perspective)
        near = np.clip((Y[:, :1] - self.horizon) / max(1.0, self.h - self.horizon), 0.0, 1.0)
        self.shadow_speed = (1.0 + (SHADOW_NEAR - 1.0) * near).astype(np.float32)
        gx, gy = np.meshgrid(np.arange(0, self.w + GRID, GRID, dtype=np.float64),
                             np.arange(0, self.h + GRID, GRID, dtype=np.float64))
        self.gx, self.gy = gx, gy

    def _up(self, f: np.ndarray) -> np.ndarray:
        return cv2.resize(f.astype(np.float32), (self.w, self.h), interpolation=cv2.INTER_CUBIC)

    def frame(self, t: float) -> np.ndarray:
        """The photo at t s with the sky in motion (source pixels, before the camera)."""
        dx = np.zeros((self.h, self.w), np.float32)
        dy = np.zeros((self.h, self.w), np.float32)
        if "drift" in self.effects:
            dx += self.wind * self.speed * self.w * (t - self.duration / 2) * self.drift_w
        if "billow" in self.effects:
            f = BILLOW_SCALE * self.w
            amp = self.k * self.billow_px
            dx += amp * self._up(self.na(self.gx / f, self.gy / f, t * self.billow_hz))
            dy += 0.6 * amp * self._up(self.nb(self.gx / f, self.gy / f, t * self.billow_hz + 17.0))
        out = self.src.astype(np.float32)
        if "shadows" in self.effects:
            s = SHADOW_SCALE * self.w
            gy_rows = np.clip(self.gy.astype(np.int64), 0, self.h - 1)
            sweep = self.wind * SHADOW_SPEED * self.speed * self.w * t * self.shadow_speed[gy_rows, 0] / s
            n = self._up(self.nc(self.gx / s - sweep, self.gy / (0.5 * s), 0.04 * t))
            shade = np.clip((n - SHADOW_COVER) / SHADOW_EDGE + 0.5, 0.0, 1.0)
            shade = shade * shade * (3 - 2 * shade)
            d = self.k * SHADOW_DEPTH * shade * self.shadow_w
            out *= np.dstack([1.0 - d * c for c in SHADOW_TINT])
        if "drift" in self.effects or "billow" in self.effects:
            # the change in the sky only: inside the sky (plate = photo) the moved sky itself; at
            # an edge the thing in front keeps its colour
            sky = cv2.remap(self.plate, self.X - dx, self.Y - dy, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT).astype(np.float32)
            out += self._share * (sky - self._plate_f)
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def load_sky(shot, spec, images_dir: str, src: np.ndarray, duration_s: float) -> SkyMotion:
    """The sky layer of a shot from its sky_drift region: the fetched mask (any size, scaled to
    the photo), the region's billow strength and rate, and the shot's wind direction."""
    import os

    from tour_worker.motion.masks import mask_relpath
    from tour_worker.motion.wind import shot_wind
    path = os.path.join(images_dir, mask_relpath(shot.shot_id, spec.id))
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError("missing")
    h, w = src.shape[:2]
    coarse = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return SkyMotion(src, coarse, seed=int(spec.seed), wind=float(shot_wind(shot, w).direction),
                     duration=duration_s, billow_px=float(spec.amplitude_px), billow_hz=float(spec.freq_hz))
