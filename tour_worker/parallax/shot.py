"""One photo shot drawn with parallax (render/frames.ShotWarp): the photo's rigid layers at the size the
shot draws it, and the camera's slide over the shot.

The layers (layers.extended_layers: each layer's colours, alpha and own pixels) are built once per photo
and size and saved as .npy beside the planes, so every render process maps the same files instead of
building them again. The camera slides the way the planned move goes - the crop window's travel
(sideways / up-down) and its zoom (forward-back) - at the shot's own pace, by the translation whose
near-far disparity is cap_px at a 1920-wide copy of the photo (the strength the prototype was approved
at, 2026-11-25: 16 px). Every layer is drawn with its plane's homography, then the shot's framing.

Dynamic content (the moving sky, motion regions) is drawn per layer through that layer's own
homography, on the layer's own pixels, so it moves with the surface it is on."""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

from tour_worker.parallax import layers as L
from tour_worker.parallax.prepare import DIRNAME, WORK_W, planes_path

PAD = 2                     # output px around a layer's projected box
MAX_W = 3840                # layers are built at most this wide (a 4K tour's photo can be ~7000 px: its
                            # layers would take several GB); the framing is scaled to match


def _planes_at(images: str, name: str, w: int, h: int):
    """The photo's planes (prepare.py) at w x h: masks resized, each plane's depth map recomputed."""
    d = np.load(planes_path(images, name))
    idx = cv2.resize(d["idx"], (w, h), interpolation=cv2.INTER_NEAREST)
    K = L.intrinsics(w, h)
    ys, xs = np.mgrid[0:h, 0:w]
    rays = np.stack([(xs - K[0, 2]) / K[0, 0], (ys - K[1, 2]) / K[1, 1], np.ones((h, w))], -1)
    planes = []
    for i, lab in enumerate(json.loads(str(d["labels"]))):
        n, c = d["n"][i], float(d["c"][i])
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = rays @ n
            zmap = np.where(denom > 1e-6, c / denom, np.inf).astype(np.float32)
        planes.append(L.Plane(n=n, c=c, mask=idx == i, label=lab, depth=float(d["depth"][i]), zmap=zmap))
    z = cv2.resize(d["z"], (w, h), interpolation=cv2.INTER_LINEAR)
    return planes, K, z


def _layer_files(images: str, name: str, w: int, h: int) -> dict[str, str]:
    base = os.path.join(images, DIRNAME, f"{name}.{w}x{h}")
    return {k: f"{base}.{k}.npy" for k in ("colour", "alpha", "meta")}


OWN_A, FILL_A = 255, 254      # alpha codes: the layer's own photo pixels / its opaque hidden fill


def _build(images: str, name: str, src: np.ndarray, files: dict[str, str]) -> None:
    """Each layer cropped to the box its alpha covers (a layer is ~10-20% of the frame: full-frame
    layers of a 4K tour's photos filled the pod's disk, 2026-11-25), packed into one colour and one
    alpha buffer; meta: per layer the plane (n, c), box and offset. Alpha 255 marks the layer's own
    pixels (the photo itself), 254 its opaque fill."""
    h, w = src.shape[:2]
    planes, _, z = _planes_at(images, name, w, h)
    L.PHOTO_DEPTH[0] = z
    layers = L.extended_layers(src, planes)
    idx = np.full((h, w), -1, np.int32)
    for i, (_, a) in enumerate(layers):
        idx[a >= 0.999] = i                                    # near layers win: the pixel's drawn owner
    cols, alphas, meta, off = [], [], [], 0
    for i, ((col, a), p) in enumerate(zip(layers, planes)):
        a8 = np.round(a * FILL_A).astype(np.uint8)
        a8[(idx == i) & (a >= 0.999)] = OWN_A
        ys, xs = np.nonzero(a8.max(axis=1))[0], np.nonzero(a8.max(axis=0))[0]
        if not len(ys):
            meta.append([*p.n, p.c, 0, 0, 0, 0, off])
            continue
        x0, y0, x1, y1 = xs[0], ys[0], xs[-1] + 1, ys[-1] + 1
        cols.append(np.ascontiguousarray(col[y0:y1, x0:x1]).ravel())
        alphas.append(np.ascontiguousarray(a8[y0:y1, x0:x1]).ravel())
        meta.append([*p.n, p.c, x0, y0, x1, y1, off])
        off += (y1 - y0) * (x1 - x0)
    for key, arr in (("colour", np.concatenate(cols) if cols else np.zeros(0, np.uint8)),
                     ("alpha", np.concatenate(alphas) if alphas else np.zeros(0, np.uint8)),
                     ("meta", np.array(meta, np.float64).reshape(-1, 9))):
        tmp = files[key] + ".tmp.npy"
        np.save(tmp, arr)
        os.replace(tmp, files[key])


class ShotParallax:
    def __init__(self, images: str, name: str, src: np.ndarray, frames: list, cap_px: float,
                 progress):
        """frames: the shot's camera [((cx, cy, zoom), roll)]; progress(fr) -> 0..1, the move's progress
        at fractional frame fr (the slide follows it)."""
        sh, sw_ = src.shape[:2]
        self.s = min(1.0, MAX_W / sw_)                         # layer px per photo px
        self.S = np.diag([self.s, self.s, 1.0])
        self.Sinv = np.diag([1 / self.s, 1 / self.s, 1.0])
        w, h = (sw_, sh) if self.s == 1.0 else (round(sw_ * self.s), round(sh * self.s))
        files = _layer_files(images, name, w, h)
        if not all(os.path.exists(p) for p in files.values()):
            small = src if self.s == 1.0 else cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)
            _build(images, name, small, files)
        colour = np.load(files["colour"], mmap_mode="r")
        alpha = np.load(files["alpha"], mmap_mode="r")
        meta = np.load(files["meta"])
        self.n, self.c = meta[:, :3], meta[:, 3]
        # per layer: its box in the photo (x0, y0, x1, y1) and its colour / alpha crops (memory-mapped)
        self.boxes, self.colour, self.alpha = [], [], []
        for row in meta:
            x0, y0, x1, y1, off = (int(v) for v in row[4:9])
            if x1 <= x0:
                self.boxes.append(None); self.colour.append(None); self.alpha.append(None)
                continue
            n_px = (y1 - y0) * (x1 - x0)
            self.boxes.append((x0, y0, x1, y1))
            self.colour.append(colour[off * 3:(off + n_px) * 3].reshape(y1 - y0, x1 - x0, 3))
            self.alpha.append(alpha[off:off + n_px].reshape(y1 - y0, x1 - x0))
        self.K = L.intrinsics(w, h)
        self.Kinv = np.linalg.inv(self.K)
        self.w, self.h = w, h
        self.progress = progress
        # the slide: the window's travel (x right, y down, in window widths) and its zoom
        (cx0, cy0, z0), _ = frames[0]
        (cx1, cy1, z1), _ = frames[-1]
        win = 1.0 / np.sqrt(z0 * z1)
        direction = np.array([(cx1 - cx0) / win, (cy1 - cy0) * h / (w * win), np.log(z1 / z0) * 0.5])
        self.T = np.zeros(3)
        if np.linalg.norm(direction) > 1e-6:
            planes = []
            for i in range(len(self.c)):
                own = np.zeros((h, w), bool)
                if self.boxes[i] is not None:
                    x0, y0, x1, y1 = self.boxes[i]
                    own[y0:y1, x0:x1] = self.alpha[i] == OWN_A
                planes.append(L.Plane(n=self.n[i], c=float(self.c[i]), mask=own, label="", depth=0.0))
            self.T = L.travel_toward(planes, self.K, direction, cap_px * w / WORK_W)

    def homographies(self, fr: float) -> list[np.ndarray]:
        """Each layer's photo px -> moved-view px at fractional frame fr."""
        t = self.T * (self.progress(fr) - 0.5)
        return [self.K @ (np.eye(3) - np.outer(t, self.n[i]) / self.c[i]) @ self.Kinv for i in range(len(self.c))]

    def layer_box(self, i: int, m: np.ndarray, ow: int, oh: int):
        """Layer i's box on screen for output -> layer-photo homography m, or None when off screen."""
        b = self.boxes[i]
        if b is None:
            return None
        x0, y0, x1, y1 = b
        p = np.linalg.inv(m) @ np.array([[x0, x1, x1, x0], [y0, y0, y1, y1], [1, 1, 1, 1]], float)
        if (p[2] <= 0).any():
            return 0, 0, ow, oh
        u, v = p[0] / p[2], p[1] / p[2]
        c0, c1 = max(0, int(np.floor(u.min())) - PAD), min(ow, int(np.ceil(u.max())) + PAD)
        r0, r1 = max(0, int(np.floor(v.min())) - PAD), min(oh, int(np.ceil(v.max())) + PAD)
        return None if c0 >= c1 or r0 >= r1 else (r0, r1, c0, c1)

    def draw(self, crop: np.ndarray, fr: float, ow: int, oh: int, dynamic=None) -> np.ndarray:
        """The frame for framing `crop` (output px -> photo px) at fractional frame fr: the layers far to
        near, each only over its box. dynamic(i, m, box) -> the layer's own pixels drawn from the live
        photo (sky, motion regions) through m over box, or None to keep the layer's stored colours."""
        acc = np.zeros((oh, ow, 3), np.float32)
        crop = self.S @ crop                                    # output px -> layer px
        for i, H in enumerate(self.homographies(fr)):
            m = np.linalg.inv(H) @ crop
            box = self.layer_box(i, m, ow, oh)
            if box is None:
                continue
            r0, r1, c0, c1 = box
            bx0, by0 = self.boxes[i][:2]
            # output box px -> photo px -> the layer's crop px
            mm = np.array([[1, 0, -bx0], [0, 1, -by0], [0, 0, 1]], float) @ m @ np.array([[1, 0, c0], [0, 1, r0], [0, 0, 1]], float)
            size = (c1 - c0, r1 - r0)
            warp = lambda img, border: cv2.warpPerspective(          # noqa: E731
                np.asarray(img), mm, size, flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR, borderMode=border)
            col = warp(self.colour[i], cv2.BORDER_REPLICATE).astype(np.float32)
            a8 = warp(self.alpha[i], cv2.BORDER_CONSTANT)
            a = np.minimum(a8.astype(np.float32) / FILL_A, 1.0)[..., None]
            if dynamic is not None:
                live = dynamic(i, self.Sinv @ m, box)          # (the live photo is at photo px)
                if live is not None:
                    k = np.clip(a8.astype(np.float32) - FILL_A, 0.0, 1.0)[..., None]   # own pixels
                    col = col * (1 - k) + live.astype(np.float32) * k
            sub = acc[r0:r1, c0:c1]
            sub *= 1 - a
            sub += col * a
        return np.clip(acc + 0.5, 0, 255).astype(np.uint8)

    def touches(self, i: int, boxes_src: list[tuple[float, float, float, float]]) -> bool:
        """Whether layer i's box meets any of the photo boxes (x0, y0, x1, y1), photo px."""
        b = self.boxes[i]
        boxes_src = [tuple(v * self.s for v in bx) for bx in boxes_src]
        return b is not None and any(b[0] < x1 and x0 < b[2] and b[1] < y1 and y0 < b[3]
                                     for x0, y0, x1, y1 in boxes_src)
