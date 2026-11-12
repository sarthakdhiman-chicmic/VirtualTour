"""Overlays drawn over the composed tour frames (SOW 3.2.1a; reviewed with the user 2026-11-05):

  room labels   each captioned shot (a room's first shot, planner/captions.py) gets the room's
                name in a script font, after a small gold index and a thin rule ("02 | Kitchen"),
                over a soft gradient that keeps white text readable. Animation, from the room's
                first frame: +0.40 s the rule draws downward (0.35 s) and the index fades in;
                +0.55 s the name writes on left to right (0.9 s); it holds until 3.6 s after it
                began or 0.35 s before the room ends, then fades out over 0.5 s.
                plan.output.room_labels sets the font, the scale (x the reviewed maximum size,
                0.6-1.0, default 0.8) and the placement (bottom/top x left/center/right; right
                placements mirror the label: "Kitchen | 02").
  staged        "VIRTUALLY STAGED" on every shot of a staged photo, the whole shot, bottom right
                (bottom left when the room labels sit bottom right). It is a disclosure, so the
                MLS copy keeps it.
  title card    (plan.output.branding: address / locality / title) over the opening shot,
                0.5 s to at most 4.6 s, on a light scrim; it names the first room, so that room
                gets no room label. Kept in the MLS copy.
  end card      (branding.agent / logo_uri) the last END_CARD_S of the tour on a dark scrim: the
                agent's photo, name, phone and email, and the brokerage logo. Branded copy only.
  corner logo   (branding.corner_logo) the brokerage logo, top right, between the title and the
                end card. Branded copy only.

Each label is drawn once (Pillow, the bundled OFL fonts) into a small layer around it and composited
per frame in float32, the same arithmetic every run: renders stay byte-identical. Sizes are set for a
1080 px short side and scale with the output."""
from __future__ import annotations

import threading

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")
# the script fonts, and each one's size at scale 1.0 for a 1080 px short side: matched by eye so
# the names read the same size in every font (Sacramento's 76 px is the reviewed maximum)
SCRIPT_FONTS = {"sacramento": ("Sacramento-Regular.ttf", 76), "mr_dafoe": ("MrDafoe-Regular.ttf", 69),
                "pacifico": ("Pacifico-Regular.ttf", 48), "kaushan_script": ("KaushanScript-Regular.ttf", 52)}
INDEX_FONT, INDEX_PX = "Lato-Regular.ttf", 26
STAGED_FONT, STAGED_PX = "Lato-Semibold.ttf", 20
STAGED_TEXT = "VIRTUALLY STAGED"
MARGIN_PX = 72                      # from the frame's edges (1080 px short side), not scaled by the label
RULE_HALF_PX, GAP_PX = 26, 18       # the rule's half height and the gaps around it, scaled by the label
GOLD, WHITE = (212, 175, 110), (255, 255, 255)

# animation (s), reviewed 2026-11-05
LABEL_IN, RULE_S, WRITE_AT, WRITE_S = 0.40, 0.35, 0.55, 0.9
HOLD_MAX, OUT_S, END_GAP = 3.6, 0.5, 0.35
INDEX_FADE_S, SHADE_FADE_S, WRITE_SOFT_PX = 0.45, 0.3, 60
STAGED_FADE_S = 0.3
TITLE_IN, TITLE_MAX_END, CARD_FADE_S, TITLE_SCRIM = 0.5, 4.6, 0.6, 0.30
END_CARD_S, END_SCRIM = 3.5, 0.62
LOGO_H_PX, LOGO_OPACITY, CORNER_LOGO_H_PX, CORNER_OPACITY = 80, 1.0, 52, 0.85
AGENT_PHOTO_PX = 200
SHADE_DEPTH, SHADE_SHARE = 0.40, 0.30    # the gradient: this dark at the edge, over this share of the height


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _font(name: str, px: float) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(os.path.join(FONT_DIR, name), max(8, round(px)))


class _Layer:
    """Premultiplied BGR + alpha (float32) of a small box of the frame, at (x0, y0)."""

    def __init__(self, box: tuple[int, int, int, int], draw_fn, blur: float, shadow: int = 175):
        x0, y0, w, h = box
        sh = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw_fn(ImageDraw.Draw(sh), (0, 0, 0, shadow), -x0, -y0 + 2)
        sh = sh.filter(ImageFilter.GaussianBlur(blur))
        top = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw_fn(ImageDraw.Draw(top), None, -x0, -y0)
        a = np.asarray(Image.alpha_composite(sh, top), np.float32) / 255.0
        self.x0, self.y0 = x0, y0
        self.alpha = a[..., 3:]
        self.rgb = a[..., [2, 1, 0]] * self.alpha * 255.0   # premultiplied, BGR, the frame's 0-255


def _image_layer(x0: float, y0: float, rgba: Image.Image, W: int, H: int) -> "_Layer":
    """A layer of an RGBA image placed at (x0, y0), clipped to the frame."""
    lay = _Layer.__new__(_Layer)
    x0, y0 = int(round(x0)), int(round(y0))
    a = np.asarray(rgba.convert("RGBA"), np.float32) / 255.0
    cx0, cy0 = max(0, -x0), max(0, -y0)
    a = a[cy0:max(cy0, H - y0), cx0:max(cx0, W - x0)]
    lay.x0, lay.y0 = max(0, x0), max(0, y0)
    lay.alpha = a[..., 3:]
    lay.rgb = a[..., [2, 1, 0]] * lay.alpha * 255.0
    return lay


def _fit_height(img: Image.Image, h: int) -> Image.Image:
    img = img.convert("RGBA")
    return img.resize((max(1, round(img.width * h / img.height)), h), Image.LANCZOS)


def _circle(img: Image.Image, d: int) -> Image.Image:
    """The image cropped to a centred square and masked to a circle d px across (anti-aliased)."""
    img = img.convert("RGBA")
    side = min(img.size)
    img = img.crop(((img.width - side) // 2, (img.height - side) // 2,
                    (img.width - side) // 2 + side, (img.height - side) // 2 + side)).resize((d, d), Image.LANCZOS)
    mask = Image.new("L", (4 * d, 4 * d), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, 4 * d - 1, 4 * d - 1], fill=255)
    img.putalpha(mask.resize((d, d), Image.LANCZOS))
    return img


def _box(x0, y0, x1, y1, W, H, pad):
    x0, y0 = max(0, int(x0 - pad)), max(0, int(y0 - pad))
    x1, y1 = min(W, int(x1 + pad + 1)), min(H, int(y1 + pad + 1))
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


class _Label:
    """One room label: its layers and the frames it is on screen."""

    def __init__(self, idx: int, name: str, labels, W: int, H: int, t0: float, t_out: float):
        k = min(W, H) / 1080.0
        s = k * labels.scale
        m = MARGIN_PX * k
        file, px = SCRIPT_FONTS[labels.font]
        script, index_f = _font(file, px * s), _font(INDEX_FONT, max(12.0, INDEX_PX * s))
        probe = ImageDraw.Draw(Image.new("L", (1, 1)))
        l, t, r, b = probe.textbbox((0, 0), name, font=script)
        il, it, ir, ib = probe.textbbox((0, 0), f"{idx:02d}", font=index_f)
        tw, th, iw = r - l, b - t, ir - il
        gap, rh = GAP_PX * s, RULE_HALF_PX * s
        self.vert, horiz = labels.placement.split("_")
        base = H - m - 2 * k if self.vert == "bottom" else m + th      # the name's ink bottom
        cy = base - th / 2
        total = iw + gap + 2 + gap + tw
        x = {"left": m, "right": W - m - total, "center": (W - total) / 2}[horiz]
        if horiz == "right":                                            # mirrored, hugging the edge
            x_name, x_rule, x_idx = x, x + tw + gap, x + tw + 2 * gap + 2
        else:
            x_idx, x_rule, x_name = x, x + iw + gap, x + iw + 2 * gap + 2
        pad = 14 * k
        iy = cy - (ib - it) / 2 - it
        self.index = _Layer(_box(x_idx, cy - (ib - it), x_idx + iw, cy + (ib - it), W, H, pad),
                            lambda d, c, ox, oy: d.text((x_idx - il + ox, iy + oy), f"{idx:02d}", font=index_f,
                                                        fill=c or GOLD + (255,)), blur=5 * k)
        self.rule = _Layer(_box(x_rule - 2, cy - rh, x_rule + 2, cy + rh, W, H, pad),
                           lambda d, c, ox, oy: d.line([(x_rule + ox, cy - rh + oy), (x_rule + ox, cy + rh + oy)],
                                                       fill=c or WHITE + (200,), width=max(1, round(2 * k))),
                           blur=3 * k)
        self.text = _Layer(_box(x_name, base - th, x_name + tw, base, W, H, pad + 4 * k),
                           lambda d, c, ox, oy: d.text((x_name - l + ox, base - b + oy), name, font=script,
                                                       fill=c or WHITE + (255,)), blur=5 * k)
        self.rule_y = (cy - rh, cy + rh)
        self.name_x = (x_name - 10 * k, x_name + tw + 14 * k)
        self.soft = WRITE_SOFT_PX * k
        self.t0, self.t_out = t0, t_out


class Overlays:
    """The overlays of a plan: build once per process (render/direct.py), apply per frame."""

    def __init__(self, plan, timeline: list[dict], images: str | None = None):
        W, H, fps = plan.output.width, plan.output.height, plan.output.fps
        self.W, self.H, self.fps = W, H, fps
        self.images = images
        labels = getattr(plan.output, "room_labels", None)
        b = self.branding = getattr(plan.output, "branding", None)
        end_t = (timeline[-1]["render_from"] + timeline[-1]["render_frames"]) / fps if timeline else 0.0
        first_end = (timeline[0]["render_from"] + timeline[0]["render_frames"]) / fps if timeline else 0.0
        self.title = ((TITLE_IN, min(first_end - 0.4, TITLE_MAX_END))
                      if b is not None and (b.address or b.title or b.kicker) and first_end > TITLE_IN + 2 * CARD_FADE_S
                      else None)
        self.end_card = ((max(0.0, end_t - END_CARD_S), end_t)
                         if b is not None and (b.agent is not None or b.logo_uri) else None)
        self.corner = None
        if b is not None and b.corner_logo and b.logo_uri:
            a = (self.title[1] if self.title else 0.0) + 0.3
            e = (self.end_card[0] if self.end_card else end_t) - 0.3
            self.corner = (a, e) if e > a + 1.0 else None
        self._cards: dict[str, _Layer | list] = {}
        self.labels = []                       # (spec, built _Label or None) in order
        self._built: dict[int, _Label] = {}
        self._lock = threading.Lock()
        shots = plan.shots
        idx = 0
        for i, (shot, t) in enumerate(zip(shots, timeline)):
            if labels is None or not getattr(shot, "caption", None):
                continue
            if i == 0 and self.title is not None:                     # the title card names this room
                continue
            j = i                                                     # the room: until the next room
            while j + 1 < len(shots) and shots[j + 1].room_key == shot.room_key and not shots[j + 1].caption:
                j += 1
            start = t["render_from"] / fps
            end = (timeline[j]["render_from"] + timeline[j]["render_frames"]) / fps
            t0 = start + LABEL_IN
            t_out = min(end - END_GAP - OUT_S, t0 + HOLD_MAX)
            if t_out < t0 + (WRITE_AT - LABEL_IN) + WRITE_S:          # too short to write the name out
                continue
            idx += 1
            self.labels.append((idx, shot.caption, labels, t0, t_out))
        self.staged = [(t["render_from"] / fps, (t["render_from"] + t["render_frames"]) / fps)
                       for shot, t in zip(shots, timeline) if getattr(shot, "staged", False)]
        side = "left" if labels is not None and labels.placement == "bottom_right" else "right"
        self._staged_layer = self._staged(side) if self.staged else None
        ramp = np.clip((np.arange(H) - H * (1 - SHADE_SHARE)) / (H * SHADE_SHARE), 0, 1) ** 1.6 * SHADE_DEPTH
        self._shade = {"bottom": ramp.astype(np.float32), "top": ramp[::-1].astype(np.float32)}

    @property
    def active(self) -> bool:
        return bool(self.labels or self.staged or self.title or self.end_card or self.corner)

    # ---- branding: built once per process, on first use -------------------------------------------
    def _image(self, uri: str | None) -> Image.Image | None:
        if not uri or not self.images:
            return None
        from tour_worker.io.images import branding_path
        try:
            return Image.open(branding_path(self.images, uri)).convert("RGBA")
        except Exception:  # noqa: BLE001 - a missing image leaves its place empty
            return None

    def _text_block(self, lines, cy: float) -> list:
        """Centred lines [(text, font, colour)] stacked around height cy, as layers."""
        W, H = self.W, self.H
        k = min(W, H) / 1080.0
        probe = ImageDraw.Draw(Image.new("L", (1, 1)))
        boxes = [probe.textbbox((0, 0), t, font=f) for t, f, _ in lines]
        gap = 14 * k
        total = sum(b[3] - b[1] for b in boxes) + gap * (len(lines) - 1)
        y = cy - total / 2
        out = []
        for (t, f, col), (l, tp, r, bt) in zip(lines, boxes):
            x = (W - (r - l)) / 2
            yy = y

            def draw(d, c, ox, oy, t=t, f=f, col=col, x=x, yy=yy, l=l, tp=tp):
                d.text((x - l + ox, yy - tp + oy), t, font=f, fill=c or col + (255,))
            out.append(_Layer(_box(x, yy, x + (r - l), yy + (bt - tp), W, H, 16 * k), draw, blur=5 * k))
            y += (bt - tp) + gap
        return out

    def _title_layers(self) -> list:
        if "title" not in self._cards:
            k = min(self.W, self.H) / 1080.0
            b = self.branding
            lines = []
            if b.kicker:
                lines.append((b.kicker.upper(), _font("Lato-Semibold.ttf", 30 * k), GOLD))
            if b.address:
                lines.append((b.address, _font(SCRIPT_FONTS["sacramento"][0], 96 * k), WHITE))
            if b.locality:
                lines.append((b.locality.upper(), _font("Lato-Semibold.ttf", 26 * k), (235, 228, 215)))
            if b.title:
                lines.append((b.title, _font("Lato-Light.ttf", 34 * k), WHITE))
            self._cards["title"] = self._text_block(lines, self.H * 0.5)
        return self._cards["title"]

    def _end_layers(self) -> list:
        if "end" not in self._cards:
            W, H = self.W, self.H
            k = min(W, H) / 1080.0
            b = self.branding
            layers, y = [], H * 0.5
            photo = self._image(b.agent.photo_uri if b.agent else None)
            logo = self._image(b.logo_uri)
            name = b.agent.name if b.agent else None
            contact = " \u00b7 ".join(x for x in ((b.agent.phone, b.agent.email) if b.agent else ()) if x)
            lines = []
            if name:
                lines.append((name, _font(SCRIPT_FONTS["sacramento"][0], 84 * k), WHITE))
            if contact:
                lines.append((contact, _font(INDEX_FONT, 28 * k), (235, 228, 215)))
            parts_h = ((AGENT_PHOTO_PX * k + 30 * k) if photo else 0) + (len(lines) * 70 * k) + \
                      ((LOGO_H_PX * k + 30 * k) if logo else 0)
            top = y - parts_h / 2
            if photo is not None:
                d = round(AGENT_PHOTO_PX * k)
                layers.append(_image_layer((W - d) / 2, top, _circle(photo, d), W, H))
                top += d + 30 * k
            if lines:
                block = self._text_block(lines, top + len(lines) * 35 * k)
                layers += block
                top += len(lines) * 70 * k
            if logo is not None:
                lg = _fit_height(logo, round(LOGO_H_PX * k))
                if lg.width > W * 0.5:
                    lg = lg.resize((round(W * 0.5), max(1, round(lg.height * W * 0.5 / lg.width))), Image.LANCZOS)
                layers.append(_image_layer((W - lg.width) / 2, top + 30 * k, lg, W, H))
            self._cards["end"] = layers
        return self._cards["end"]

    def _corner_layer(self):
        if "corner" not in self._cards:
            k = min(self.W, self.H) / 1080.0
            logo = self._image(self.branding.logo_uri)
            lay = None
            if logo is not None:
                lg = _fit_height(logo, round(CORNER_LOGO_H_PX * k))
                m = MARGIN_PX * k * 0.7
                lay = _image_layer(self.W - m - lg.width, m, lg, self.W, self.H)
            self._cards["corner"] = lay
        return self._cards["corner"]

    def _staged(self, side: str) -> _Layer:
        W, H = self.W, self.H
        k = min(W, H) / 1080.0
        f = _font(STAGED_FONT, STAGED_PX * k)
        probe = ImageDraw.Draw(Image.new("L", (1, 1)))
        widths = [probe.textlength(ch, font=f) for ch in STAGED_TEXT]
        track = 4 * k
        tw = sum(widths) + track * (len(STAGED_TEXT) - 1)
        m, ph, px = MARGIN_PX * k, 40 * k, 16 * k
        x0 = W - m - tw - px if side == "right" else m - px
        y1 = H - m + 10 * k

        def draw(d, colour, ox, oy):
            if colour is not None:                                  # the tag has no drop shadow
                return
            d.rounded_rectangle([x0 + ox, y1 - ph + oy, x0 + tw + 2 * px + ox, y1 + oy], radius=8 * k,
                                fill=(0, 0, 0, 120))
            x = x0 + px
            for ch, cw in zip(STAGED_TEXT, widths):
                d.text((x + ox, y1 - ph + 7 * k + oy), ch, font=f, fill=(255, 255, 255, 235))
                x += cw + track
        return _Layer(_box(x0, y1 - ph, x0 + tw + 2 * px, y1, W, H, 2), draw, blur=1)

    def _label(self, i: int) -> _Label:
        with self._lock:                       # 3D chunks draw from parallel threads
            got = self._built.get(i)
            if got is None:              # kept: a tour has a handful of labels, each a small text layer
                idx, name, labels, t0, t_out = self.labels[i]
                got = self._built[i] = _Label(idx, name, labels, self.W, self.H, t0, t_out)
            return got

    @staticmethod
    def _blend(f: np.ndarray, layer: _Layer, k):
        """f (float32 BGR frame) <- layer over it at opacity k (a scalar or an array the layer's shape)."""
        h, w = layer.alpha.shape[:2]
        sl = f[layer.y0:layer.y0 + h, layer.x0:layer.x0 + w]
        a = layer.alpha * k
        sl *= 1.0 - a
        sl += layer.rgb * k

    def apply(self, frame: np.ndarray, g: int, captions: bool = True) -> np.ndarray:
        """Frame g (BGR uint8) with its overlays. captions=False: the MLS (unbranded) copy - the
        title card and the staged label only. Returns the frame itself when nothing is on screen."""
        t = g / self.fps
        live = [i for i, (_, _, _, t0, t_out) in enumerate(self.labels) if t0 <= t <= t_out + OUT_S] if captions else []
        staged = [s for s in self.staged if s[0] <= t < s[1]]
        title = self.title is not None and self.title[0] <= t <= self.title[1]
        end = captions and self.end_card is not None and self.end_card[0] <= t
        corner = captions and self.corner is not None and self.corner[0] <= t <= self.corner[1]
        if not (live or staged or title or end or corner):
            return frame
        f = frame.astype(np.float32)
        if title:
            a, e = self.title
            k = _smooth((t - a) / CARD_FADE_S) * (1.0 - _smooth((t - (e - CARD_FADE_S)) / CARD_FADE_S))
            f *= 1.0 - TITLE_SCRIM * k
            for lay in self._title_layers():
                self._blend(f, lay, k)
        if corner:
            a, e = self.corner
            lay = self._corner_layer()
            if lay is not None:
                self._blend(f, lay, CORNER_OPACITY * _smooth((t - a) / 0.5) * (1.0 - _smooth((t - (e - 0.5)) / 0.5)))
        for i in live:
            L = self._label(i)
            fade = 1.0 - _smooth((t - L.t_out) / OUT_S)
            shade = self._shade[L.vert] * (fade * _smooth((t - L.t0) / SHADE_FADE_S))
            f *= (1.0 - shade)[:, None, None]
            p = _smooth((t - L.t0) / RULE_S)                          # the rule draws downward
            y0, y1 = L.rule_y
            ys = np.arange(L.rule.y0, L.rule.y0 + L.rule.alpha.shape[0], dtype=np.float32)[:, None, None]
            self._blend(f, L.rule, fade * np.clip((y0 - 4 + p * (y1 - y0 + 8) - ys) / 4.0, 0, 1))
            self._blend(f, L.index, fade * _smooth((t - L.t0) / INDEX_FADE_S))
            q = _smooth((t - L.t0 - (WRITE_AT - LABEL_IN)) / WRITE_S)   # the name writes on
            x0, x1 = L.name_x
            xs = np.arange(L.text.x0, L.text.x0 + L.text.alpha.shape[1], dtype=np.float32)[None, :, None]
            self._blend(f, L.text, fade * np.clip((x0 + q * (x1 - x0 + L.soft) - xs) / L.soft, 0, 1))
        for a, e in staged:
            k = _smooth((t - a) / STAGED_FADE_S) * (1.0 - _smooth((t - (e - STAGED_FADE_S)) / STAGED_FADE_S))
            self._blend(f, self._staged_layer, k)
        if end:                                   # last: the card sits over everything
            k = _smooth((t - self.end_card[0]) / CARD_FADE_S)
            f *= 1.0 - END_SCRIM * k
            for lay in self._end_layers():
                self._blend(f, lay, k)
        return np.clip(f + 0.5, 0, 255).astype(np.uint8)
