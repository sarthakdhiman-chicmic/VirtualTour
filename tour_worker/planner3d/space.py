"""The scene as a floor plan (5 cm cells), in the planner's own frame: x, y across the floor, z up,
z = 0 on the floor. Scenes come in any axes (z up, COLMAP's y down, ...): scene/orient.py finds up
and the floor, and every point the planner makes goes back to the scene's axes with to_scene().

  free       floor a camera can stand over: floor splats below, nothing between 0.15 and 2.0 m
  clearance  metres from each free cell to the nearest obstacle or wall (dt)
  furniture  low pieces (beds, sofas, tables, desks, islands): cells with stuff at 0.3-1.0 m and
             nothing at 1.3-2.0 m, as rotated rectangles with their top height
  tall       obstacles that block a view (walls, wardrobes, tall units) - not low furniture
  rooms      the free floor split among the caller's points by walking distance, a step costing
             1 / clearance^2 (squeezing through a doorway is dear): rooms meet at the pinch between them (a doorway); rooms joined by an
             opening wider than OPEN_PLAN_M are one open-plan space, which a camera may move through
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from tour_worker.scene.orient import estimate_frame
from tour_worker.scene.splat_io import SplatModel

CELL = 0.05
OPAQUE, MAX_SPLAT_M = 0.35, 0.25       # splats that make the floor plan: opaque, not huge background blobs
BODY = (0.15, 2.0)                     # heights a camera body could hit
FURN_BODY, FURN_TALL = (0.30, 1.0), (1.3, 2.0)
MIN_PIECE_M2 = 0.25
# what a room's main piece is, by room type (scoring/rooms.py slugs): composed on first, largest first
PIECE_CLASSES = ("bed", "sofa", "armchair", "table", "coffee table", "desk", "kitchen island", "counter",
                 "countertop", "bathtub", "pool table", "chest of drawers", "buffet", "ottoman", "bench")
PREFER = {
    "bed": ("primary_bedroom", "master_bedroom", "secondary_bedroom", "guest_bedroom", "childrens_bedroom",
            "bedroom", "bed_room", "nursery"),
    "sofa": ("living_room", "family_room", "great_room", "sunroom", "open_concept", "media_room", "bonus_room",
             "basement"),
    "table": ("dining_room", "formal_dining_room", "breakfast_nook"),
    "kitchen island": ("kitchen",), "desk": ("home_office", "study", "library"),
    "bathtub": ("primary_bathroom", "master_bathroom", "ensuite", "bathroom", "secondary_bathroom"),
    "pool table": ("game_room",),
}
SECOND = {"kitchen": ("countertop", "counter", "table"), "living_room": ("armchair", "coffee table"),
          "family_room": ("armchair", "coffee table"), "dining_room": ("buffet",)}
VIEW_BLOCKERS = ("door", "screen door", "windowpane", "mirror", "wardrobe", "glass", "shower")
OPEN_PLAN_M = 0.9                      # shared boundary (on cells >= ROOM_RMIN clear) longer than this: one
                                       # open-plan space - an opening ~1.3 m+ wide; a 0.9 m door measures ~0.5 m
ROOM_RMIN = 0.15                       # the room map uses cells at least this clear
_NB = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
       (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142))


def basis(up: np.ndarray) -> np.ndarray:
    """R (3x3, rows x', y', up): scene -> planner axes. x' is the scene axis least like up, made
    square to it, so a z-up scene keeps its x and y."""
    up = np.asarray(up, np.float64)
    up = up / np.linalg.norm(up)
    a = np.eye(3)[int(np.argmin(np.abs(up)))] if abs(up[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = a - (a @ up) * up
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(up, x), up])


@dataclass
class Piece:
    """A low piece of furniture: a rotated rectangle on the floor plan and its top height (m)."""
    centre: np.ndarray        # (2,)
    u: np.ndarray             # unit, along the long side
    v: np.ndarray             # unit, along the short side
    half_long: float
    half_short: float
    top: float
    area: float
    cls: str | None = None    # what it is, from the scene's labels (bed, sofa, kitchen island ...)
    # what a shot frames: the labelled object of its class inside it (a floor-plan piece often
    # merges a bed with its nightstands and bench; the labels know the bed alone) - else itself
    frame: "Piece | None" = None

    def local(self, p) -> np.ndarray:
        r = np.asarray(p, np.float64)[..., :2] - self.centre
        return np.stack([r @ self.u, r @ self.v], -1)

    def contains(self, p, pad: float = 0.0) -> np.ndarray:
        p = np.asarray(p)
        if p.ndim == 2 and len(p) > 4096:             # many points (splats): only those near it can be in
            r = math.hypot(self.half_long + pad, self.half_short + pad)
            near = (np.abs(p[:, 0] - self.centre[0]) <= r) & (np.abs(p[:, 1] - self.centre[1]) <= r)
            out = np.zeros(len(p), bool)
            idx = np.nonzero(near)[0]
            if len(idx):
                out[idx] = self.contains(p[idx], pad) if len(idx) <= 4096 else self._contains(p[idx], pad)
            return out
        return self._contains(p, pad)

    def _contains(self, p, pad: float) -> np.ndarray:
        q = self.local(p)
        return (np.abs(q[..., 0]) <= self.half_long + pad) & (np.abs(q[..., 1]) <= self.half_short + pad)

    def dist(self, p) -> float:
        """Metres from p to the footprint (0 inside)."""
        q = self.local(p)
        return float(np.hypot(max(abs(q[0]) - self.half_long, 0.0), max(abs(q[1]) - self.half_short, 0.0)))

    def top_corners(self, core: float = 1.0) -> np.ndarray:
        """(4, 3) corners of its top surface, the long side cut to `core` (the ends of a long group
        may leave the frame)."""
        return np.array([[*(self.centre + sx * self.half_long * core * self.u + sy * self.half_short * self.v), self.top]
                         for sx in (-1, 1) for sy in (-1, 1)])

    def to_dict(self) -> dict:
        return {"class": self.cls, "centre": [round(float(x), 2) for x in self.centre],
                "long_m": round(2 * self.half_long, 2), "short_m": round(2 * self.half_short, 2),
                "top_m": round(self.top, 2), "area_m2": round(self.area, 2)}


FRAME_MIN_AREA = 0.6           # m2: a labelled object this big is a piece a shot may frame


class Space:
    def __init__(self, model: SplatModel, up_hint=None):
        frame = estimate_frame(model, up_hint)
        self.frame = frame
        self.R = basis(frame.up)
        self.floor_along_up = float(frame.floor)
        self.room_height = float(frame.room_height)
        self.warnings = list(frame.warnings)
        # every splat in planner axes, z = height above the floor (kept: occlusion picks from it)
        self.local = (model.means.astype(np.float64) @ self.R.T).astype(np.float32)
        self.local[:, 2] -= self.floor_along_up
        self._grid(model)
        self._furniture()
        self.room_keys: list[str] = []
        self.room_types: dict[str, str | None] = {}
        self.objects: list[dict] = []
        self.labels = None
        self.view_block = np.zeros(self.shape, bool)

    # ---- axes -----------------------------------------------------------------------------------
    def to_scene(self, p) -> np.ndarray:
        """Planner points (..., 3), z above the floor -> the scene's own axes."""
        q = np.asarray(p, np.float64).copy()
        q[..., 2] = q[..., 2] + self.floor_along_up
        return q @ self.R

    def to_local(self, p) -> np.ndarray:
        q = np.asarray(p, np.float64) @ self.R.T
        q[..., 2] -= self.floor_along_up
        return q

    @property
    def up_scene(self) -> np.ndarray:
        return self.R[2].copy()

    # ---- the floor plan -------------------------------------------------------------------------
    def _grid(self, model: SplatModel) -> None:
        keep = (model.opacities > OPAQUE) & (model.scales.max(1) < MAX_SPLAT_M)
        P = self.local[keep].astype(np.float64)
        h = P[:, 2]
        self.lo = np.percentile(P[:, :2], 0.2, axis=0) - 0.5
        hi = np.percentile(P[:, :2], 99.8, axis=0) + 0.5
        W, H = (np.ceil((hi - self.lo) / CELL)).astype(int)
        ix = ((P[:, 0] - self.lo[0]) / CELL).astype(int).clip(0, W - 1)
        iy = ((P[:, 1] - self.lo[1]) / CELL).astype(int).clip(0, H - 1)

        def hist(sel):
            g = np.zeros((H, W), np.int32)
            np.add.at(g, (iy[sel], ix[sel]), 1)
            return g
        floor = hist((h > -0.08) & (h < 0.08))
        obst = hist((h > BODY[0]) & (h < BODY[1]))
        obst_m = ndi.binary_opening(obst >= 4, iterations=1) | (obst >= 25)     # stray floaters are not walls
        floor_m = ndi.binary_closing(floor >= 2, iterations=3)
        inside = ndi.binary_fill_holes(ndi.binary_closing(floor_m | obst_m, iterations=6))
        free = floor_m & ~ndi.binary_dilation(obst_m, iterations=1) & inside
        lab, n = ndi.label(free)
        if n == 0:
            raise ValueError("the scene has no free floor to stand on")
        sizes = ndi.sum(free, lab, range(1, n + 1))
        self.free = lab == (1 + int(np.argmax(sizes)))                           # the connected interior
        self.dt = ndi.distance_transform_edt(self.free) * CELL
        self.obst, self.inside = obst_m, inside
        self.shape = (H, W)
        # furniture inputs (body / tall counts, top heights), kept for _furniture
        self._body = hist((h > FURN_BODY[0]) & (h < FURN_BODY[1]))
        self._tall = hist((h > FURN_TALL[0]) & (h < FURN_TALL[1]))
        top = np.full((H, W), -1.0)
        s = (h > 0.1) & (h < 1.25)
        np.maximum.at(top, (iy[s], ix[s]), h[s])
        self._top = top

    def _furniture(self) -> None:
        import cv2
        low = (self._body >= 6) & (self._tall < 3)
        low = ndi.binary_opening(ndi.binary_closing(low, iterations=2), iterations=1)
        lab, n = ndi.label(low)
        pieces = []
        for i in range(1, n + 1):
            msk = lab == i
            area = msk.sum() * CELL * CELL
            if area < MIN_PIECE_M2:
                continue
            ys, xs = np.nonzero(msk)
            pts = np.c_[self.lo[0] + (xs + .5) * CELL, self.lo[1] + (ys + .5) * CELL].astype(np.float32)
            (cx, cy), (w, hh), ang = cv2.minAreaRect(pts)
            if w < hh:
                w, hh, ang = hh, w, ang + 90
            a = math.radians(ang)
            u = np.array([math.cos(a), math.sin(a)])
            tops = self._top[msk]
            pieces.append(Piece(np.array([cx, cy], np.float64), u, np.array([-u[1], u[0]]), w / 2, hh / 2,
                                float(np.percentile(tops[tops > 0], 90)) if (tops > 0).any() else 0.8, float(area)))
        pieces.sort(key=lambda p: (-p.area, p.centre[0], p.centre[1]))
        self.pieces = pieces
        tall = self.obst.copy()                          # low furniture does not block a view the way a wall does
        yy, xx = np.mgrid[0:self.shape[0], 0:self.shape[1]]
        cells = np.stack([self.lo[0] + (xx + .5) * CELL, self.lo[1] + (yy + .5) * CELL], -1)
        for p in pieces:
            tall[p.contains(cells, 0.05)] = False
        self.tall = tall

    # ---- cells ----------------------------------------------------------------------------------
    def cell(self, p) -> tuple[int, int]:
        return int((p[1] - self.lo[1]) / CELL), int((p[0] - self.lo[0]) / CELL)

    def centre_of(self, i, j) -> np.ndarray:
        return np.array([self.lo[0] + (j + .5) * CELL, self.lo[1] + (i + .5) * CELL])

    def clearance(self, p) -> float:
        i, j = self.cell(p)
        if not (0 <= i < self.shape[0] and 0 <= j < self.shape[1]):
            return 0.0
        return float(self.dt[i, j])

    def snap(self, p, rmin: float, within: float = 0.8) -> np.ndarray | None:
        """p, or the nearest point within `within` m with clearance >= rmin (None: none)."""
        if self.clearance(p) >= rmin:
            return np.asarray(p, np.float64)[:2]
        ys, xs = np.nonzero(self.dt >= rmin)
        if not len(ys):
            return None
        pts = np.c_[self.lo[0] + (xs + .5) * CELL, self.lo[1] + (ys + .5) * CELL]
        d = np.linalg.norm(pts - np.asarray(p)[:2], axis=1)
        k = int(np.argmin(d))
        return pts[k] if d[k] <= within else None

    # ---- rooms ----------------------------------------------------------------------------------
    def set_rooms(self, points: dict[str, np.ndarray]) -> list[str]:
        """Grow the rooms from their points (planner axes): walking distance that is dear through
        narrow passages, from each point, so two rooms meet where the free space pinches - at the doorway between
        them - not halfway between their points. Returns warnings (a point off the floor)."""
        warnings, keys, cells = [], [], []
        self.room_points: dict[str, np.ndarray] = {}
        for key, p in points.items():
            s = self.snap(p, ROOM_RMIN, within=1.5)
            if s is None:
                warnings.append(f"room_point_off_floor:{key}")
                continue
            if self.cell(s) in cells:
                warnings.append(f"room_point_shared:{key}")
                continue
            keys.append(key)
            cells.append(self.cell(s))
            self.room_points[key] = s
        self.room_keys = keys
        lab = np.full(self.shape, -1, np.int32)
        if keys:
            # walking distance where a step costs 1 / clearance^2: crossing a doorway (~0.4 m clear)
            # costs ~10x crossing an open room, so every cell goes to the room it is reached from
            # without squeezing through a door - wherever in its room the caller put the point
            ok = self.free
            g, idx, ij = self._graph(ok, weighted=False, step_cost=1.0 / np.maximum(self.dt, 0.08) ** 2)
            _, _, src = dijkstra(g, indices=[idx[c] for c in cells], min_only=True, return_predecessors=True)
            order = {int(idx[c]): k for k, c in enumerate(cells)}
            reached = src >= 0
            lab[ij[reached, 0], ij[reached, 1]] = [order[int(x)] for x in src[reached]]
        self.room_of = lab
        # every cell of the home with a room, the floor under furniture included (nearest room):
        # what a view's sight lines end on (own_share)
        _, (ii, jj) = ndi.distance_transform_edt(lab < 0, return_indices=True)
        self.room_fill = np.where(self.inside | (lab >= 0), lab[ii, jj], -1)
        if not keys:
            self.spaces = {}
            return warnings
        self.spaces = self._open_plan()
        return warnings

    def _graph(self, ok: np.ndarray, weighted: bool = True, step_cost: np.ndarray | None = None):
        """The 8-connected graph over the cells in `ok`: (csr matrix, cell -> node index (-1 off),
        node -> (i, j)). Weighted: a step costs more near walls (paths keep to the middle);
        step_cost: a per-cell cost instead."""
        H, W = self.shape
        idx = np.full((H, W), -1, np.int64)
        ii, jj = np.nonzero(ok)
        idx[ii, jj] = np.arange(len(ii))
        if step_cost is not None:
            cost = step_cost
        else:
            cost = 1.0 + 2.5 * np.clip(0.9 - self.dt, 0, None) if weighted else np.ones((H, W))
        rows, cols, vals = [], [], []
        for di, dj, w in _NB:
            a, b = ii + di, jj + dj
            m = (a >= 0) & (a < H) & (b >= 0) & (b < W)
            m[m] &= ok[a[m], b[m]]
            rows.append(idx[ii[m], jj[m]])
            cols.append(idx[a[m], b[m]])
            vals.append(w * cost[a[m], b[m]])
        n = len(ii)
        g = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
        return g, idx, np.c_[ii, jj]

    def _open_plan(self) -> dict[str, set[str]]:
        R = np.where(self.dt >= ROOM_RMIN, self.room_of, -1)
        pairs: dict[tuple[int, int], int] = {}
        for a, b in ((R[:, :-1], R[:, 1:]), (R[:-1, :], R[1:, :])):
            m = (a >= 0) & (b >= 0) & (a != b)
            for x, y in zip(a[m].tolist(), b[m].tolist()):
                k = (min(x, y), max(x, y))
                pairs[k] = pairs.get(k, 0) + 1
        parent = list(range(len(self.room_keys)))

        def find(i):
            while parent[i] != i:
                i = parent[i]
            return i
        for (x, y), n in sorted(pairs.items()):
            if n * CELL > OPEN_PLAN_M:
                parent[find(x)] = find(y)
        keys = self.room_keys
        return {k: {keys[j] for j in range(len(keys)) if find(j) == find(i)} for i, k in enumerate(keys)}

    def room_at(self, p) -> str | None:
        i, j = self.cell(p)
        if not (0 <= i < self.shape[0] and 0 <= j < self.shape[1]):
            return None
        k = int(self.room_of[i, j])
        return self.room_keys[k] if k >= 0 else None

    def room_of_object(self, p, radius: float = 1.0) -> str | None:
        """The room most of the free floor within radius m of p belongs to (closer cells count
        more): a basin by a bathroom door is the bathroom's, not the hall's nearest strip."""
        i, j = self.cell(p)
        k = int(radius / CELL)
        H, W = self.shape
        a0, a1, b0, b1 = max(0, i - k), min(H, i + k + 1), max(0, j - k), min(W, j + k + 1)
        win = self.room_of[a0:a1, b0:b1]
        ys, xs = np.nonzero(win >= 0)
        if not len(ys):
            return self.room_near(p, 1.5)
        d = np.hypot(ys + a0 - i, xs + b0 - j) * CELL
        w = np.clip(1.0 - d / radius, 0.05, 1.0)
        votes = np.bincount(win[ys, xs], weights=w, minlength=len(self.room_keys))
        return self.room_keys[int(np.argmax(votes))]

    def room_near(self, p, radius: float = 1.5) -> str | None:
        """The room of the free floor nearest p (within radius m): a sofa's middle is not floor."""
        r = self.room_at(p)
        if r is not None:
            return r
        i, j = self.cell(p)
        k = int(radius / CELL)
        H, W = self.shape
        a0, a1, b0, b1 = max(0, i - k), min(H, i + k + 1), max(0, j - k), min(W, j + k + 1)
        win = self.room_of[a0:a1, b0:b1]
        ys, xs = np.nonzero(win >= 0)
        if not len(ys):
            return None
        q = int(np.argmin((ys + a0 - i) ** 2 + (xs + b0 - j) ** 2))
        return self.room_keys[int(win[ys[q], xs[q]])]

    def space_mask(self, room: str, rmin: float) -> np.ndarray:
        ids = [self.room_keys.index(r) for r in sorted(self.spaces.get(room, {room}))]
        return (self.dt >= rmin) & np.isin(self.room_of, ids)

    def room_area(self, room: str) -> float:
        return float((self.room_of == self.room_keys.index(room)).sum() * CELL * CELL)

    def _tall_at(self, p) -> bool:
        i, j = self.cell(p)
        return not (0 <= i < self.shape[0] and 0 <= j < self.shape[1]) or bool(self.tall[i, j])

    def piece_room(self, piece: Piece) -> str | None:
        """The room of the free floor right around a piece."""
        votes: dict[str, int] = {}
        for sx in np.linspace(-1, 1, 7):
            for sy in (-1, 1):
                edge = piece.centre + sx * piece.half_long * piece.u + sy * piece.half_short * piece.v
                for pad in (0.3, 0.5):
                    q = piece.centre + sx * piece.half_long * piece.u + sy * (piece.half_short + pad) * piece.v
                    # only floor the piece faces: none behind the wall it stands against
                    if any(self._tall_at(edge + f * (q - edge)) for f in (0.25, 0.5, 0.75, 1.0)):
                        continue
                    r = self.room_at(q) if self.clearance(q) > 0 else None
                    if r:
                        votes[r] = votes.get(r, 0) + 1
        return max(sorted(votes), key=votes.get) if votes else None

    def main_pieces(self, room: str, min_area: float = 0.8) -> list[Piece]:
        """The room's low furniture big enough to compose on: what its type is about first (a
        bedroom's bed, a living room's sofa, a kitchen's island - from the scene's labels), then
        other labelled pieces, then unlabelled ones; largest first within each."""
        from tour_worker.scoring.rooms import canon
        rtype = canon(self.room_types.get(room))
        want = [c for c, types in PREFER.items() if rtype in types] + list(SECOND.get(rtype, ()))

        def rank(p):
            if p.cls in want:
                return want.index(p.cls)
            return len(want) + (0 if p.cls in PIECE_CLASSES else 1)
        mine = [p for p in self.pieces if p.area >= min_area and self.piece_room(p) == room]
        if self.labels is not None:          # labelled: a small piece the labels cannot name is a cabinet, not a subject
            mine = [p for p in mine if p.cls is not None or p.area >= 1.5]
        if not want:        # a hall or a balcony: never another room type's furniture (a living room's coffee table)
            others = {c for c in PREFER} | {c for cs in SECOND.values() for c in cs}
            mine = [p for p in mine if p.cls not in others]
        # in an open plan a room may compose on what it is about standing in a neighbouring part of
        # the space (a dining room's table drawn into the living room's floor) - unless that piece
        # is what its own room is about too
        mates = self.spaces.get(room, {room}) - {room}
        for p in self.pieces:
            if p.area < min_area or any(p is q for q in mine) or p.cls is None or p.cls not in want[:1]:
                continue
            home = self.piece_room(p)
            if home in mates and canon(self.room_types.get(home)) not in PREFER.get(p.cls, ()):
                mine.append(p)
        return sorted(mine, key=lambda p: (rank(p), -p.area))

    def attach_labels(self, labels) -> None:
        """The scene's labels (scene/semantics.py): each piece gets the class most of its splats
        carry (a piece class), and the labelled objects become known (features, details)."""
        self.labels = labels
        L, lab = self.local, labels.labels
        base = np.nonzero((L[:, 2] > 0.1) & (lab > 0))[0]     # labelled splats off the floor, once
        Lb = L[base]
        for p in self.pieces:
            sub = np.nonzero(Lb[:, 2] < p.top + 0.2)[0]
            m = base[sub[p.contains(Lb[sub], 0.05)]]
            if len(m) < 30:
                continue
            vals, cnt = np.unique(lab[m], return_counts=True)
            best = max(((int(c), labels.name(int(v))) for v, c in zip(vals, cnt) if labels.name(int(v)) in PIECE_CLASSES),
                       default=None)
            if best and best[0] >= 0.15 * len(m):
                p.cls = best[1]
        # glass, mirrors and doors stop a sight line: their faint splats are not walls on the floor
        # plan, but a view does not reach "into the home" through a frosted door or a mirror
        blk = labels.mask(*VIEW_BLOCKERS) & (L[:, 2] > 0.3) & (L[:, 2] < 2.0)
        if blk.any():
            q = L[blk]
            ix = ((q[:, 0] - self.lo[0]) / CELL).astype(int).clip(0, self.shape[1] - 1)
            iy = ((q[:, 1] - self.lo[1]) / CELL).astype(int).clip(0, self.shape[0] - 1)
            cnt = np.zeros(self.shape, np.int32)
            np.add.at(cnt, (iy, ix), 1)
            self.view_block = ndi.binary_closing(cnt >= 3, iterations=1)
        self.objects = [dict(o) for o in labels.objects]
        for p in self.pieces:                       # the labelled object a piece is about
            inside = [o for o in self.objects if o["class"] in PIECE_CLASSES and o.get("footprint")
                      and (p.cls is None or o["class"] == p.cls)
                      and bool(p.contains(np.asarray(o["footprint"]["centre"]), 0.1))
                      and o["footprint"]["long"] * o["footprint"]["short"] >= FRAME_MIN_AREA]
            if not inside:
                continue
            o = max(inside, key=lambda o: o["footprint"]["long"] * o["footprint"]["short"])
            f = o["footprint"]
            a = math.radians(f["axis_deg"])
            u = np.array([math.cos(a), math.sin(a)])
            p.cls = p.cls or o["class"]
            p.frame = Piece(np.asarray(f["centre"], np.float64), u, np.array([-u[1], u[0]]), f["long"] / 2,
                            f["short"] / 2, float(o.get("top", o["max"][2])), f["long"] * f["short"], o["class"])
        for o in self.objects:                      # the room of the free floor nearest it (rooms set later)
            o["room"] = self.room_of_object(np.asarray(o["centre"][:2])) if self.room_keys else o.get("room")

    def room_objects(self, room: str, classes=None) -> list[dict]:
        return [o for o in getattr(self, "objects", []) if o.get("room") == room
                and (classes is None or o["class"] in classes)]

    # ---- sampling and paths ---------------------------------------------------------------------
    def spread_points(self, mask: np.ndarray, n: int = 36, min_gap: float = 0.25) -> np.ndarray:
        """Up to n points spread over mask (farthest-point sampling from its most open cell)."""
        ys, xs = np.nonzero(mask)
        if not len(ys):
            return np.zeros((0, 2))
        pts = np.c_[self.lo[0] + (xs + .5) * CELL, self.lo[1] + (ys + .5) * CELL]
        pick = [int(np.argmax(self.dt[ys, xs]))]
        d = np.linalg.norm(pts - pts[pick[0]], axis=1)
        while len(pick) < min(n, len(pts)) and d.max() > min_gap:
            pick.append(int(np.argmax(d)))
            d = np.minimum(d, np.linalg.norm(pts - pts[pick[-1]], axis=1))
        return pts[pick]

    def paths_between(self, pts: np.ndarray, mask: np.ndarray):
        """Shortest walking paths (through mask, keeping off the walls) between every pair of pts:
        path(a, b) -> (k, 2) polyline or None."""
        g, idx, ij = self._graph(mask)
        nodes = [idx[self.cell(p)] for p in pts]
        valid = [k for k, n in enumerate(nodes) if n >= 0]
        if not valid:
            return lambda a, b: None
        _, pred = dijkstra(g, indices=[nodes[k] for k in valid], return_predecessors=True)
        row = {k: r for r, k in enumerate(valid)}

        def path(a: int, b: int):
            if a not in row or nodes[b] < 0:
                return None
            p, out = pred[row[a]], [nodes[b]]
            while out[-1] != nodes[a]:
                nxt = p[out[-1]]
                if nxt < 0:
                    return None
                out.append(nxt)
            out.reverse()
            poly = np.array([self.centre_of(*ij[n]) for n in out])
            poly[0], poly[-1] = pts[a], pts[b]
            return poly
        return path

    def route(self, a: str, b: str) -> tuple[np.ndarray, list[str]] | None:
        """How one walks from room a to room b: the shortest walking path between their points
        (keeping off the walls) and the rooms it passes through, in order - what way b lies from a,
        and whether a hallway joins them. None when no path exists."""
        if a not in self.room_points or b not in self.room_points:
            return None
        ends = np.array([self.room_points[a], self.room_points[b]])
        for rmin in (0.22, 0.15):
            poly = self.paths_between(ends, (self.dt >= rmin) & (self.room_of >= 0))(0, 1)
            if poly is not None and len(poly) >= 2:
                rooms: list[str] = []
                for q in poly:
                    r = self.room_at(q)
                    if r is not None and (not rooms or rooms[-1] != r):
                        rooms.append(r)
                return poly, rooms
        return None

    # ---- views ----------------------------------------------------------------------------------
    def own_share(self, room: str, p, heading_rad: float, hfov_deg: float, rays: int = 9) -> float:
        """Share of the middle of a view (half the frame's width) whose sight line ends in `room`:
        a kitchen shot shows the kitchen, not the living room through the open plan."""
        half = math.radians(hfov_deg) / 4
        k = self.room_keys.index(room)
        hs = heading_rad + np.linspace(-half, half, rays)
        d = np.maximum(self.rays(p, hs) - 0.15, 0)
        i, j = self._cells(p[0] + np.cos(hs) * d, p[1] + np.sin(hs) * d)
        ok = (i >= 0) & (i < self.shape[0]) & (j >= 0) & (j < self.shape[1])
        return int((self.room_fill[i[ok], j[ok]] == k).sum()) / rays

    def ray(self, p, heading_rad: float, max_m: float = 8.0) -> float:
        """How far the eye sees from p along a heading before a wall or tall obstacle (or leaving
        the home): low furniture does not stop a view."""
        return float(self.rays(p, np.array([heading_rad]), max_m)[0])

    # sight lines step CELL at a time; the planner casts ~10^5 of them a tour, so they run as one
    # array operation per view (the same steps, cells and stops as a step-by-step walk)
    def _cells(self, x, y):
        """cell() on arrays: int() truncates toward zero, as astype does."""
        return ((np.asarray(y) - self.lo[1]) / CELL).astype(np.int64), ((np.asarray(x) - self.lo[0]) / CELL).astype(np.int64)

    def _blocked(self) -> np.ndarray:
        key = (id(self.tall), id(self.inside), id(self.view_block))
        if getattr(self, "_blk_key", None) != key:
            self._blk = self.tall | ~self.inside | self.view_block
            self._blk_key = key
        return self._blk

    def _steps(self, max_m: float) -> np.ndarray:
        """The distances a walk visits: CELL added step by step (the same float sums) while the
        previous one is short of max_m."""
        cache = self.__dict__.setdefault("_steps_cache", {})
        if max_m not in cache:
            d = np.cumsum(np.full(int(max_m / CELL) + 3, CELL))
            keep = np.r_[True, d[:-1] < max_m]
            cache[max_m] = d[:np.argmin(keep) if not keep.all() else len(d)]
        return cache[max_m]

    def rays(self, p, headings, max_m: float = 8.0) -> np.ndarray:
        """ray() for many headings from one spot at once."""
        hs = np.asarray(headings, np.float64)
        d = self._steps(max_m)
        x = p[0] + np.cos(hs)[:, None] * d[None, :]
        y = p[1] + np.sin(hs)[:, None] * d[None, :]
        i, j = self._cells(x, y)
        H, W = self.shape
        out = (i < 0) | (i >= H) | (j < 0) | (j >= W)
        blk = out.copy()
        inb = ~out
        blk[inb] = self._blocked()[i[inb], j[inb]]
        first = np.where(blk.any(1), blk.argmax(1), len(d) - 1)
        return d[first]

    def view_depth(self, p, heading_rad: float, hfov_deg: float, rays: int = 11) -> float:
        """Mean reach (m) of the view across the frame's width: how much room it shows."""
        half = math.radians(hfov_deg) / 2
        return float(np.mean(self.rays(p, heading_rad + np.linspace(-half, half, rays))))
