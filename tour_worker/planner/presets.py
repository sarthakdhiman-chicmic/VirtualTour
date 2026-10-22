from typing import Any

from tour_worker.scoring.rooms import canon

# shift_x / shift_y are fractions of the visible window, so on-screen travel is the same at
# any zoom. Motion rules, from watching renders of 3:2 listing photos in a 16:9 frame:
#   - Every move changes scale. A lateral move at constant zoom reads as a flat scan of a
#     photo; pairing it with a push or pull is what gives it depth.
#   - Every move is linear (constant speed). Shots cross-fade, and an eased move sits
#     still exactly while it fades in and out, which reads as a slideshow.
#   - The values below are full-speed magnitudes; the planner scales each move by room and
#     shot length (PlannerConfig.motion_speed*). Keep moves continuous rather than large.
#   - No bare zooms: pushes and pulls also drift 6-11% sideways (toward the subject) and a
#     little vertically, so they read as a dolly move rather than a digital zoom.
#   - Every move banks a little (roll_deg, reached by the end of the shot, in the direction
#     of travel), like a gimbal. The renderer fits the rolled frame inside the solved window,
#     so roll never changes the bounds maths here; it costs ~3-5% extra effective zoom.
# Lateral presets carry their own max_zoom: past ~1.35x the upscale softens the image.
PRESETS: dict[str, dict[str, Any]] = {
    "push_in_gentle": {
        "zoom_total": 0.12,
        "shift_x": -0.09,
        "shift_y": -0.03,
        "easing": "linear",
        "roll_deg": 0.5,
    },
    "push_in_strong": {
        "zoom_total": 0.20,
        "shift_x": -0.07,
        "shift_y": 0.05,
        "easing": "linear",
        "roll_deg": 0.8,
    },
    # lateral truck while dollying in
    "push_slide": {
        "zoom_total": 0.10,
        "shift_x": -0.18,
        "shift_y": -0.02,
        "easing": "linear",
        "roll_deg": 1.0,
        "max_zoom": 1.35,
    },
    # lateral truck while easing out - reveals the room as it moves
    "slide": {
        "zoom_total": -0.08,
        "shift_x": -0.20,
        "shift_y": 0.00,
        "easing": "linear",
        "roll_deg": 1.2,
        "max_zoom": 1.35,
    },
    # hero move for wide rooms: a big truck combined with a push, like a gimbal sweep
    "hero_sweep": {
        "zoom_total": 0.16,
        "shift_x": -0.18,
        "shift_y": -0.04,
        "easing": "linear",
        "roll_deg": 1.5,
        "max_zoom": 1.35,
    },
    # pedestal up with a slight push
    "tilt": {
        "zoom_total": 0.06,
        "shift_x": 0.00,
        "shift_y": 0.14,
        "easing": "linear",
        "roll_deg": 0.4,
    },
    "pull_back": {
        "zoom_total": -0.13,
        "shift_x": -0.10,
        "shift_y": -0.04,
        "easing": "linear",
        "roll_deg": 0.6,
    },
}

LATERAL_PRESETS = frozenset({"slide", "push_slide", "hero_sweep"})

def _solve_camera(
    preset_name: str,
    src_w: float,
    src_h: float,
    out_w: float,
    out_h: float,
    subject_x: float,
    subject_y: float,
    order: int,
    scale_factor: float,
    max_zoom: float = 1.6,
    zoom_step: float = 0.005,
    subject_x_high: float = 0.58,
    subject_x_low: float = 0.42,
    speed: float = 1.0,
) -> dict[str, Any] | None:
    config = PRESETS[preset_name]
    # speed scales the whole move (zoom, travel and roll) for slower rooms or shorter shots
    zoom_total = config["zoom_total"] * speed
    shift_x = config["shift_x"] * scale_factor * speed
    shift_y = config["shift_y"] * scale_factor * speed
    easing = config["easing"]
    roll_deg = config.get("roll_deg", 0.0) * speed
    max_zoom = min(max_zoom, config.get("max_zoom", max_zoom))

    a_src = src_w / src_h
    a_out = out_w / out_h
    if a_src >= a_out:
        ww = a_out / a_src
        wh = 1.0
    else:
        ww = 1.0
        wh = a_src / a_out

    if subject_x > subject_x_high:
        d = 1.0
    elif subject_x < subject_x_low:
        d = -1.0
    else:
        d = 1.0 if (order % 2 != 0) else -1.0

    if max_zoom < 1.0:
        return None

    steps = int(round((max_zoom - 1.0) / zoom_step)) + 1
    eps = 1e-6

    for i in range(steps):
        z_start = round(1.0 + i * zoom_step, 5)
        z_end = z_start * (1.0 + zoom_total)
        if z_end < 1.0 - eps:
            continue

        wm = (ww / 2.0) * (1.0 / z_start + 1.0 / z_end)
        hm = (wh / 2.0) * (1.0 / z_start + 1.0 / z_end)
        dx = -d * shift_x * wm
        dy = -shift_y * hm

        w_start = ww / (2.0 * z_start)
        h_start = wh / (2.0 * z_start)
        w_end = ww / (2.0 * z_end)
        h_end = wh / (2.0 * z_end)

        mx0 = 0.5 + 0.5 * (subject_x - 0.5)
        my0 = 0.5 + 0.5 * (subject_y - 0.5)

        mx_min = max(w_start + dx / 2.0, w_end - dx / 2.0)
        mx_max = min(1.0 - w_start + dx / 2.0, 1.0 - w_end - dx / 2.0)

        if mx_min > mx_max + 1e-9:
            continue

        mx = min(max(mx0, mx_min), mx_max)

        my_min = max(h_start + dy / 2.0, h_end - dy / 2.0)
        my_max = min(1.0 - h_start + dy / 2.0, 1.0 - h_end - dy / 2.0)

        if my_min > my_max + 1e-9:
            continue

        my = min(max(my0, my_min), my_max)

        cx_start = mx - dx / 2.0
        cy_start = my - dy / 2.0
        cx_end = mx + dx / 2.0
        cy_end = my + dy / 2.0

        if (
            cx_start - w_start >= -eps
            and cx_start + w_start <= 1.0 + eps
            and cx_end - w_end >= -eps
            and cx_end + w_end <= 1.0 + eps
            and cy_start - h_start >= -eps
            and cy_start + h_start <= 1.0 + eps
            and cy_end - h_end >= -eps
            and cy_end + h_end <= 1.0 + eps
        ):
            return {
                "type": "ken_burns",
                "easing": easing,
                "preset": preset_name,
                "start": {
                    "cx": round(cx_start, 6),
                    "cy": round(cy_start, 6),
                    "zoom": round(z_start, 6),
                    "roll_deg": 0.0,
                },
                "end": {
                    "cx": round(cx_end, 6),
                    "cy": round(cy_end, 6),
                    "zoom": round(z_end, 6),
                    "roll_deg": round(d * roll_deg * scale_factor, 4),
                },
            }

    return None


def preset_to_camera(
    preset: str,
    src_w: float,
    src_h: float,
    out_w: float,
    out_h: float,
    subject_x: float = 0.5,
    subject_y: float = 0.5,
    order: int = 0,
    max_zoom: float = 1.6,
    zoom_step: float = 0.005,
    scale_factors: tuple[float, ...] = (1.0, 0.75, 0.5),
    subject_x_high: float = 0.58,
    subject_x_low: float = 0.42,
    speed: float = 1.0,
) -> tuple[dict[str, Any], str]:
    """Construct Ken Burns camera parameters for a preset.

    Returns (camera_dict, fallback_label).
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}")

    # 1. Try original preset with scales (e.g. 1.0, 0.75, 0.5)
    for scale in scale_factors:
        label = "none" if scale == 1.0 else f"scale_{scale}"
        cam = _solve_camera(
            preset,
            src_w,
            src_h,
            out_w,
            out_h,
            subject_x,
            subject_y,
            order,
            scale,
            max_zoom=max_zoom,
            zoom_step=zoom_step,
            subject_x_high=subject_x_high,
            subject_x_low=subject_x_low,
            speed=speed,
        )
        if cam is not None:
            return cam, label

    # 2. Downgrade to push_in_gentle (if not already tried)
    if preset != "push_in_gentle":
        for scale in scale_factors:
            cam = _solve_camera(
                "push_in_gentle",
                src_w,
                src_h,
                out_w,
                out_h,
                subject_x,
                subject_y,
                order,
                scale,
                max_zoom=max_zoom,
                zoom_step=zoom_step,
                subject_x_high=subject_x_high,
                subject_x_low=subject_x_low,
                speed=speed,
            )
            if cam is not None:
                return cam, "downgrade_push_in_gentle"

    # 3. Static fallback
    static_cam = {
        "type": "ken_burns",
        "easing": "linear",
        "preset": "static",
        "start": {"cx": 0.5, "cy": 0.5, "zoom": 1.0, "roll_deg": 0.0},
        "end": {"cx": 0.5, "cy": 0.5, "zoom": 1.0, "roll_deg": 0.0},
    }
    return static_cam, "static"


def select_preset(
    idx: int,
    total_shots: int,
    role: str,
    order: int,
    prev_preset: str | None,
    cycle: tuple[str, ...] = (
        "push_slide",
        "slide",
        "push_in_strong",
        "push_in_gentle",
        "pull_back",
    ),
    first_preset: str = "push_in_gentle",
    last_preset: str = "pull_back",
    hero_preferred: str = "push_in_strong",
    hero_alternate: str = "push_slide",
    detail_preferred: str = "slide",
    detail_alternate: str = "push_in_gentle",
    room_type: str | None = None,
    lateral_run: int = 0,
    last_lateral: str | None = None,
    wide_rooms: frozenset[str] = frozenset(),
    tall_rooms: frozenset[str] = frozenset(),
    max_lateral_run: int = 1,
    tall_preferred: str = "tilt",
    tall_alternate: str = "push_in_gentle",
    hero_sweep: str = "hero_sweep",
) -> str:
    """Select a motion preset for a shot based on selection rules.

    - first shot: push_in_gentle
    - last shot: pull_back
    - wide rooms (living, kitchen, exterior, ...): slide / push_slide, broken up by a
      push-in after `max_lateral_run` lateral shots in a row (`lateral_run` counts them);
      consecutive lateral shots alternate, keyed off `last_lateral`; heroes get hero_sweep
    - tall rooms (foyer, stairs, bathrooms, ...): tilt or push_in_gentle
    - role 'hero': prefer push_in_strong or push_slide
    - role 'detail': slide or push_in_gentle
    - otherwise: cycle [push_slide, slide, push_in_strong, push_in_gentle, pull_back] by order
    - Never the same preset on two consecutive shots (take the next in the cycle)
    """
    room = canon(room_type)

    if idx == 0:
        candidate = first_preset
    elif idx == total_shots - 1:
        candidate = last_preset
    elif room in wide_rooms:
        if lateral_run >= max_lateral_run:
            candidate = hero_preferred if role == "hero" else detail_alternate
        elif role == "hero" and last_lateral != hero_sweep:
            candidate = hero_sweep
        elif last_lateral == "slide":
            candidate = "push_slide"
        else:
            candidate = "slide"
    elif room in tall_rooms:
        if prev_preset == tall_preferred:
            candidate = tall_alternate
        else:
            candidate = tall_preferred
    elif role == "hero":
        if prev_preset == hero_preferred:
            candidate = hero_alternate
        else:
            candidate = hero_preferred
    elif role == "detail":
        if prev_preset == detail_preferred:
            candidate = detail_alternate
        else:
            candidate = detail_preferred
    else:
        cycle_idx = (order - 1) % len(cycle) if order > 0 else idx % len(cycle)
        candidate = cycle[cycle_idx]

    # Never the same preset on two consecutive shots (take the next in the cycle)
    if prev_preset is not None and candidate == prev_preset:
        idx_in_cycle = cycle.index(candidate) if candidate in cycle else 0
        candidate = cycle[(idx_in_cycle + 1) % len(cycle)]

    return candidate
