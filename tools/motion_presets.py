from typing import Tuple, Dict, Any, Optional

# Mirror of the backend planner's PRESETS (articulait_backend
# app/domains/virtual_tour/planner/presets.py) - keep the two in sync; the rationale for the
# values lives there. Lateral presets carry their own max_zoom (upscale softness cap).
PRESETS = {
    "push_in_gentle": {"zoom_total": 0.12, "shift_x": -0.09, "shift_y": -0.03, "easing": "linear", "roll_deg": 0.5},
    "push_in_strong": {"zoom_total": 0.20, "shift_x": -0.07, "shift_y": 0.05, "easing": "linear", "roll_deg": 0.8},
    "push_slide":     {"zoom_total": 0.10, "shift_x": -0.18, "shift_y": -0.02, "easing": "linear", "roll_deg": 1.0, "max_zoom": 1.35},
    "slide":          {"zoom_total": -0.08, "shift_x": -0.20, "shift_y": 0.00, "easing": "linear", "roll_deg": 1.2, "max_zoom": 1.35},
    "hero_sweep":     {"zoom_total": 0.16, "shift_x": -0.18, "shift_y": -0.04, "easing": "linear", "roll_deg": 1.5, "max_zoom": 1.35},
    "tilt":           {"zoom_total": 0.06, "shift_x": 0.00, "shift_y": 0.14, "easing": "linear", "roll_deg": 0.4},
    "pull_back":      {"zoom_total": -0.13, "shift_x": -0.10, "shift_y": -0.04, "easing": "linear", "roll_deg": 0.6},
}

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
    speed: float = 1.0,
) -> Optional[Dict[str, Any]]:
    config = PRESETS[preset_name]
    # speed scales the whole move (zoom, travel and roll); mirrors the backend planner
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

    if subject_x > 0.58:
        d = 1.0
    elif subject_x < 0.42:
        d = -1.0
    else:
        d = 1.0 if (order % 2 != 0) else -1.0

    if max_zoom < 1.0:
        return None

    steps = int(round((max_zoom - 1.0) / 0.005)) + 1
    eps = 1e-6

    for i in range(steps):
        z_start = round(1.0 + i * 0.005, 5)
        z_end = z_start * (1.0 + zoom_total)
        if z_end < 1.0 - eps:
            continue

        Wm = (ww / 2.0) * (1.0 / z_start + 1.0 / z_end)
        Hm = (wh / 2.0) * (1.0 / z_start + 1.0 / z_end)
        dx = -d * shift_x * Wm
        dy = -shift_y * Hm

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

        if (cx_start - w_start >= -eps and cx_start + w_start <= 1.0 + eps and
            cx_end - w_end >= -eps and cx_end + w_end <= 1.0 + eps and
            cy_start - h_start >= -eps and cy_start + h_start <= 1.0 + eps and
            cy_end - h_end >= -eps and cy_end + h_end <= 1.0 + eps):
            return {
                "type": "ken_burns",
                "easing": easing,
                "preset": preset_name,
                "start": {
                    "cx": round(cx_start, 6),
                    "cy": round(cy_start, 6),
                    "zoom": round(z_start, 6),
                    "roll_deg": 0.0
                },
                "end": {
                    "cx": round(cx_end, 6),
                    "cy": round(cy_end, 6),
                    "zoom": round(z_end, 6),
                    "roll_deg": round(d * roll_deg * scale_factor, 4)
                }
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
    speed: float = 1.0,
) -> Tuple[Dict[str, Any], str]:
    """
    Construct Ken Burns camera parameters for a preset.
    Returns (camera_dict, fallback_label).
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}")

    # 1. Try original preset with scale 1.0, 0.75, 0.5
    for scale, label in [(1.0, "none"), (0.75, "scale_0.75"), (0.5, "scale_0.5")]:
        cam = _solve_camera(preset, src_w, src_h, out_w, out_h, subject_x, subject_y, order, scale, max_zoom=max_zoom, speed=speed)
        if cam is not None:
            return cam, label

    # 2. Downgrade to push_in_gentle (if not already tried)
    if preset != "push_in_gentle":
        for scale in [1.0, 0.75, 0.5]:
            cam = _solve_camera("push_in_gentle", src_w, src_h, out_w, out_h, subject_x, subject_y, order, scale, max_zoom=max_zoom, speed=speed)
            if cam is not None:
                return cam, "downgrade_push_in_gentle"

    # 3. Static fallback
    static_cam = {
        "type": "ken_burns",
        "easing": "linear",
        "preset": "static",
        "start": {"cx": 0.5, "cy": 0.5, "zoom": 1.0, "roll_deg": 0.0},
        "end": {"cx": 0.5, "cy": 0.5, "zoom": 1.0, "roll_deg": 0.0}
    }
    return static_cam, "static"
