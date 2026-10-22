from .schemas import CameraParams, CameraPoint


def clamp(val: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, val))


def generate_ken_burns(
    source_w: int,
    source_h: int,
    output_w: int,
    output_h: int,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    zoom_start: float,
    zoom_end: float,
) -> tuple[CameraParams, str | None]:
    """
    Generate Ken Burns camera parameters.

    Returns (CameraParams, warning_string_if_portrait).
    """
    warning = "portrait_source_cropped" if source_w < source_h else None

    # Base aspect ratios
    source_aspect = source_w / source_h
    output_aspect = output_w / output_h

    # At zoom 1.0, what fraction of the source image is visible?
    if source_aspect > output_aspect:
        # Source is wider than output. Window height is 1.0 of source, width is scaled.
        base_h = 1.0
        base_w = output_aspect / source_aspect
    else:
        # Source is taller than output (or equal). Window width is 1.0 of source, height is scaled.
        base_w = 1.0
        base_h = source_aspect / output_aspect

    def compute_clamped_center(target_cx: float, target_cy: float, z: float) -> CameraPoint:
        # At zoom z, the visible window is (base_w / z, base_h / z)
        win_w = base_w / z
        win_h = base_h / z

        # Clamp center to ensure the window lies entirely within [0, 1]x[0, 1]
        min_cx, max_cx = win_w / 2.0, 1.0 - (win_w / 2.0)
        min_cy, max_cy = win_h / 2.0, 1.0 - (win_h / 2.0)

        # If the window is somehow larger than the image
        # (shouldn't happen for z >= 1.0), lock to center
        if min_cx > max_cx:
            cx = 0.5
        else:
            cx = clamp(target_cx, min_cx, max_cx)

        if min_cy > max_cy:
            cy = 0.5
        else:
            cy = clamp(target_cy, min_cy, max_cy)

        return CameraPoint(cx=round(cx, 4), cy=round(cy, 4), zoom=round(z, 4))

    c_start = compute_clamped_center(start_x, start_y, zoom_start)
    c_end = compute_clamped_center(end_x, end_y, zoom_end)
    return CameraParams(
        type="ken_burns", easing="ease_in_out", start=c_start, end=c_end
    ), warning
