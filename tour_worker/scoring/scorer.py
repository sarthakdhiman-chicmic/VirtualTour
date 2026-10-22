"""Combine the four criteria into one score per photo. Criteria that cannot be measured are
dropped, remaining weights are renormalised, and the photo is flagged (never silent)."""
import cv2
import numpy as np

from .config import ScoringConfig
from .coverage import score_coverage
from .depth_of_field import score_depth_of_field
from .framing import score_framing
from .imaging import resize_long_side
from .inputs import PhotoInput
from .lighting import score_lighting
from .motion import motion_signals
from .rooms import is_mapped, room_key_for
from .schemas import CRITERIA, CriterionResult, PhotoScore


def score_photo(photo: PhotoInput, cfg: ScoringConfig) -> PhotoScore:
    height, width = photo.image.shape[:2]
    img = resize_long_side(photo.image, cfg.analysis_long_side)
    gray_u8 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray01 = gray_u8.astype(np.float32) / 255.0

    criteria: dict[str, CriterionResult] = {
        "coverage": score_coverage(gray_u8, photo.focal_35mm, cfg.coverage),
        "framing": score_framing(gray01, gray_u8, cfg.framing),
        "lighting": score_lighting(gray01, cfg.lighting),
        "depth_of_field": score_depth_of_field(gray_u8, photo.depth, cfg.depth_of_field),
    }

    full_gray01 = cv2.cvtColor(photo.image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    motion = motion_signals(full_gray01, photo.room_type,
                            criteria["coverage"].details.get("fov_score"), cfg.motion)

    weights = cfg.weights.model_dump()
    available = {k: criteria[k].score for k in CRITERIA if criteria[k].score is not None}
    missing = [k for k in CRITERIA if k not in available]
    wsum = sum(weights[k] for k in available)
    used = {k: round(weights[k] / wsum, 4) for k in available} if wsum > 0 else {}
    total = sum(used[k] * available[k] for k in available)

    flags: list[str] = []
    if total < cfg.min_acceptable_score:
        flags.append("low_score")
    flags += [f"missing_{k}" for k in missing]
    if not photo.room_type:
        flags.append("no_room_type")
    elif not is_mapped(photo.room_type):
        flags.append("unmapped_room_type")

    return PhotoScore(
        photo_id=photo.photo_id, room_type=photo.room_type, room_label=photo.room_label,
        room_key=room_key_for(photo.room_type, photo.room_label),
        criteria=criteria, weights_used=used, missing_criteria=missing,
        total=round(float(total), 4),
        image_width=width, image_height=height,
        motion=motion,
        flags=flags,
    )
