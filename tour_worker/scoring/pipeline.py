from .config import ScoringConfig
from .inputs import PhotoInput
from .schemas import ScoringReport
from .scorer import score_photo
from .selector import build_report


def analyze(photos: list[PhotoInput], cfg: ScoringConfig | None = None,
            warnings: list[str] | None = None) -> ScoringReport:
    """Score every photo (needs OpenCV), then pick heroes and order rooms (pure Python)."""
    cfg = cfg or ScoringConfig()
    scores = [score_photo(p, cfg) for p in sorted(photos, key=lambda p: p.photo_id)]
    return build_report(scores, cfg, warnings)


def score_photos(inputs: list[tuple[str, bytes, str | None]],
                 cfg: ScoringConfig | None = None) -> ScoringReport:
    """Score encoded photos: (photo_id, image bytes, room_type) each. CPU-bound (OpenCV).
    The backend's scoring_service._run_scoring, moved here with the tour_plan job."""
    from .imaging import photo_from_bytes
    photos = [photo_from_bytes(photo_id, data, room_type=room_type)
              for photo_id, data, room_type in inputs]
    return analyze(photos, cfg or ScoringConfig())
