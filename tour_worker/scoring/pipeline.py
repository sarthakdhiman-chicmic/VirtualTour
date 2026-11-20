from .config import ScoringConfig
from .inputs import PhotoInput
from .schemas import ScoringReport
from .scorer import score_photo
from .selector import build_report


def analyze(photos: list[PhotoInput], cfg: ScoringConfig | None = None,
            warnings: list[str] | None = None) -> ScoringReport:
    """Score every photo (needs OpenCV), then pick heroes and order rooms (pure Python)."""
    from .duplicates import find_duplicates, thumb
    cfg = cfg or ScoringConfig()
    photos = sorted(photos, key=lambda p: p.photo_id)
    scores = [score_photo(p, cfg) for p in photos]
    # the same picture twice is shown once (scoring/duplicates.py)
    dup = find_duplicates({p.photo_id: thumb(p.image) for p in photos if p.image is not None},
                          {s.photo_id: s.total for s in scores})
    if dup:
        scores = [s for s in scores if s.photo_id not in dup]
        warnings = list(warnings or []) + [f"duplicate_photo:{a}:{b}" for a, b in sorted(dup.items())]
    return build_report(scores, cfg, warnings)


def score_photos(inputs: list[tuple[str, bytes, str | None]],
                 cfg: ScoringConfig | None = None) -> ScoringReport:
    """Score encoded photos: (photo_id, image bytes, room_type) each. CPU-bound (OpenCV).
    The backend's scoring_service._run_scoring, moved here with the tour_plan job."""
    from .imaging import photo_from_bytes
    photos = [photo_from_bytes(photo_id, data, room_type=room_type)
              for photo_id, data, room_type in inputs]
    return analyze(photos, cfg or ScoringConfig())
