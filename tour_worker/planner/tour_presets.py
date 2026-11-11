"""Tour presets (client review 2026-11-05; definitions agreed with the user 2026-11-05). A preset sets
defaults; anything the request sets itself wins (the pacing here; room labels on the backend, which
knows whether the user chose them).

  property_showcase   Normal pacing, room labels on
  just_listed         Fast pacing, "Just Listed" above the address on the title card, the music
                      preferably opening in a high-energy section
  lifestyle           Slow pacing, exteriors / outdoor living / amenities 1.5x more time,
                      room labels off"""
from __future__ import annotations

TOUR_PRESETS: dict[str, dict] = {
    "property_showcase": {"pacing": "normal", "room_labels": True},
    "just_listed": {"pacing": "fast", "room_labels": True, "kicker": "Just Listed", "open_high": True},
    "lifestyle": {"pacing": "slow", "room_labels": False, "outdoor_boost": 1.5},
}
DEFAULT_PRESET = "property_showcase"


def preset(name: str | None) -> dict:
    return TOUR_PRESETS.get(name or DEFAULT_PRESET, TOUR_PRESETS[DEFAULT_PRESET])
