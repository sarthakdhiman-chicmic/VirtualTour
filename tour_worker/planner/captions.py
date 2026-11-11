"""Room labels (user 2026-11-05): each room's first shot carries its display name as the shot's
caption; tour_worker/render/overlays.py draws it. A room is a run of consecutive shots with the
same room_key, so a room filmed in several shots is labelled once. Staged photos (virtually
staged generations, as the backend marks them) carry staged=True: the renderer draws the
"Virtually Staged" disclosure on them, captions or not."""
from __future__ import annotations

from tour_worker.scoring.rooms import canon

# every room type the backend allows (app/common/enums.py TourRoomType), as a viewer reads it
ROOM_LABELS = {
    "living_room": "Living Room", "family_room": "Family Room", "great_room": "Great Room",
    "sunroom": "Sunroom", "dining_room": "Dining Room", "formal_dining_room": "Formal Dining",
    "open_concept": "Living Space", "kitchen": "Kitchen", "breakfast_nook": "Breakfast Nook",
    "butlers_pantry": "Butler's Pantry",
    "primary_bedroom": "Primary Suite", "master_bedroom": "Primary Suite",
    "secondary_bedroom": "Bedroom", "guest_bedroom": "Guest Bedroom", "childrens_bedroom": "Bedroom",
    "bedroom": "Bedroom", "nursery": "Nursery",
    "primary_bathroom": "Primary Bath", "master_bathroom": "Primary Bath", "ensuite": "Ensuite",
    "secondary_bathroom": "Bathroom", "powder_room": "Powder Room", "bathroom": "Bathroom",
    "entry": "Entry", "entryway": "Entry", "foyer": "Foyer", "hallway": "Hallway", "mudroom": "Mudroom",
    "staircase": "Staircase", "stairs": "Staircase", "stairway": "Staircase",
    "home_office": "Home Office", "study": "Study", "library": "Library", "media_room": "Media Room",
    "game_room": "Game Room", "home_gym": "Home Gym", "basement": "Basement", "bonus_room": "Bonus Room",
    "exterior": "Exterior", "front_exterior": "The Residence", "facade": "The Residence",
    "aerial": "Aerial View", "backyard": "Backyard", "balcony": "Balcony", "patio": "Patio",
    "outdoor_patio": "Outdoor Patio", "deck": "Deck", "pool_area": "Pool", "garden": "Garden",
    "laundry_room": "Laundry", "utility_room": "Utility Room", "garage": "Garage", "workshop": "Workshop",
    "closet": "Closet", "walk_in_closet": "Walk-in Closet",
}


def room_label(room_type: str | None) -> str | None:
    """The display name of a room type; an unknown type is title-cased, none / other gets none."""
    c = canon(room_type)
    if not c or c in ("other", "unknown"):
        return None
    return ROOM_LABELS.get(c) or c.replace("_", " ").title()


def apply_captions(shots, labels: bool, staged_photos: frozenset[str] = frozenset()) -> None:
    """In place: caption on the first shot of each room (when labels are on), staged on the shots
    of staged photos. Shots without either keep None, so their JSON is unchanged."""
    prev_room = None
    for shot in shots:
        if labels and shot.room_key != prev_room:
            shot.caption = room_label(shot.room_type)
        prev_room = shot.room_key
        if shot.asset_id in staged_photos:
            shot.staged = True
