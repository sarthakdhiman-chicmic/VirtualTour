"""Room-type knowledge, kept as DATA so it is easy to change.

Room types come from the user's per-image selection at upload (enum or super-admin API).
Everything is keyed by a canonical slug ("Children's Bedroom" -> "childrens_bedroom"), so the
only work needed to match your real room-type list is to edit the two tables below.

TIER   = importance (SOW 5.2C): Living > Kitchen > Primary Bedroom > Primary Bathroom >
         Other Bedrooms > Other Bathrooms > Exterior > Other. The SOW names only these groups;
         the assignment of individual types to groups is OUR PROPOSAL.
STAGE  = narrative order (SOW 5.2C): exterior -> entry -> living spaces -> private spaces ->
         exterior/amenities. 'service' (garage, laundry) is our addition; the SOW is silent.
No numpy/cv2 imports here."""
import re

TIERS = ["living_room", "kitchen", "primary_bedroom", "primary_bathroom",
         "other_bedrooms", "other_bathrooms", "exterior", "other"]

STAGES = ["exterior_open", "entry", "living", "private", "service", "amenity_close"]

_TIER_MEMBERS: dict[str, list[str]] = {
    # --- SA live room types (roomTypeEnum values) ---
    # 'living_room'  -> roomTypeEnum=living_room  (display: "Living Room")
    # 'bedroom'      -> roomTypeEnum=bedroom       (display: "Bed Room")
    # --- full slug vocabulary ---
    "living_room": ["living_room", "family_room", "great_room", "sunroom", "open_concept"],
    "kitchen": ["kitchen", "breakfast_nook", "butlers_pantry"],
    "primary_bedroom": ["primary_bedroom", "master_bedroom"],
    "primary_bathroom": ["primary_bathroom", "master_bathroom", "ensuite"],
    # 'bedroom' and 'bed_room' both map here so the SA enum ('bedroom') and
    # the display-name slug ('bed_room' from canon("Bed Room")) both resolve.
    "other_bedrooms": [
        "secondary_bedroom", "guest_bedroom", "childrens_bedroom",
        "bedroom", "bed_room", "nursery",
    ],
    "other_bathrooms": ["secondary_bathroom", "powder_room", "bathroom"],
    "exterior": ["exterior", "front_exterior", "facade", "aerial", "backyard", "balcony",
                 "patio", "outdoor_patio", "deck", "pool_area", "garden"],
}

_STAGE_MEMBERS: dict[str, list[str]] = {
    "exterior_open": ["exterior", "front_exterior", "facade", "aerial"],
    "entry": ["entry", "entryway", "foyer", "hallway", "mudroom", "staircase", "stairs", "stairway"],
    "living": ["living_room", "family_room", "great_room", "sunroom", "kitchen", "breakfast_nook",
               "butlers_pantry", "dining_room", "formal_dining_room", "open_concept", "home_office", "study",
               "library", "media_room", "game_room", "home_gym", "basement", "bonus_room"],
    # 'bedroom' and 'bed_room' both placed in private stage
    "private": [
        "primary_bedroom", "master_bedroom", "secondary_bedroom", "guest_bedroom",
        "childrens_bedroom", "bedroom", "bed_room", "nursery",
        "primary_bathroom", "master_bathroom",
        "ensuite", "secondary_bathroom", "powder_room", "bathroom",
        "closet", "walk_in_closet",
    ],
    "service": ["laundry_room", "utility_room", "garage", "workshop"],
    "amenity_close": ["backyard", "balcony", "patio", "outdoor_patio", "deck", "pool_area", "garden"],
}

ROOM_TIER = {r: t for t, rs in _TIER_MEMBERS.items() for r in rs}
ROOM_STAGE = {r: s for s, rs in _STAGE_MEMBERS.items() for r in rs}


def canon(name: str | None) -> str:
    if not name:
        return ""
    s = name.lower().replace("'", "").replace("\u2019", "")
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def tier_for(room_type: str | None) -> str:
    return ROOM_TIER.get(canon(room_type), "other")


def stage_for(room_type: str | None) -> str:
    # unknown types are placed with the living spaces rather than dropped
    return ROOM_STAGE.get(canon(room_type), "living")


def is_mapped(room_type: str | None) -> bool:
    c = canon(room_type)
    return c in ROOM_TIER or c in ROOM_STAGE


def tier_index(tier: str) -> int:
    return TIERS.index(tier)


def stage_index(stage: str) -> int:
    return STAGES.index(stage)


def room_key_for(room_type: str | None, room_label: str | None = None) -> str:
    """One room per (type, optional label). Without a label, all photos of a type form ONE room,
    so two 'Secondary Bedroom' rooms merge and share a single hero shot."""
    base = canon(room_type) or "unknown"
    lab = canon(room_label)
    return f"{base}::{lab}" if lab else base
