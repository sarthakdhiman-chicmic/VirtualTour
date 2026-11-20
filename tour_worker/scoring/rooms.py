"""Room-type knowledge, kept as DATA so it is easy to change.

Room types come from the user's per-image selection at upload (enum or super-admin API).
Everything is keyed by a canonical slug ("Children's Bedroom" -> "childrens_bedroom"), so the
only work needed to match your real room-type list is to edit the two tables below.

TIER   = importance (SOW 5.2C): Living > Kitchen > Primary Bedroom > Primary Bathroom >
         Other Bedrooms > Other Bathrooms > Exterior > Other. The SOW names only these groups;
         the assignment of individual types to groups is OUR PROPOSAL.
STAGE  = narrative order. The SOW (5.2C: exterior -> entry -> living spaces -> private spaces ->
         exterior/amenities) as the user set it on 2026-11-20: the entrance (the front of the
         home, then the entry) -> living room -> kitchen -> dining -> the primary suite (bedroom,
         its bathroom, its closet) -> the other rooms (other bedrooms and bathrooms, office, gym,
         media, ... and photos whose room is not known) -> service (garage, laundry) -> every other
         exterior (back, terraces, balconies, pool, garden, a plain "exterior"): only the entrance
         plays before the house.
         'circulation' (hallways, stairs, 2026-11-20): a staircase or landing is how one walks
         from the primary suite to the other rooms, never where a tour opens - as the 3D tours
         treat hallways as links.
No numpy/cv2 imports here."""
import re

TIERS = ["living_room", "kitchen", "primary_bedroom", "primary_bathroom",
         "other_bedrooms", "other_bathrooms", "exterior", "other"]

STAGES = ["exterior_open", "entry", "living", "kitchen", "dining", "primary", "circulation", "other",
          "service", "amenity_close"]

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
    "exterior_open": ["front_exterior", "facade", "aerial"],
    "entry": ["entry", "entryway", "foyer", "mudroom"],
    "living": ["living_room", "family_room", "great_room", "sunroom", "open_concept"],
    "kitchen": ["kitchen", "breakfast_nook", "butlers_pantry"],
    "dining": ["dining_room", "formal_dining_room"],
    "primary": ["primary_bedroom", "master_bedroom", "primary_bathroom", "master_bathroom", "ensuite",
                "walk_in_closet"],
    "circulation": ["hallway", "staircase", "stairs", "stairway"],
    # 'bedroom' and 'bed_room' both placed here (the SA enum and the display-name slug)
    "other": ["secondary_bedroom", "guest_bedroom", "childrens_bedroom", "bedroom", "bed_room", "nursery",
              "secondary_bathroom", "powder_room", "bathroom", "closet", "home_office", "study", "library",
              "media_room", "game_room", "home_gym", "basement", "bonus_room", "other"],
    "service": ["laundry_room", "utility_room", "garage", "workshop"],
    "amenity_close": ["exterior", "backyard", "balcony", "patio", "outdoor_patio", "deck", "pool_area", "garden"],
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
    # unknown types play with the other rooms rather than being dropped
    return ROOM_STAGE.get(canon(room_type), "other")


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
