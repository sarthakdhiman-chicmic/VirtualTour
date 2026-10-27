from pydantic import BaseModel, ConfigDict


class PlannerConfig(BaseModel):
    # shot lengths (user 2026-10-19): "content" = each shot its natural length from what the photo
    # shows (planner/scene_context.py richness), no lower limit on the tour, max_total_s the hard
    # ceiling, nothing stretched to fill time. "fill" = the old fill-to-target allocation.
    length_mode: str = "content"
    normal_shot_s: tuple[float, float] = (2.8, 3.6)
    hero_shot_s: tuple[float, float] = (3.2, 5.0)
    target_total_s: float = 75.0        # "fill" only
    min_total_s: float = 0.0
    max_total_s: float = 120.0
    shot_min_s: float = 2.8
    shot_max_s: float = 5.5
    base_shot_s: float = 4.2
    detail_shot_factor: float = 0.85
    max_detail_shots_per_room: int = 2
    final_shot_min_s: float = 3.5
    min_photos: int = 5
    min_room_types: int = 2
    slideshow_max_shots: int = 20
    fps: int = 30
    output_width: int = 1920
    output_height: int = 1080
    kb_zoom_delta: float = 0.08
    kb_hero_zoom_delta: float = 0.10
    kb_final_zoom_delta: float = 0.05
    use_motion_presets: bool = True

    # Motion presets configuration
    motion_preset_cycle: tuple[str, ...] = (
        "push_slide",
        "slide",
        "push_in_strong",
        "push_in_gentle",
        "pull_back",
    )
    motion_preset_first: str = "push_in_gentle"
    motion_preset_last: str = "pull_back"
    motion_preset_hero_preferred: str = "push_in_strong"
    motion_preset_hero_alternate: str = "push_slide"
    motion_preset_detail_preferred: str = "slide"
    motion_preset_detail_alternate: str = "push_in_gentle"
    motion_preset_tall_preferred: str = "tilt"
    motion_preset_tall_alternate: str = "push_in_gentle"
    motion_max_lateral_run: int = 1
    motion_preset_hero_sweep: str = "hero_sweep"

    # Camera speed: every preset is scaled by the room's speed and by shot length relative to
    # motion_reference_shot_s, so on-screen speed is the same per second. Small rooms get the
    # slowest moves; fast moves on sharp stills step visibly from frame to frame (judder).
    motion_speed: float = 0.6
    motion_speed_small: float = 0.45
    motion_reference_shot_s: float = 5.5
    motion_small_rooms: tuple[str, ...] = (
        "bathroom", "primary_bathroom", "master_bathroom", "secondary_bathroom", "powder_room",
        "ensuite", "bedroom", "bed_room", "secondary_bedroom", "guest_bedroom",
        "childrens_bedroom", "nursery", "closet", "walk_in_closet", "laundry_room",
        "utility_room", "home_office", "study", "entry", "entryway", "foyer", "hallway",
        "mudroom", "staircase", "stairs", "stairway",
    )

    # Room-type slugs (scoring.rooms.canon) that steer motion: wide rooms favour lateral
    # moves, tall/narrow rooms favour tilts. Anything else uses the role/cycle rules.
    motion_wide_rooms: tuple[str, ...] = (
        "living_room", "family_room", "great_room", "sunroom", "kitchen", "breakfast_nook",
        "dining_room", "formal_dining_room", "open_concept", "media_room", "game_room",
        "basement", "bonus_room", "exterior", "front_exterior", "facade", "aerial", "backyard",
        "patio", "outdoor_patio", "deck", "pool_area", "garden",
    )
    motion_tall_rooms: tuple[str, ...] = (
        "entry", "entryway", "foyer", "hallway", "staircase", "stairs", "stairway",
        "bathroom", "primary_bathroom", "master_bathroom", "secondary_bathroom",
        "powder_room", "ensuite", "closet", "walk_in_closet", "laundry_room",
    )

    # Motion preset solver parameters
    motion_max_zoom: float = 1.6
    motion_zoom_step: float = 0.005
    motion_scale_factors: tuple[float, ...] = (1.0, 0.75, 0.5)
    motion_subject_x_high: float = 0.58
    motion_subject_x_low: float = 0.42

    # Camera engine. "cinematic" (plan 1.3, planner/cinematic.py): reveals that always end wider,
    # chosen per shot from room type, role, coverage class and the shot's allocated length.
    # "presets" (plan 1.2): the older ken_burns preset table (presets.py), kept as a fallback.
    motion_engine: str = "cinematic"
    # cinematic camera speed vs the reference pace (1.0 = the v8 tours; 0.8 = 20% slower peaks)
    cinematic_speed: float = 1.0   # user 2026-10-16: back to the v8 pace (was 0.8)

    # Perceptual motion (plan 1.4, planner/perceptual.py): subtle movement inside detect masks.
    perceptual_motion: bool = True
    perceptual_effects: dict[str, str] = {
        "curtain": "fabric_sway", "blinds": "fabric_sway", "potted_plant": "foliage_sway",
        "tree": "canopy_flutter", "water": "calm_water",
        # ADE20K OneFormer (2026-10-16): "plant" covers potted plants AND outdoor shrubs (still:
        # perceptual_still_outdoors); "fire" = a lit fire found inside a fireplace
        "plant": "foliage_sway", "fire": "fire_flicker",
    }
    # effects never planned in outdoor shots (shrubs and hedges stay still - user decision)
    perceptual_still_outdoors: tuple[str, ...] = ("foliage_sway",)
    perceptual_outdoor_rooms: tuple[str, ...] = (
        "exterior", "front_exterior", "facade", "aerial", "backyard", "balcony", "patio",
        "outdoor_patio", "deck", "pool_area", "garden",
    )
    perceptual_detect_max_side: int = 1600     # detect resizes larger photos to this long side
    perceptual_min_confidence: float = 0.5
    perceptual_min_area: float = 0.005        # bbox area / photo area
    perceptual_min_visible: float = 0.5       # share of the bbox inside the camera's end window
    # water is often a thin far band (a lake behind the trees): smaller and partly framed water
    # still gets the effect (review 2026-10-19: "didn't get applied to all the objects")
    # ... and flames are small (an insert's or a linear fire's flame zone is 0.05-0.6% of the
    # photo; the worker detects them only when lit, so size is no evidence against them)
    perceptual_min_area_by_effect: dict[str, float] = {"calm_water": 0.002, "fire_flicker": 0.0002}
    perceptual_min_visible_by_effect: dict[str, float] = {"calm_water": 0.25}
    # every water body and tree in a photo (review 2026-10-19: lake pieces were left out)
    perceptual_max_regions: int = 6
    # peak displacement = clamp(frac * bbox width, min, max) source px; sway frequency in Hz.
    # Effects in perceptual_amplitude_of_photo use frac * PHOTO width instead: ripples and leaves
    # look as big as the photo's scale, not the object's box. calm_water / canopy_flutter values
    # are the reviewed prototype's (pool photo, 2026-10-15), checked with time-slices.
    perceptual_amplitude: dict[str, tuple[float, float, float]] = {
        "fabric_sway": (0.012, 1.5, 6.0), "foliage_sway": (0.010, 1.0, 5.0),
        "water_ripple": (0.004, 1.0, 3.0),
        "calm_water": (0.0063, 4.0, 20.0), "canopy_flutter": (0.0028, 2.0, 10.0),
        # fire 2026-10-21 (user: "scale our flames according to the original size and the
        # firepit itself ... too much flame motion will warp the mantles"): a share of the
        # flames' own HEIGHT (perceptual_amplitude_of_height), not a fixed on-screen target
        # (screen targets 6 -> 16 -> 10 px were the 2026-10-19 rounds)
        # (0.04 with a light-led flicker was tried the same day: "a downgrade" - back to 0.12;
        # the worker adds the variety and holds the structure near the fire still)
        "fire_flicker": (0.12, 1.0, 20.0),
    }
    perceptual_amplitude_of_height: tuple[str, ...] = ("fire_flicker",)
    perceptual_amplitude_of_photo: tuple[str, ...] = ("calm_water", "canopy_flutter")
    # Curtains and plants: strength set ON SCREEN (output px at the camera's widest view of the
    # shot), so a curtain moves visibly whatever the photo size or zoom. Review 2026-10-16: on
    # Downing (1920 px photos) the bbox rule gave 0.4-1.5 px on screen - "barely noticeable";
    # the trial's 5.3 px curtain read well. Capped per object (fraction of its bbox width, so a
    # narrow curtain does not swing past its own width) and by the worker's 48 source px bound;
    # a region that cannot reach perceptual_min_screen_px within its caps is dropped (invisible
    # motion only costs render time).
    # Trees too (2026-10-16): with leaves-only weighting the photo-width rule gave 1.3-1.9 px p95
    # on Downing; 10 px peak flutter reads ~3 px p95, like the reviewed pool trees.
    perceptual_screen_target_px: dict[str, float] = {
        # trees 2026-10-19: the coherent sway reaches its peak far more often than the old noise
        # did (p95 4-8 px at 10): calmer, and the far boost smaller
        "fabric_sway": 7.0, "foliage_sway": 6.0, "canopy_flutter": 9.0,
    }
    perceptual_bbox_cap: dict[str, float] = {
        "fabric_sway": 0.08, "foliage_sway": 0.05, "canopy_flutter": 0.05, "fire_flicker": 0.4,
    }
    # distant trees and water move a bit more (review 2026-10-19: "when they are far away a bit
    # more intensive"): an object whose box is under far_height of the photo high gets up to
    # (1 + far_boost) x its strength, and its bbox cap grows with it; distance read from the
    # box height (a far tree line / lake band is thin)
    perceptual_far_effects: tuple[str, ...] = ("canopy_flutter", "calm_water")
    perceptual_far_height: float = 0.35
    perceptual_far_boost: float = 0.3
    perceptual_max_amplitude_px: float = 48.0    # source px (= worker validation bound)
    # peak px on screen; an anchored sway's p95 is ~0.4x its peak, and the quality gate wants
    # p95 >= 2 px (Downing s05: a 78 px curtain capped at 4.9 px peak read 1.9 px p95)
    perceptual_min_screen_px: dict[str, float] = {
        "fabric_sway": 5.0, "foliage_sway": 5.0,
        # leaves-only trees read lower (p95 ~0.26-0.44x peak; Turkey Trot s10: 6.6 px -> 1.7);
        # the coherent sway (2026-10-19) reads ~0.5-0.8x its peak
        "canopy_flutter": 5.0,
        "calm_water": 5.0,
        "fire_flicker": 3.0,
    }
    # water whose box spans this share of the photo's height (or more) moves at full strength
    perceptual_water_full_height: float = 0.25
    perceptual_freq_hz: dict[str, float] = {
        # plants at the curtains' rate: opposite phases only stay opposite at the same frequency
        "fabric_sway": 0.25, "foliage_sway": 0.25, "water_ripple": 0.2,
        "calm_water": 0.2, "canopy_flutter": 1.1, "fire_flicker": 1.0,
    }
    perceptual_anchor: dict[str, str] = {
        "fabric_sway": "top", "foliage_sway": "bottom", "water_ripple": "none",
        "calm_water": "none", "canopy_flutter": "none", "fire_flicker": "bottom",
    }

    # Beat sync (plan 1.5, planner/music_sync.py; rules from find_transitions.py `sync`)
    music_max_drift: float = 2.5        # s any boundary may move from its planned position
    music_length_weight: float = 4.0    # cost of changing a shot's planned length
    music_bar_weight: float = 0.3       # preference for cuts on a bar start
    music_offbeat_penalty: float = 0.1
    # drum hits (analysis v3): a cut on a hit harder than the track's typical beat earns
    # hit_weight x (hit - 1), capped; the hardest hits (~1.6x) outweigh the bar line (review
    # 2026-10-19: "a few beats way before the scene ends" - cuts sat half a bar after the hit)
    music_hit_weight: float = 1.0
    music_hit_cap: float = 1.0
    music_bar_weight_with_hits: float = 0.0
    music_fade_out_s: float = 1.5
    # hooks (user 2026-10-19: "synced to some random beat"; cuts belong on the drop / the hook):
    # whooshes and big musical changes get a bonus that outweighs any allowed length change
    music_hook_change: float = 0.6
    music_hook_bonus: float = 3.0
    music_hook_skip_start_s: float = 1.0
    # synced length per role, as a share of the planned length: heroes stay the long shots
    music_role_bounds: dict[str, tuple[float, float]] = {
        "hero": (0.8, 1.1), "detail": (0.8, 1.15),
    }
    # synced shots shrink more freely than they stretch (review 2026-10-19: shots "too long for
    # no reason")
    music_shot_abs_min_s: float = 2.5
    music_shot_abs_max_s: float = 6.0
    # retiming: the first 40% of a shot's real time plays at normal pace, the rear speeds up (or
    # slows) to land the planned move on the cut; at the 0.75 bound the rear runs ~1.56x
    music_warp_hold: float = 0.4
    # hooks first (user 2026-10-19): every hook moment gets a cut; hooks closer than merge_s are
    # one moment (one cut, the rest accents); when the role bounds cannot fit them, every shot
    # may flex to wide_bounds before hooks fall back to merely preferred
    music_hook_merge_s: float = 3.0
    music_wide_bounds: tuple[float, float] = (0.7, 1.2)
    # a shot cutting on a hook speeds into the cut: its pace rises over the last ramp_share of
    # the shot to ramp_peak x (CapCut-style "velocity" into the hit)
    music_ramp_share: float = 0.35
    music_ramp_peak: float = 1.6
    # cuts one frame before the beat (a cut exactly on the onset reads as late)
    music_cut_lead_frames: int = 1
    # zoom punches: a shot opening on a hook, and accents (hooks with no cut) mid-shot
    # zoom punches (a quick zoom in / out on a hero moment): OFF (user 2026-10-23: "we dont need
    # that rapid zoom in and zoom out for a bass or smth" -> none at all). The music is followed by
    # where the cuts land; the hook ramp into a hero cut stays (camera timing, not a zoom)
    music_punches: bool = False
    music_punch_cut: float = 0.025
    music_punch_accent: float = 0.02
    music_punch_attack_s: float = 0.067
    music_punch_decay_s: float = 0.35
    music_accent_edge_s: float = 0.4     # no accent this close to a shot's ends
    music_accent_min_gap_s: float = 2.0  # ... nor this close to a cut or another accent
    # beat sync v2 (2026-10-23, planner/music_sync_v2.py; needs analysis v4 heroes, else v3)
    music_sync_version: int = 2
    music_v2_tier1: tuple[float, float] = (10.0, 4.0)   # cut value: base + x strength
    music_v2_tier2: tuple[float, float] = (3.0, 2.0)
    music_v2_taken_bonus: float = 2.0    # a tier-2 hero picked for a cut (not left for a punch)
    music_v2_phrase: float = 0.4         # filler: a phrase start ...
    music_v2_bar: float = 0.15           # ... a bar start
    # cut quality: + strike x clip(strike - 1, 0, 1) (how hard the sound hits at the cut: below
    # 1.0, cuts fell a beat after a strong hit onto the phrase line) - pre x clip(pre - strike, 0,
    # 1.5) (a harder sound up to a beat before: "delayed after the ting")
    music_v2_strike: float = 1.0
    # a cut on a beat that does not stand out from the music around it (clarity below CLEAR in
    # music_sync_v2): up to this much less value, and a start whose first cuts sit in a soft intro
    # loses to one where they are clear (review 2026-10-23: "the first 3 frames are really off")
    # (regression set: 0 -> 54 unclear cuts, 4 -> 23, 8 -> 16, 12 -> 12; from 8 on, Turkey Trot x
    # solarflex opens on the breakdown with every cut clear; the tours fit 18 drops instead of 22,
    # every one of them still cut)
    music_v2_unclear: float = 8.0
    # per LATE_RATIO step of pre / strike above 1 (up to 3x). Regression set 2026-10-23 (16 tours x
    # tracks): 1.0 -> 18 late cuts, 2.0 -> 10-12 for 2 points of hero coverage (0.979 -> 0.958)
    music_v2_pre: float = 2.0
    music_v2_hero_gap_s: float = 2.5     # heroes closer than this to a cut hero get a punch
    # opening (review 2026-10-23: the tour opened 1.65 s before a breakdown, on a loud tail)
    music_v2_open_on_change: float = 8.0     # the first frame on a change of the music
    music_v2_open_natural: float = 5.0       # the track's own beginning
    music_v2_open_before_change: float = 6.0 # a change this soon after the first frame ...
    music_v2_open_before_s: float = 3.0      # ... within this many seconds: penalised
    # music edits (user 2026-10-23: "trim the music ... according to our music profile")
    music_v2_max_edits: int = 2
    music_v2_splice_max: float = 0.015       # bar-print distance of a join (neighbouring bars: ~0.003-0.011)
    music_v2_edit_cost: float = 2.0          # per edit, + 100 x its splice cost
    music_v2_crossfade_s: float = 0.03
    music_v2_start_share: float = 0.45   # the music may start at a phrase in this first share
    music_v2_start_cost: float = 0.02    # per s of skipped music (earliest start on near ties)
    music_v2_early_drop_bonus: float = 4.0   # the first drop opens shot 2-4
    music_v2_punch_tier1: float = 0.03
    music_v2_punch_tier2: tuple[float, float] = (0.015, 0.01)   # base + x strength

    # sway phases: curtains at the shot's base phase, plants half a cycle later; each further
    # object of the same kind a little later still, so none move in lockstep
    perceptual_phase_step: float = 0.6

    version: str = "3.0.0"

    model_config = ConfigDict(frozen=True)
