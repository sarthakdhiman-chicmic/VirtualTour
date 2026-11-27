"""Camera moves for one shot of a room, generated from the room's free floor and checked against the
tour rules (user reviews 2026-11-12/13), then scored; the best one that passes wins.

Kinds of move (the green-house v3 tour is the standard, user 2026-11-13):
  walkway  a room shot: along a free path in the room (or its open-plan space), looking at the
           room's main piece of furniture: the piece's top surface stays in frame all along, seen
           three-quarter-on from a sitting-back distance; moving across the view gives parallax
  push     a room shot for a room with no piece to compose on but with its own working surfaces
  pull     (a kitchen's counter run, sink, hob): a slow move toward them (push) or back from them
           (pull - preferred: the view widens), aimed at them
  reveal   the opening shot from a room with no piece (a hall): along the hall toward the next
           room of the tour, the view turning from down the path onto that room
  pan      the closing shot: from an open spot of the last room, a slow sweep looking back through
           the home (its deepest view)
  glide    the fallback when none of the above exists: along a free path into the room's deepest view
  link     between two rooms a hallway joins (user 2026-11-13: "from dining pan onto the hallway,
           then move into the bedroom"): from the room being left, near where the way out is, a
           slow pan from that room onto the hallway, toward the next room
  enter    the next room's first shot after a link: along the walking route from the hallway
           through the door into the room, looking at its main piece
  hold     the fallback when every move runs into something near the lens: the least blocked framed
           spot with a breath of pull-back, and a warning
Never as room shots: pans (they sweep past what a room holds) or aims above eye level (a ceiling,
upper cabinets).

Rules every move is checked against:
  speed    peak turn <= MAX_TURN_DPS and travel <= MAX_MOVE_MPS as the worker draws the path
           (easing included), planned to PRE_SYNC_SHARE of them so the beat sync cannot push it over
  horizon  pitch at most PITCH_LADDER[0] below level (a small room may step down the ladder)
  zoom     one field of view for the whole tour (HFOV), no zoom moves
  framing  the piece's top surface in frame (as much of its long side as fits: CORES)
  lens     nothing near the lens (OCC_NEAR_KIND m) covers more than MAX_OCC of the frame's middle
           (rendered)
  cuts     no jump cut from the previous shot (JUMP_ANGLE / JUMP_DIST) and no repeat of an earlier
           framing of the same room
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from tour_worker.scene.camera import BezierCameraPath, intrinsics, look_at

EYE = 1.50                    # eye height (m)
LOOK_DROP = 0.30              # a level look drops this much over ~2.5 m
HFOV = 78.0                   # one field of view for the whole tour
MAX_TURN_DPS, MAX_MOVE_MPS = 10.0, 0.8
# moves are planned to this share of the limits: the beat sync only lengthens room and hero shots
# (planner/lengths.SYNC_BOUNDS), but may shorten a close-up to 0.7x
PRE_SYNC_SHARE = {"room": 0.9, "hero": 0.9, "detail": 0.65}
PEAK_STEADY = 1.1           # continuous cuts: a steady move's peak over its mean speed   # 0.9: the planner's 91-sample check vs per frame
PITCH_LADDER = (10.0, 12.0)     # (13 / 16 read as looking at the floor - Home Scan v11)
JUMP_ANGLE, JUMP_DIST = 30.0, 1.5
RMIN_LADDER = (0.28, 0.22)    # camera body clearance, a little less in small rooms
MOVE_LEN = (0.6, 2.6)
CORES = (0.75, 0.6, 0.5, 0.4)
SIT_BACK = 2.4                # m from the piece at the start of a walkway move
OCC_NEAR, MAX_OCC, OCC_W, OCC_CHECK_KIND = 2.0, 0.20, 3.0, 30   # (0.15 tried for item 17: the Home Scan lost 3 rooms)
SHOT_CHECKS = 60              # rendered checks (lens / void) a further angle's choice may spend over all its tiers
OCC_CACHE_M = 0.05            # m: poses this close share one rendered check
FIRST_SHOT_CHECKS = 160       # ... and a room's first shot (60 skipped three Home Scan rooms, 2026-11-26)
                              # (160 until 2026-11-26: the Home Scan's plan took 19 min on the pod)
FLIGHT_CHECKS = 40            # ... and a room change's flight, its own (it used to get what the room's
                              # last shot left - often nothing, and the flight failed unchecked)
LATE_CHECKS = 6               # past the planning budget (plan.PLAN_BUDGET_S): a further angle's / flight's last few checks
OPENING_GLIDE = 1.5          # the tour opens by moving into the home
# near-lens distance per move: a walkway composes on a piece, so nothing may stand within 2 m in
# front of it; a glide or pan looks down a room, where walls 1.2 m+ away are the room itself
# (user 2026-11-16, Home Scan: shots closing in on walls the floor plan does not hold - a poorly
# captured wall is no "tall" cell - so the rendered check looks 1.8 m out for every room move)
OCC_NEAR_KIND = {"backout": 0.5, "door_pan": 0.5, "loop": 0.9, "bridge": 0.9, "wall_pan": 1.5, "corner_sweep": 1.8, "panorama": 1.5, "descend": 1.8, "truck": 2.0, "rise": 1.8, "exit": 0.9, "through": 0.6, "walk": 0.9, "walkway": 2.0, "arc": 2.0, "room_pan": 1.8, "hold_pan": 1.8, "push": 1.8, "pull": 1.8, "reveal": 1.8, "link": 1.2, "enter": 1.5, "glide": 1.8, "pan": 1.2, "detail": 0.7}
PAN_SWEEP = 50.0
AIM_Z_MAX = EYE - 0.10        # a room shot never looks up: not at a ceiling, not at upper cabinets
LOW_TOP = 1.4                 # m: an object whose top is below this is a surface a room shot may aim at
# what a room without a main piece is about: its working surfaces (low objects of these classes)
ANCHORS = {"kitchen": ("countertop", "counter", "sink", "stove", "oven", "dishwasher", "kitchen island", "cabinet"),
           "bathroom": ("bathtub", "sink", "shower", "countertop", "cabinet"),
           "ensuite": ("bathtub", "sink", "shower", "countertop", "cabinet"),
           "laundry": ("washer", "cabinet", "countertop", "sink"),
           "home_office": ("desk", "table"), "dining_room": ("table",), "living_room": ("sofa", "coffee table"),
           "bedroom": ("bed",), "primary_bedroom": ("bed",), "childrens_bedroom": ("bed",), "guest_bedroom": ("bed",)}
PUSH_LEN = (0.6, 2.2)
PUSH_END = (1.3, 3.2)         # m from the anchor where a push ends
REVEAL_SHARE = 0.35           # of the reveal's last view on the next room
CLOSING_DEPTH = 3.0           # m: a closing pan that sees less than this is no look-back
LINK_BEFORE = (1.2, 4.0)      # m along the route before the hallway where a link pan stands
LINK_AHEAD = 2.5              # m into the hallway a link pan ends looking at
LINK_SWING = (15.0, 50.0)     # deg a link pan turns
LINK_DRIFT = 0.4              # m it drifts toward the hallway
ENTER_BEFORE = (0.6, 1.8)     # m before the doorway an enter shot starts
ENTER_AFTER = (0.8, 2.6)      # m past it (straight-line) it ends
ENTER_SHARE = 0.4             # of its first view on the room it enters
# room to breathe (user 2026-11-13: "to keep the length of a shot it goes too near the walls - going
# back and wider is better"): with one field of view, closing in on a wall reads as a zoom
FRONT_MIN = 2.2               # m to the nearest wall or tall thing across the middle third of the frame,
BREATHE_MIN = 1.3             # m: the least room to breathe, in the smallest rooms
BREATHE_SHARE = 0.4           # x the room's side (sqrt of its footprint), up to FRONT_MIN / AIM_MIN
FRONT_MIN_ENTER = 1.5         # at a room shot's start and end (an enter starts in a doorway)
AIM_MIN = 2.0                 # m from what a room shot aims at, at its end
OPEN_IN_W = 1.5               # the opening shot: score per m it closes in on its subject
WIDE_W = 0.6                  # score per m the view opens up from start to end (pulling back)
# coverage (user 2026-11-16: "each room should get up to 3 shots so all its walls get covered"):
# a room's further shots must each show this share of its walls the earlier ones did not, and the
# more new wall a candidate shows the better it scores
COVER_NEW_MIN = 0.15
COVER_W = 3.0
FRAME_MARGIN = 0.03          # the subject's box this far inside the frame's edges ...
BED_SEEN = 0.7               # a bed too big for the frame from a tight room's free floor: this share of its
                             # top in frame, its headboard end included (green house v15: the primary
                             # bed filled the room - a 1 m aisle beside it - and no shot framed it)
FAR_FLOOR = ("table", "kitchen island")   # framed whole with the floor beyond them (frames_on_far_floor)
BED_SEEN_PATH = 0.55         # ... and this share all along a move onto it
BED_AIM = 0.35               # x the bed's half length: a shot onto a bed aims this far toward its head
FLOOR_MARGIN = 0.04          # ... and the floor under it this far above the frame's bottom (a strip of floor)
MIN_SCALE_PS = 0.035          # or a push's subject grows (ln of its distance ratio) this much a second
MIN_TRAVEL_PS = 0.12         # m/s: a room shot travels at least this much for its length ...
MIN_TURN_PS = 2.5            # ... or turns this much (deg/s) - else it reads as a still (user 2026-11-16, Krk)
# planning at the tour's 3D pace (RoomMoves.to_pace, speed_band.BAND_3D)
PACE_MIN_KEEP = 0.25          # a move keeps at least this share of its path when slowed to the pace
PACE_SLOW_RATIO = 1.2         # slower than pace / this: it holds the next cut down ...
PACE_SLOW_W = 4.0             # ... and loses this score at no speed at all (1.5 let slow pans win, 2026-11-27)
ON_PIECE_W = 0.3              # a room's first shot walking / arcing on its bed or sofa
PACE_SEEN_MARGIN = 1.05       # a paced move keeps this much over the 'seen to move' floor ...
PACE_SEEN_KINDS = ("door_pan",)   # ... these moves (a pan held in a doorway: trimmed under it, nothing passed)
PACE_PEAK_SHARE = 0.85        # a paced move's peak turn / travel under this share of the caps
PACE_CUT_RATIO = 1.2          # a move enters within this of the previous shot's exit speed (the band's PLAN_RATIO)
DOOR_LEAD = 0.6               # m before a room's door a walk in through it always starts (paced or shortened)
TILT_MAX = 12.0              # deg: a push may tilt down this far at its end to frame its subject with floor
KEEP_PER_KIND = 40            # a room's arcs / pans judged further: the best by their own score
OWN_POINTS = 16              # candidate points spread over a room's own floor, besides the space's
DOOR_NEAR = 0.4             # m: a room shot never starts or ends this close to a doorway
WALL_CLEAR = 0.2             # m: nor pressed against a wall
DOOR_CLEAR = 0.45            # m: a room shot never starts or ends in a doorway (clearance under this)
EDGE_NEAR, EDGE_MAX = 0.8, 0.12   # anything within EDGE_NEAR m covering EDGE_MAX of the whole frame blocks it
NEAR_ANY = 1.0                # m: anything this close to the lens ...
NEAR_ANY_MAX = 0.25           # ... over this share of the frame, at any of NEAR_ANY_POSES, fails a move
                              # (0.15 tried 2026-11-26 for item 17: the Home Scan skipped three rooms)
NEAR_ANY_MAX_THROUGH = 0.25   # a flight through a door passes its frame
NEAR_ANY_POSES = 3            # (5 until 2026-11-26: renders per candidate, the planning time)
VOID_MAX = 0.12              # at most this share of a view may be what the scan never captured
CEIL_PITCH = 0.0             # deg the aim drops when the scan has no ceiling (its black band stays out)
ANCHOR_HALF = 0.8           # m: the working area a push / pull must show whole
ARC_R = (2.3, 2.7, 3.1, 3.5)
ARC_SWEEP = (30.0, 40.0, 50.0, 65.0)     # an orbit: wider sweeps where the speed limit allows
HOLD_SWEEP = 25.0
NO_AIM = ("door", "curtain", "windowpane", "blind")   # never what a room shot ends on
ACROSS_ZONE_MAX = 0.25        # a room's first shot looking more than this share across another zone goes last
END_CONE_DEG = 12.0           # the middle of the view a shot ends on (ends_on_door)
DOOR_END_W = 2.0              # score a room's first shot loses ending on a door (an extra angle: dropped)
REVEAL_SWING = 40.0           # deg: the most a reveal turns (from down its path onto the next room)
SAMPLES = 50
# scene-picked moves and the flow between rooms (user 2026-11-16: "a multitude of camera moves
# selected according to the scenes ... use the floor plan by moving toward the next room, more a
# drone shot vibe"; decided: a rise may reach RISE_Z with RISE_PITCH down, only in open or large
# rooms; doorways are cuts with matched motion)
TRUCK_LEN = (1.2, 1.7, 2.2)   # m a truck slides sideways ...
TRUCK_DIST = (2.3, 2.8, 3.3)  # ... this far out from its subject ...
TRUCK_AIM = 0.4               # ... its aim sliding this share of the way with it (parallax)
INTERIOR_CRANE = False        # rise / descend moves inside (off: client review 2026-11-20, item 17)
RISE_Z = 2.1                  # m: the top of a rise
RISE_PITCH = 20.0             # deg down at the top of a rise
RISE_AREA = 16.0              # m2: a room this big (or open plan) may rise
RISE_HEADROOM = 0.35          # m under the ceiling a rise stops
RISE_DRIFT = (0.3, 0.7)       # m it drifts toward what it shows
RISE_DIST = (2.6, 6.0)        # m to what it shows
EXIT_STOP = 1.0               # m before the doorway an exit ends (door frames stay off the lens)
EXIT_LEN = (1.2, 3.0)         # m an exit travels along the route
EXIT_AHEAD = 2.5              # m ahead along the route it looks (where it is going)
AFTER_EXIT = (0.5, 1.2)       # m into the next room its walk-in starts after an exit
# room changes (client review 2026-11-20 item 6; user 2026-11-26 option c): the camera flies THROUGH an
# open-plan seam or a doorway at least THROUGH_MIN_W wide (and clean: the near-lens checks pass) in one
# move; at a narrower door it cuts, matched - the shot before ends heading for the door, the next starts
# just past it on the same heading. Between two narrow doors (a bedroom -> hall -> bedroom) a walk
# along the hall carries the move, so every cut is at a door and keeps its heading.
THROUGH_MIN_W = 1.2           # m: a doorway this wide is flown through
THROUGH_BEFORE = (0.8, 1.8)   # m before the opening a through flight starts
THROUGH_AFTER = (0.9, 2.2)    # m past it it ends, turning onto what the room is about
THROUGH_TURN_FROM = 0.45      # of the flight: where its look starts turning from ahead onto the room
FLIGHT_S = (3.0, 5.5)         # s a flight leg lasts (item 9: never under 3 s; short, so the 90 s cap keeps them)
FLIGHT_PACE = (0.25, 0.45)    # m/s: a leg flies at the previous shot's pace, within these
# a flight through a door into the next room flies at the tour's band pace (speed_band: forward
# 0.2 m/s is ~3.3 % frame width/s at the 3 m reference depth) and turns onto the room gently - planned
# faster, the speed band kept only a quarter of it and the walk through the door was lost (user
# 2026-11-27, green house study); the room's next shot shows the room
THROUGH_PACE = 0.22           # m/s: the 3D band's walk (0.18 left joined room changes 2-3 s long each, 2026-11-27)
THROUGH_S_MAX = 14.0          # s a through flight may last at THROUGH_PACE (a longer way there: the band trims its start)
THROUGH_TURN_DPS = 3.0        # deg/s its turn onto the room may reach
# the flight through a door is the room's opening shot - one move: under the doorway, on into the
# room and onto what it holds, never cut to another angle of the room just past the door (user
# 2026-11-27); it walks this far past the door (m) and takes the room's opening time
# one take (RoomMoves.bridge): the moves of the rooms a walking route joins, carried on into each other
BRIDGE_MPS = 0.22             # m/s it walks (the 3D pace: 3.6 %fw/s at the 3 m reference depth)
BRIDGE_DPS = 6.0              # deg/s it turns at most (a walk turning onto the next move)
BRIDGE_CLEAR = 0.15           # m: its walking route keeps this off the walls (moves end near them)
JOIN_BACK_M = 0.1             # m a join may fall back along the walk-in's way (else it walks back)
JOIN_BACK_SHARE = 0.3         # a first scene setting off this much backward does not carry a walk-in on
JOIN_EST_MAX_S = 6.0          # s: a first scene this short a bridge from the walk-in is tried first
BRIDGE_MIN_S = 3.0            # s at least (no piece of the tour under 3 s, client item 9)
BRIDGE_MAX_M = 6.0            # m: a longer walk between two moves is a cut (a transition) instead
BRIDGE_STILL_M = 0.05         # m: closer than this it only turns
BRIDGE_AHEAD_FROM_M = 1.0     # m: a longer bridge looks ahead along its way in between
BRIDGE_TURN_SHARE = 0.3       # of a walking bridge's length its look turns from a's / onto b's
NEAR_WALL_M = 1.0             # m: a wall this near the lens fills its part of the frame (wall_share)
NEAR_WALL_W = 2.0             # score a through flight loses with its first and last views all wall
THROUGH_AFTER_LEAD = (0.9, 2.2)     # (1.2-3.5 m at the 3D pace: 9-14 s walk-ins took three rooms' coverage, 2026-11-27)
WALK_LEN = (0.8, 4.5)         # m a walk between two narrow doors travels (longer: no matched chain)
THROUGH_NEAR = 0.45           # m: a flight through a door may pass its frame no closer than this
DOOR_PASS_CLEAR = 0.14        # m clear a flight needs in a doorway it passes through
FLIGHT_TRIES = 8              # starts and ends a flight leg tries, each, spread along the route
MATCH_TURN = 30.0             # deg: a walk-in after a door keeps within this of the heading through it
AFTER_DOOR_FRONT = 1.0        # m in front a matched walk-in needs at its start (it is mid-flight, just past a door)
UNFRAMED_W = 1.0              # score a matched walk-in loses ending without its subject framed (kept: the cut matters more)
PRIOR_W = 0.6                 # score of a kind the room's profile asks for
ROOM_REPEAT_W = 0.5           # score lost repeating a kind the room already had
# the photo tours' sequence rules (planner/cinematic.py), on this planner's score scale (x0.4)
# (user 2026-11-17: "a lot of camera rules from the 2d implementation ... would be true to this")
SAME_KIND_W = 1.0             # the previous shot's kind again            (2D: -2.5)
TWO_BACK_W = 0.4              # the kind of two shots back                (2D: -1.0)
SAME_SIDE_W = 0.8             # sideways the same way as the previous shot (2D: -2.0, sides alternate)
# continuous cuts (client review 2026-11-20, item 5; RoomMoves.cut_cost): replaces SAME_SIDE_W
CUT_TURN_W = 2.5              # entering at a lateral turn over 45 deg from the previous exit
CUT_AXIAL_W = 2.5             # entering backward after a forward exit (or the reverse)
CUT_SAME_W = 0.5              # carrying the previous shot's move on
USED_DIR_W = 0.4              # per earlier use of the kind, same way      (2D: -1.0)
OVERUSE_W = 0.6               # per use of a kind past its second          (2D: -1.5)
DOLLY_KINDS = ("push", "pull", "rise", "descend", "enter")   # the moves led by moving in / out (2D: zoom-led)
DOLLY_CAP_W = 2.0             # past a third of the tour's shots          (2D: -5.0)
DOLLY_RUN_W = 0.8             # two in a row                              (2D: -2.0)
HERO_TRAVEL_W = 0.2           # a hero shot that travels                  (2D: +0.5)
TRAVELS = ("backout", "walkway", "truck", "arc", "rise", "descend", "corner_sweep", "exit", "through", "walk", "enter", "push", "pull", "glide")
MIN_KIND_S = {"backout": 3.0, "door_pan": 3.0, "wall_pan": 3.0, "room_pan": 3.8, "pan": 3.8, "panorama": 4.5, "truck": 3.5, "arc": 3.5, "rise": 2.8, "descend": 2.8,
              "corner_sweep": 3.5}
# the move library mixes (user 2026-11-17: "we need like 10 15 camera moves we mix and match. not all
# camera moves need to be in every tour"): no kind more than KIND_CAP times a tour (rise / descend
# once or twice), nor more than KIND_SHARE of its shots - the flight between rooms is exempt
KIND_CAP = {"backout": 3, "door_pan": 2, "wall_pan": 3, "rise": 2, "descend": 1, "panorama": 2, "corner_sweep": 2, "truck": 2, "arc": 2, "walkway": 3,
            "room_pan": 1, "push": 2, "pull": 2, "glide": 2, "hold_pan": 1}
KIND_SHARE = 0.25
CAP_W = 3.0                   # past its cap
FLIGHT = ("exit", "through", "walk", "enter", "link", "reveal")
# moves that read the same played backwards (a backout reversed would end facing a wall)
REVERSIBLE = ("walkway", "glide", "truck", "arc", "room_pan", "wall_pan", "pan", "corner_sweep", "panorama", "door_pan")      # the moves between rooms (not capped)
# small rooms (user 2026-11-17: "the bathrooms never got included, even though some shot can be
# managed, like from the deepest point in the bath escaping backwards towards the hall"): the camera
# starts deep in the room looking into it and backs out toward its door, the room opening up
BACKOUT_SMALL_M2 = 16.0       # rooms up to this footprint (and every bathroom / bedroom) get it
BACKOUT_CLEAR = 0.2           # m clear it needs (a bathroom's strip of floor)
BACKOUT_IN = 0.6              # m it starts in from the deepest point (the far wall not in its face)
BACKOUT_FRONT = (0.6, 0.8)    # m in front it needs at the start / end
BACKOUT_NEAR = 0.35            # m: its near-lens radius (a bathroom's side walls are always close)
# a small room looked into from its door: the camera stands DOOR_PAN_IN inside it and pans across it
# (user 2026-11-27, green house: the bathroom's back-out, then the bedroom walked into, reversed at the
# cut - a pan leaves the room sideways, so the walk on follows)
DOOR_PAN_IN = (0.3, 0.6)      # m in from the door it stands
DOOR_PAN_SWEEP = (15.0, 40.0)  # deg it pans, within the turn budget
DOOR_PAN_ROOMS = ()           # the rooms looked into from their door: none (user 2026-11-27: the bathroom's
                              # door pan stared at its glass screen - "the walk back we had was better")
DOOR_PAN_CLUSTER_M = 1.5     # fittings this near the one nearest the door are what it ends on
DOOR_PAN_SIGHT_TOL = 0.3     # m short of them the floor plan's clear line may stop
DOOR_PAN_FRONT = 0.6          # m in front at both ends (it stands still: a back-out's 0.8 m end lost the pan one way)
# what a small room is about: its pan ends on these (the green house bathroom's first door pan swept the
# middle of its floor - a shower screen and a wall, user 2026-11-27)
DOOR_PAN_ON = ("sink", "toilet", "bathtub", "counter", "mirror", "shower", "bed", "desk", "wardrobe")
COVER_FEATURE_MIN = 0.5       # a further angle ends on something (FEATURES weight in view), not bare wall
WALL_PAN = 30.0               # deg a coverage pan turns across the unseen wall
SWEEP_LEN = (1.0, 1.4, 1.8)   # m a corner sweep travels toward the room's middle ...
SWEEP_TURN = (25.0, 35.0)     # ... turning this much across it
PANO_SWEEP = (45.0, 80.0)     # deg a panorama turns (the speed limit decides within)   # shorter: a simpler move
SINK_M = 0.3                  # a view may not drop this much past what it shows (the interest above it)
# what a room's first shot (the one its label plays on) must show: what makes it that room (user
# 2026-11-16: the Home Scan's kitchen opened on its dining table - "the camera shows the room its
# mentioning in the frame")
TYPE_SIGNS = {"kitchen": ("stove", "oven", "sink", "refrigerator", "dishwasher", "kitchen island", "countertop",
                          "counter", "hood", "microwave", "toaster"),
              "bedroom": ("bed",), "primary_bedroom": ("bed",), "guest_bedroom": ("bed",),
              "bathroom": ("toilet", "bathtub", "shower", "sink"),
              "living_room": ("sofa", "television receiver", "fireplace"), "family_room": ("sofa", "television receiver", "fireplace"),
              "dining_room": ("table",), "home_office": ("desk", "computer", "monitor", "keyboard", "mouse", "table")}
TYPE_WHOLE = ("bed", "sofa")    # ... framed whole, when the room has one
TOP_VOID_MAX = 0.15           # at most this share of the frame's top fifth may be uncaptured (a missing ceiling; 0.3 let the Home Scan kitchen band through)
TOP_TILT = (3.0, 6.0)         # deg a shot tilts down to clear it, before it is dropped
PITCH_MAX_ROOM = 12.0         # deg down at most, every tilt included (user 2026-11-17: Home Scan v11 "angles too low" - 16-30 deg)
PITCH_MAX_NO_CEILING = 15.0   # with no ceiling in the scan, a little more (the Home Scan lost its living rooms at 12)
BAND_TOLERANCE = 1.5          # deg of band a frame may keep (a sliver at the very top)
PITCH_MAX_RISE = 20.0         # a rise / descend at its steepest
BAND_MARGIN = 0.05            # m: the frame's top edge stays this far under where the scan's walls end
BAND_TILT_MAX = 6.0           # deg a shot may tilt down to keep it there (no ceiling), else it goes
WHOLE_GAP = 0.4               # m: labelled pieces of one bed / sofa this close are one piece
FRAME_W, FRAME_H = 1920, 1080
# what a view is worth showing (scene labels): a fireplace or a window beats a door
FEATURES = {"fireplace": 1.0, "windowpane": 0.8, "kitchen island": 0.8, "bathtub": 0.8, "bed": 0.8,
            "stairs": 0.7, "stairway": 0.7, "countertop": 0.6, "stove": 0.6, "bookcase": 0.6, "chandelier": 0.6,
            "sofa": 0.6, "television receiver": 0.5, "painting": 0.5, "table": 0.5, "desk": 0.5, "mirror": 0.4,
            "curtain": 0.3, "door": 0.2, "sink": 0.5, "hood": 0.5, "oven": 0.4, "refrigerator": 0.3,
            "cabinet": 0.15}
# of a view's middle on the shot's own room: under this the shot loses score (OWN_W)
OWN_SHARE = {"backout": 0.5, "door_pan": 0.5, "wall_pan": 0.5, "corner_sweep": 0.5, "panorama": 0.5, "descend": 0.35, "truck": 0.5, "rise": 0.35, "walkway": 0.35, "arc": 0.35, "room_pan": 0.5, "push": 0.5, "pull": 0.5, "glide": 0.5}
CONFIDENT_M2 = 0.5            # m2 of a room's own furniture for its rules to be eased
RELAX = 0.8                   # the second try: room to breathe x this ...
OWN_MIN_RELAXED = 0.35        # ... and this much of its own room in view
OWN_PIECE_NEAR = 3.5          # m: a walk / arc framing the piece its type is about from this close needs no own share
OWN_MIN = 0.5                 # a room shot's view is at least half its own room
OWN_FREE = ("enter", "reveal", "link", "pan", "detail", "exit", "through", "walk")   # they look in from outside it, or back over the home
COVER_OWN_MIN = 0.6           # a room's further shot shows mostly that room (not the hallway beyond)
COMPOSED = ("walkway", "arc", "room_pan", "push", "pull", "truck", "rise", "corner_sweep", "panorama", "descend",
            "wall_pan", "backout", "door_pan")    # the kinds a room's further shots may be
OWN_W = 4.0                                                # score lost per missing share
FEATURE_W = 0.4
# close-ups (detail shots, 1.5-2.5 s): fixtures, finishes, decor
DETAILS = {"chandelier": 1.0, "fireplace": 1.0, "vase": 0.9, "flower": 0.9, "sculpture": 0.9, "painting": 0.8,
           "stove": 0.8, "sink": 0.7, "lamp": 0.7, "sconce": 0.7, "clock": 0.6, "plant": 0.6, "mirror": 0.6,
           "hood": 0.6, "bathtub": 0.6, "television receiver": 0.3}
DETAIL_SIZE = (0.12, 1.6)     # m across: a vase to a fireplace
DETAIL_DIST = (0.9, 1.15, 1.4)
DETAIL_PUSH = 0.25
DETAIL_PITCH_MAX = 30.0
DETAIL_NOT_PREV = 0.9       # m: a close-up's object this near the previous shot's last aim is what it already showed


def move_motion(m, end: bool):
    """A move's on-screen motion leaving (end) or entering its shot, at a steady pace, in the units
    of render/continuity.py (only its direction is compared): the eye's travel seen at
    REF_DEPTH_M and the view's turn over the field of view, over the first / last few samples."""
    from tour_worker.render.continuity import REF_DEPTH_M, Motion
    eye, tgt = m.eye3, np.asarray(m.tgt, float)
    k = min(4, len(eye) - 1)
    i0, i1 = (len(eye) - 1 - k, len(eye) - 1) if end else (0, k)
    up = np.array([0.0, 0.0, 1.0])
    d0 = (tgt[i0] - eye[i0]) / max(np.linalg.norm(tgt[i0] - eye[i0]), 1e-9)
    d1 = (tgt[i1] - eye[i1]) / max(np.linalg.norm(tgt[i1] - eye[i1]), 1e-9)
    fwd = (d0 + d1) / max(np.linalg.norm(d0 + d1), 1e-9)
    right = np.cross(fwd, up)
    right /= max(np.linalg.norm(right), 1e-9)
    cam_up = np.cross(right, fwd)
    v, w = eye[i1] - eye[i0], d1 - d0
    hf = math.radians(HFOV)
    half = REF_DEPTH_M * math.tan(hf / 2)
    return Motion(float(v @ right) / (2 * half) + float(w @ right) / hf,
                  float(v @ cam_up) / (2 * half) + float(w @ cam_up) / hf,
                  float(v @ fwd) / REF_DEPTH_M * 0.5)


def trim_move_start(m, keep: float) -> None:
    """In place: m with only the last `keep` share of its path (by the renderer's arc measure: the
    eye's travel + TARGET_WEIGHT x the aim's), resampled to its sample count."""
    from tour_worker.scene.camera import TARGET_WEIGHT
    eye3, tgt = m.eye3, np.asarray(m.tgt, float)
    step = np.linalg.norm(np.diff(eye3, axis=0), axis=1) + TARGET_WEIGHT * np.linalg.norm(np.diff(tgt, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(step)]
    if arc[-1] < 1e-9:
        return
    u = np.linspace((1.0 - keep) * arc[-1], arc[-1], len(m.eye))
    m.eye = np.c_[[np.interp(u, arc, m.eye[:, j]) for j in range(2)]].T
    m.tgt = np.c_[[np.interp(u, arc, tgt[:, j]) for j in range(3)]].T
    if m.z is not None:
        m.z = np.interp(u, arc, m.z)
    m.why["travel_m"] = round(float(np.linalg.norm(np.diff(m.eye3, axis=0), axis=1).sum()), 2)


def _route_point(poly: np.ndarray, s_: np.ndarray, s: float) -> np.ndarray:
    """The point s m along a route; past its end, straight on along its last stretch (a look ahead
    does not stop on the eye where the route ends)."""
    if s <= s_[-1]:
        return np.array([np.interp(s, s_, poly[:, j]) for j in (0, 1)])
    k = int(np.searchsorted(s_, s_[-1] - 0.5))
    d = poly[-1] - poly[min(k, len(poly) - 2)]
    n = float(np.linalg.norm(d))
    return poly[-1] + (d / n if n > 1e-9 else 0.0) * (s - s_[-1])


def ends_backward(m) -> bool:
    """The move leaves its shot moving backward (render/continuity.py's forward-back test against a
    forward move): the shot before a forward flight out of the room would reverse into it."""
    from tour_worker.render.continuity import Motion, _axial_reversal
    return _axial_reversal(_unit(move_motion(m, end=True)), Motion(0.0, 0.0, 10.0))


def breaks_into(m, nxt) -> bool:
    """m's exit breaks the cut into what follows it: nxt the next move's start motion (render/
    continuity units - a dry run of the flight on, user 2026-11-27: the primary bedroom's last pan
    swept left into a walk heading forward-right, a 157 deg turn), or True - a move forward of
    unknown heading (a walk in from the next room's door): m ends moving backward."""
    if m.kind == "backout":
        return False          # (a small room is backed out of - user 2026-11-27; the next room is then
                              # entered by a cut, not walked into: plan._fly_on)
    if nxt is True:
        return ends_backward(m)
    from tour_worker.render.continuity import MAX_TURN_DEG, _axial_reversal, _turn_deg
    a, b = _unit(move_motion(m, end=True)), _unit(nxt)
    turn = _turn_deg(a, b)
    return _axial_reversal(a, b) or (turn is not None and turn > MAX_TURN_DEG)


def _unit(mo):
    """The direction only, at a speed of 10 (the still threshold never applies)."""
    from tour_worker.render.continuity import Motion
    sp = mo.speed
    return mo if sp < 1e-12 else Motion(10 * mo.x / sp, 10 * mo.y / sp, 10 * mo.f / sp)


@dataclass
class Move:
    kind: str
    eye: np.ndarray             # (n, 2) planner xy, at EYE
    tgt: np.ndarray             # (n, 3) planner points
    score: float
    pitch: float = PITCH_LADDER[0]
    piece: object = None
    occ: float | None = None
    why: dict = field(default_factory=dict)
    eye_z: float = EYE           # detail moves sit lower / higher than the room eye height
    subject: np.ndarray | None = None   # (8, 3) box of what the shot ends on (planner coords), floor to top
    z: np.ndarray | None = None  # (n,) eye height along the move (a rise); None: eye_z throughout
    easing: str = "ease_in_out"  # a fly-out speeds up into the cut (ease_in), the fly-in settles (ease_out)

    @property
    def zs(self) -> np.ndarray:
        return self.z if self.z is not None else np.full(len(self.eye), self.eye_z)

    @property
    def eye3(self) -> np.ndarray:
        return np.c_[self.eye, self.zs]

    @property
    def first(self):
        return self.eye[0], self.tgt[0]

    @property
    def last(self):
        return self.eye[-1], self.tgt[-1]


# ---- geometry helpers ----------------------------------------------------------------------------
def resample(poly: np.ndarray, n: int) -> np.ndarray:
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))]
    if d[-1] < 1e-9:
        return np.repeat(poly[:1], n, axis=0)
    s = np.linspace(0, d[-1], n)
    return np.c_[[np.interp(s, d, poly[:, k]) for k in range(poly.shape[1])]].T


def smooth(space, poly: np.ndarray, rmin: float, sigma_m: float = 0.35) -> np.ndarray:
    """Gaussian-smoothed path (ends fixed), as smooth as the clearance allows."""
    step = 0.05
    d = float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum())
    p = resample(poly, max(2, int(d / step) + 1))
    for sig in (sigma_m, sigma_m * 0.6, sigma_m * 0.3):
        q = ndi.gaussian_filter1d(p, sig / step, axis=0, mode="nearest")
        w = np.clip(np.r_[np.linspace(0, 1, min(len(q), 12)), np.ones(max(0, len(q) - 12))], 0, 1)
        w = np.minimum(w, w[::-1])[:, None]
        q = p * (1 - w) + q * w
        if min(space.clearance(x) for x in q) >= rmin * 0.95:
            return q
    return p


def fit_cubic(pts: np.ndarray):
    """Least-squares cubic through pts with the end points fixed (chord-length parameters)."""
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    u = d / d[-1] if d[-1] > 1e-9 else np.linspace(0, 1, len(pts))
    P0, P3 = pts[0], pts[-1]
    b0, b1, b2, b3 = (1 - u) ** 3, 3 * u * (1 - u) ** 2, 3 * u ** 2 * (1 - u), u ** 3
    sol, *_ = np.linalg.lstsq(np.c_[b1, b2], pts - np.outer(b0, P0) - np.outer(b3, P3), rcond=None)
    ctrl = np.array([P0, sol[0], sol[1], P3])
    fit = np.outer(b0, ctrl[0]) + np.outer(b1, ctrl[1]) + np.outer(b2, ctrl[2]) + np.outer(b3, ctrl[3])
    return ctrl, float(np.abs(fit - pts).max())


def fit_pair(eye: np.ndarray, tgt: np.ndarray, tol: float = 0.03):
    """Eye and target polylines -> 1-3 joined cubic segments each: [(eye ctrl, target ctrl)], error."""
    n = len(eye)
    for k in (1, 2, 3):
        cuts = np.linspace(0, n - 1, k + 1).astype(int)
        segs, worst = [], 0.0
        for a, b in zip(cuts[:-1], cuts[1:]):
            ce, ee = fit_cubic(eye[a:b + 1])
            ct, et = fit_cubic(tgt[a:b + 1])
            worst = max(worst, ee, et / 4)            # the target can be looser
            segs.append((ce, ct))
        if worst <= tol:
            break
    return segs, worst


def eye3(eye2: np.ndarray) -> np.ndarray:
    return np.c_[eye2, np.full(len(eye2), EYE)]


def speeds_of(segments: list[dict], duration_s: float, easing: str = "ease_in_out", samples: int | None = None,
              fps: int = 30, path_s: float | None = None, speed_profile=None) -> tuple[float, float]:
    """(peak turn deg/s, peak travel m/s) of bezier segments as the worker draws them. path_s
    (continuous cuts): the move runs over path_s at its planned pace; duration_s is where it is cut."""
    path = BezierCameraPath(segments, [HFOV, HFOV], easing=easing, speed_profile=speed_profile)
    n = samples or max(2, int(round(duration_s * fps)))
    if path_s:
        from tour_worker.render.camera_path import ramp_profile
        poses = [path.pose_at_fraction(path.along(ramp_profile(easing, min(1.0, f / (n - 1) * duration_s / path_s))))
                 for f in range(n)]
    else:
        poses = [path.pose(f / (n - 1)) for f in range(n)]
    eye = np.array([p[0] for p in poses])
    tgt = np.array([p[1] for p in poses])
    d = tgt - eye
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    dt = duration_s / (n - 1)
    turn = np.degrees(np.arccos(np.clip((d[1:] * d[:-1]).sum(1), -1, 1))) / dt
    move = np.linalg.norm(np.diff(eye, axis=0), axis=1) / dt
    return float(turn.max()), float(move.max())


def trial_speeds(eye2: np.ndarray, tgt: np.ndarray, dur: float, z=EYE, easing: str = "ease_in_out") -> tuple[float, float]:
    segs, _ = fit_pair(np.c_[eye2, np.broadcast_to(np.asarray(z, float), (len(eye2),))], tgt)
    return speeds_of([{"eye": e, "target": t} for e, t in segs], dur, easing=easing, samples=91)


def cap_pitch(eye2: np.ndarray, tgt: np.ndarray, pitch_max: float) -> np.ndarray:
    """Targets raised where the camera would look more than pitch_max below the horizon."""
    t = np.array(tgt, float)
    h = np.linalg.norm(t[:, :2] - np.asarray(eye2)[:, :2], axis=1)
    t[:, 2] = np.maximum(t[:, 2], EYE - h * math.tan(math.radians(pitch_max)))
    return t


def _dir(e, t) -> np.ndarray:
    d = np.array([t[0] - e[0], t[1] - e[1], t[2] - EYE])
    return d / np.linalg.norm(d)


def jump_cut(a, b) -> bool:
    """Two poses (eye xy, target) too alike for a cut: < JUMP_ANGLE apart and < JUMP_DIST apart."""
    if a is None or b is None:
        return False
    (e0, t0), (e1, t1) = a, b
    ang = math.degrees(math.acos(max(-1.0, min(1.0, float(_dir(e0, t0) @ _dir(e1, t1))))))
    return ang < JUMP_ANGLE and float(np.linalg.norm(np.asarray(e0[:2]) - np.asarray(e1[:2]))) < JUMP_DIST


def repeats(first, last, earlier) -> bool:
    """The move starts or ends on a framing an earlier shot of the room already had."""
    return any(jump_cut(p, q) for p in (first, last) for f, l_ in earlier for q in (f, l_))


def in_frame(p2, target, piece, core: float) -> bool:
    """The piece's top surface (long side cut to core) inside the frame from p2 aimed at target."""
    V, K = look_at([p2[0], p2[1], EYE], target), intrinsics(HFOV, FRAME_W, FRAME_H)
    cam = piece.top_corners(core) @ V[:3, :3].T + V[:3, 3]
    if (cam[:, 2] < 0.3).any():
        return False
    px = K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2]
    py = K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]
    return px.min() > 0.03 * FRAME_W and px.max() < 0.97 * FRAME_W and py.min() > 0.03 * FRAME_H and py.max() < 0.97 * FRAME_H


def piece_box(fp) -> np.ndarray:
    """A piece's box: its footprint at the floor and at its top (8, 3)."""
    c = []
    for z in (0.0, float(fp.top)):
        for sx in (-1, 1):
            for sy in (-1, 1):
                c.append([*(fp.centre + sx * fp.half_long * fp.u + sy * fp.half_short * fp.v), z])
    return np.array(c, float)


def objects_box(objs: list[dict]) -> np.ndarray | None:
    """The box around labelled objects, down to the floor (8, 3)."""
    if not objs:
        return None
    lo = np.min([o["min"] for o in objs], 0)
    hi = np.max([o["max"] for o in objs], 0)
    return np.array([[x, y, z] for z in (0.0, float(hi[2])) for x in (lo[0], hi[0]) for y in (lo[1], hi[1])], float)


def frames_fully(eye2, target, box: np.ndarray, eye_z: float = EYE) -> bool:
    """The whole subject in frame, with room around it: its box inside the frame by FRAME_MARGIN on
    every side, and the floor under it above the frame's bottom by FLOOR_MARGIN (user 2026-11-16:
    "when we zoom onto an object to end the scene we can see it fully - a sofa relative to the floor,
    a painting or fireplace with enough wall and a bit of floor, so we know what's where")."""
    V, K = look_at([eye2[0], eye2[1], eye_z], target), intrinsics(HFOV, FRAME_W, FRAME_H)
    cam = box @ V[:3, :3].T + V[:3, 3]
    if (cam[:, 2] < 0.3).any():
        return False
    px = K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2]
    py = K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]
    mx, my = FRAME_MARGIN * FRAME_W, FRAME_MARGIN * FRAME_H
    if px.min() < mx or px.max() > FRAME_W - mx or py.min() < my:
        return False
    floor = py[box[:, 2] <= 1e-6]                         # the corners on the floor
    return bool(floor.max() <= (1.0 - FLOOR_MARGIN) * FRAME_H)


def frames_on_far_floor(eye2, target, box: np.ndarray, eye_z: float = EYE) -> bool:
    """frames_fully, but the floor strip only under the subject's far side: a table filmed from a
    dining area's own floor (green house: at most 2.75 m from it) cannot show the floor under its
    near edge at the tilt cap - the whole table with the floor beyond it reads where it stands."""
    V, K = look_at([eye2[0], eye2[1], eye_z], target), intrinsics(HFOV, FRAME_W, FRAME_H)
    cam = box @ V[:3, :3].T + V[:3, 3]
    if (cam[:, 2] < 0.3).any():
        return False
    px = K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2]
    py = K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]
    mx, my = FRAME_MARGIN * FRAME_W, FRAME_MARGIN * FRAME_H
    if px.min() < mx or px.max() > FRAME_W - mx or py.min() < my:
        return False
    top = box[:, 2] > 1e-6
    if py[top].max() > (1.0 - FLOOR_MARGIN) * FRAME_H:      # the top itself whole, near edge included
        return False
    floor = np.flatnonzero(~top)
    far = floor[np.argsort(-cam[floor, 2])[:2]]              # the two floor corners farthest away
    return bool(py[far].max() <= (1.0 - FLOOR_MARGIN) * FRAME_H)


def _in_view(eye2, target, P: np.ndarray, eye_z: float = EYE) -> np.ndarray:
    """Which of the points P (n, 3) are in the frame from eye2 (at eye_z) aimed at target."""
    V, K = look_at([eye2[0], eye2[1], eye_z], target), intrinsics(HFOV, FRAME_W, FRAME_H)
    cam = P @ V[:3, :3].T + V[:3, 3]
    z = np.maximum(cam[:, 2], 1e-6)
    px = K[0, 0] * cam[:, 0] / z + K[0, 2]
    py = K[1, 1] * cam[:, 1] / z + K[1, 2]
    return (cam[:, 2] > 0.3) & (px > 0.02 * FRAME_W) & (px < 0.98 * FRAME_W) & (py > 0.02 * FRAME_H) & (py < 0.98 * FRAME_H)


def bed_seen(eye2, target, piece, head: tuple, eye_z: float = EYE) -> tuple[float, bool]:
    """(share of the bed's top in frame, its headboard side in frame): a grid over the top surface;
    head: (axis, sign) of the side against the wall - axis 0 along the piece's long side, 1 across."""
    a = np.linspace(-0.9, 0.9, 7)
    b = np.linspace(-0.9, 0.9, 5)
    g = np.array([[*(piece.centre + x * piece.half_long * piece.u + y * piece.half_short * piece.v), piece.top]
                  for x in a for y in b])
    share = float(_in_view(eye2, target, g, eye_z).mean())
    ax, sg = head
    if ax == 0:
        ends = [piece.centre + sg * 0.85 * piece.half_long * piece.u + y * 0.8 * piece.half_short * piece.v for y in (-1, 1)]
    else:
        ends = [piece.centre + sg * 0.85 * piece.half_short * piece.v + x * 0.8 * piece.half_long * piece.u for x in (-1, 1)]
    ends = np.array([[*e, piece.top] for e in ends])
    return share, bool(_in_view(eye2, target, ends, eye_z).all())


def _keep_best(moves: list, n: int = None) -> list:
    """The KEEP_PER_KIND best of a kind's candidates by their own score, before the rendered and
    framing checks (a room's pans or arcs run to hundreds - 659 in the Home Scan's living room,
    planning 6x slower, 2026-11-16)."""
    n = KEEP_PER_KIND if n is None else n
    return sorted(moves, key=lambda m: -m.score)[:n]


def parallax(eye2: np.ndarray, tgt: np.ndarray) -> float:
    """1: the camera moves straight across its view; 0: along it."""
    mv = eye2[-1] - eye2[0]
    if np.linalg.norm(mv) < 1e-6:
        return 0.0
    mv /= np.linalg.norm(mv)
    k = len(eye2) // 2
    view = tgt[k][:2] - eye2[k]
    view /= max(1e-9, np.linalg.norm(view))
    return abs(mv[0] * view[1] - mv[1] * view[0])


# ---- the planner for one room --------------------------------------------------------------------
class RoomMoves:
    def __init__(self, space, occlusion=None):
        self.space, self.occ = space, occlusion

    def best(self, room: str, dur: float, **kw) -> tuple[Move | None, list[str]]:
        """_best; a room with no view under the full rules gets a second try with the softest ones
        eased (room to breathe x RELAX, own-room share OWN_MIN_RELAXED, framing not required) - a
        room skipped is worse than a room shown a little closer (v10: the green house's dining
        room, study and hall had no shot). A further angle (covered=) is held to the planning budget."""
        self._late_ok = kw.get("covered") is not None
        try:
            mv, w = self._best(room, dur, **kw)
        finally:
            self._late_ok = False
        if mv is None and kw.get("covered") is None and kw.get("kind", "room") != "detail" and self.confident(room):
            self.relaxed = True
            try:
                mv, w2 = self._best(room, dur, **kw)
            finally:
                self.relaxed = False
            if mv is not None:
                mv.why["relaxed"] = True
                return mv, [f"room_rules_relaxed:{room}"] + [x for x in w2 if not x.startswith("room_no")]
        return mv, w

    def confident(self, room: str) -> bool:
        """The scan shows what the room is: a main piece, or CONFIDENT_M2 of what makes it that room
        (TYPE_SIGNS) - only then may its rules be eased for a view (green house v11: a laundry nook
        typed "family room" off 0.05 m2 of "sofa" got a relaxed glide at the washing machines)."""
        if self.space.main_pieces(room):
            return True
        a = 0.0
        for o in self.type_signs(room):
            a += float(np.prod(np.asarray(o["max"][:2]) - np.asarray(o["min"][:2])))
        return a >= CONFIDENT_M2

    def _best(self, room: str, dur: float, *, earlier=(), prev=None, piece_rank: int = 0,
             opening: bool = False, closing: bool = False, kind: str = "room",
             next_room: str | None = None, enter_route=None, after_exit: bool = False,
             final: bool = False, follows_link: bool = False, covered: np.ndarray | None = None,
             then_forward=False, last_shot: bool = True, join_from=None) -> tuple[Move | None, list[str]]:
        """The best move for a shot of `room` lasting dur s. earlier: (first, last) poses of the
        room's earlier shots; prev: the previous shot's last pose; next_room: the tour's next room
        (an opening reveal turns onto it). Returns (move, warnings).

        Which move, in this order (the first whose candidates pass the speed, own-room and lens
        checks wins; each kind at the roomier camera clearance first):
          close-up          a detail shot
          look-back pan     the closing shot, when the last room sees back into the home (straight
                            after a link, the enter comes first)
          enter             a room's first shot when the route to it runs through a hallway: in
                            along the route from the hallway
          reveal            the opening shot of a room with no main piece, toward the next room
          walkway           the room's main piece (a second angle: its next piece, or the same one
                            from elsewhere - never a framing an earlier shot had)
          push              no piece: toward the room's own working surfaces (ANCHORS)
          glide / pan       nothing else exists"""
        sp = self.space
        self._room_now = room
        self._cov_now = covered is not None
        type_warn: list[str] = []
        share = PRE_SYNC_SHARE.get(kind, 0.8)
        self.turn_budget, self.move_budget = MAX_TURN_DPS * share, MAX_MOVE_MPS * share
        pieces = sp.main_pieces(room)
        piece = pieces[piece_rank % len(pieces)] if pieces else None
        anchor = self.anchor(room)
        # straight after a link the tour walks in, even into its last room; otherwise the closing
        # look-back comes first
        order = (("detail", "enter", "closing", "reveal", "walkway", "push") if follows_link or after_exit
                 else ("detail", "closing", "enter", "reveal", "walkway", "push"))
        found: dict[str, list[list[Move]]] = {k: [] for k in order}
        for rmin in RMIN_LADDER:                       # each kind at the roomier clearance first
            mask = sp.space_mask(room, rmin)
            pts = self._points(room, mask)
            if len(pts) < 2:
                continue
            paths = sp.paths_between(pts, mask)
            # a shot starts in its own room (it may then move on through an open-plan space)
            own = {k for k, p in enumerate(pts) if sp.room_at(p) == room}
            if not own:
                continue
            if kind == "detail":                      # a close-up of one of the room's details
                found["detail"].append(self._detail(room, rmin, dur, prev, earlier))
                continue
            if enter_route is not None:
                found["enter"].append(self._enter(room, piece, enter_route, rmin, dur, prev, earlier, after_exit))
            if opening and piece is None and next_room is not None:
                found["reveal"].append(self._reveal(room, next_room, pts, paths, rmin, dur, prev, own))
            if closing:
                found["closing"].append([m for m in self._pan(room, [pts[k] for k in sorted(own)], rmin, dur, prev,
                                                              earlier, True) if m.why["view_depth_m"] >= CLOSING_DEPTH])
            # the composed moves compete on score: walkways on the piece, arcs round it, pans across
            # the room's own things (user 2026-11-16: small rooms "almost no pan")
            composed: list[Move] = []
            if piece is not None:
                for pitch in PITCH_LADDER:
                    walk = self._walkway(room, piece, pts, paths, rmin, dur, pitch, prev, earlier, own)
                    if walk:
                        composed += walk
                        break
                composed += self._arc(room, piece, rmin, dur, prev, earlier)
            composed += self._room_pan(room, [pts[k] for k in sorted(own)], rmin, dur, prev, earlier)
            composed += self._truck(room, rmin, dur, prev, earlier)
            if INTERIOR_CRANE:       # (client review item 17: interiors are an eye-level gimbal walk -
                                     # no rise or crane moves; drone-style moves belong outside)
                composed += self._rise(room, [pts[k] for k in sorted(own)], rmin, dur, prev, earlier)
                composed += self._descend(room, [pts[k] for k in sorted(own)], rmin, dur, prev, earlier)
            composed += self._corner_sweep(room, [pts[k] for k in sorted(own)], rmin, dur, prev, earlier)
            composed += self._panorama(room, [pts[k] for k in sorted(own)], rmin, dur, prev, earlier)
            composed += self._backout(room, [pts[k] for k in sorted(own)], rmin, dur, prev, earlier)
            composed += self._door_pan(room, dur, prev, earlier)
            if covered is not None:              # a further angle: onto the walls not seen yet
                composed += self._wall_views(room, [pts[k] for k in sorted(own)], rmin, dur, prev, earlier, covered)
            if composed:
                found["walkway"].append(composed)
            if anchor is not None:
                found["push"].append(self._push(room, anchor, pts, paths, rmin, dur, prev, earlier, own))
        tiers: list[list[Move]] = [t for k in order for t in found[k]]
        stats = self.stage_stats(room, covered is not None)
        _n = lambda ts: len({id(m) for t in ts for m in t})          # noqa: E731
        # nothing to compose on, or every composed move blocked: glides into the deepest views (a pan
        # is never a room shot - it sweeps past what a room holds; only the closing look-back pans)
        glides: list[Move] = []
        mask = sp.space_mask(room, RMIN_LADDER[-1])
        pts = self._points(room, mask)
        if len(pts) >= 2 and kind != "detail":
            own = {k for k, p in enumerate(pts) if sp.room_at(p) == room}
            if own:
                paths = sp.paths_between(pts, mask)
                glides = self._glide(room, pts, paths, RMIN_LADDER[-1], dur, prev, earlier, opening, own, closing)
        # room to breathe first: composed moves, then (only when none has it) the closest composed
        # calls, then glides the same way. A room's second angle is a composed move with room to
        # breathe, or none: without one the room's first shot keeps the time.
        # the aims as they will play, before anything is judged: never up, and lower when the scan
        # has no ceiling (its black band stays out of the frame)
        drop = math.tan(math.radians(CEIL_PITCH)) if "scene_ceiling_missing" in getattr(sp, "warnings", []) else 0.0
        for m in {id(m): m for t in tiers + [glides] for m in t}.values():
            if m.kind != "detail":
                if drop:
                    m.tgt[:, 2] -= drop * np.linalg.norm(m.tgt[:, :2] - m.eye, axis=1)
                m.tgt[:, 2] = np.minimum(m.tgt[:, 2], AIM_Z_MAX)
                self.to_pace(m, dur)
        # a move that would turn back across the cut (or end moving backward before a flight out of
        # the room) is offered played in reverse too, when its kind reads the same both ways - judged
        # by every rule like the rest (client review item 5; the green house's in-room cuts)
        if self.continuous:
            tiers = [t + self._reversed(t, then_forward) for t in tiers]
            glides = glides + self._reversed(glides, then_forward)
        strict = piece_rank > 0 and kind != "detail"
        stats["generated"] += _n(tiers + [glides])
        roomy = lambda t: [m for m in t if self._roomy(m) and self._good(m, dur)]
        if covered is not None and kind != "detail":
            # a further shot of the room: it must show walls the earlier ones did not (glides
            # included - they face away from the piece the first shot framed), with room to breathe
            # (user 2026-11-16: glides and doorway views as further shots made "close up walls, in
            # hallways without getting into the room") - only composed moves, of this room
            def fresh(t):
                out = []
                for m in t:
                    if m.kind not in COMPOSED:
                        continue
                    k = len(m.eye) // 2
                    own_ = sp.own_share(room, m.eye[k], math.atan2(*(m.tgt[k][:2] - m.eye[k])[::-1]), HFOV)
                    if own_ < COVER_OWN_MIN:
                        continue
                    new = self.new_walls(room, m, covered)
                    if not self.ends_on_something(room, m.eye[-1], m.tgt[-1]) or self.ends_on_door(m):
                        continue                   # new wall, but nothing on it (bare wall, a door frame, a door)
                    if new >= COVER_NEW_MIN:
                        m.why["walls_new"] = round(new, 2)
                        m.score += COVER_W * new
                        out.append(m)
                return out
            # how many further angles each rule leaves (user 2026-11-17: "have we abandoned our
            # coverage policies?" - the green house's rooms got one shot each, and nothing said why)
            n_all = sum(len(t) for t in tiers)
            kept = [roomy(t) for t in tiers]
            n_roomy = sum(len(t) for t in kept)
            tiers = [fresh(t) for t in kept]
            self.cover_stats = {"candidates": n_all, "room_to_breathe_and_good": n_roomy,
                                "composed_own_room_new_walls": sum(len(t) for t in tiers)}
        else:
            # every room shot keeps room to breathe and shows mostly its own room, or the room is not
            # filmed (user 2026-11-16: "close up walls" - the closest-call fallback is gone)
            def own_room(t):
                out = []
                for m in t:
                    if m.kind in OWN_FREE:
                        out.append(m)
                        continue
                    k = len(m.eye) // 2
                    o = sp.own_share(room, m.eye[k], math.atan2(*(m.tgt[k][:2] - m.eye[k])[::-1]), HFOV)
                    m.why["own_share"] = round(o, 2)
                    need = OWN_MIN_RELAXED if getattr(self, "relaxed", False) else OWN_MIN
                    if self.on_own_piece(room, m):
                        # it frames what makes the room that room, from close by: that is a shot of
                        # the room whatever lies beyond (own_share counts where the view's sight
                        # lines end - an open-plan dining zone has no walls of its own, so every view
                        # of its table "ends" in the living room or kitchen: green house v14-v16
                        # scored 0-0.22, and the dining room fell back to a held pan)
                        need = 0.0
                    if o >= need:
                        out.append(m)
                return out
            tiers = ([own_room(roomy(t)) for t in tiers] if strict
                     else [own_room(roomy(t)) for t in tiers] + [own_room(roomy(glides))])
        stats["passed_rules"] += _n(tiers)
        # the room's first shot shows what makes it that room (TYPE_SIGNS), when anything does
        typed: set | None = None
        if piece_rank == 0 and covered is None and kind != "detail" and self.type_signs(room):
            # in order of preference, never a hard filter: the bed / sofa whole (user 2026-11-16:
            # the primary bedroom opened on a mirror and a corner of the bed), then what shows the
            # room's kind, then the rest - a room is shown even where no view of its kind passes the
            # render checks (Home Scan v17: six whole-sofa views, all with a door frame at the edge,
            # and the living room was skipped while 1369 other views were never tried)
            # (a walk-in picking up a matched cut at the door counts: the cut is the point of it)
            shown = [[m for m in t if m.kind in ("link",) or m.why.get("after_exit") or self.shows_type(room, m)]
                     for t in tiers]
            typed = {id(m) for t in shown for m in t}
            if any(o["class"] in TYPE_WHOLE for o in self.type_signs(room)):
                # (a walk or arc framing the room's bed / sofa piece counts: a tight room's bed is
                # framed by bed_seen - most of it and its head - the whole of it fits no frame there)
                whole = [[m for m in t if m.kind in ("link",) or m.why.get("after_exit") or self.shows_type(room, m, whole=True)
                          or (m.kind in ("walkway", "arc") and m.piece is not None and m.piece.cls in TYPE_WHOLE
                              and self.framed(m))] for t in shown]
                w_ids = {id(m) for t in whole for m in t}
                # a walk or arc on the bed / sofa first, then any other whole view (green house: the
                # back-outs took the primary bedroom from its walk along the bed - user 2026-11-18
                # liked the walk, and the back-outs were "horrible")
                on_piece = [[m for m in t if m.kind in ("walkway", "arc") and m.piece is not None
                             and m.piece.cls in TYPE_WHOLE] for t in whole]
                p_ids = {id(m) for t in on_piece for m in t}
                shown = on_piece + [[m for m in t if id(m) not in p_ids] for t in whole] \
                    + [[m for m in t if id(m) not in w_ids] for t in shown]
            rest = [[m for m in t if id(m) not in typed] for t in tiers]
            tiers = shown + rest
        # a room's first shot does not look across another zone's floor at its subject (the Home Scan's
        # living room was filmed from beyond the dining table, through its chairs, user 2026-11-26):
        # such views go last, used only when nothing else shows the room
        if piece_rank == 0 and covered is None and kind != "detail" and len(sp.spaces.get(room, {room})) > 1:
            across = [m for t in tiers for m in t if m.kind not in FLIGHT and self.across_zone(room, m) > ACROSS_ZONE_MAX]
            if across:
                a_ids = {id(m) for m in across}
                tiers = [[m for m in t if id(m) not in a_ids] for t in tiers] + [across]
        if after_exit:                             # the matched walk-in first: the cut at the door comes first
            walk_in = [m for t in tiers for m in t if m.why.get("after_exit")]
            if walk_in:
                w_ids = {id(m) for m in walk_in}
                tiers = [walk_in] + [[m for m in t if id(m) not in w_ids] for t in tiers]
        # what the room asks for (its profile) and variety (never the same kind twice running)
        prof = self.profile(room)
        used = self.__dict__.setdefault("_kinds", {}).get(room, [])
        # moves that cannot play in the time (a pan under 3.8 s reads rushed), that sink past what
        # they show, or (the tour's last room) close by pushing in: the 2D tours' rules
        tiers = [[m for m in t if dur >= MIN_KIND_S.get(m.kind, 0.0) and not self.sinks(m)
                  and not (final and m.kind in ("push", "exit"))] for t in tiers]
        for m in {id(m): m for t in tiers for m in t}.values():
            pr = PRIOR_W * prof["prior"].get(m.kind, 0.0)
            vr = self.sequence_cost(m) + (ROOM_REPEAT_W if m.kind in used else 0.0)
            if kind == "hero" and m.kind in TRAVELS:
                pr += HERO_TRAVEL_W
            if then_forward and breaks_into(m, then_forward):   # the room's next move (item 5)
                vr += CUT_AXIAL_W
            if m.kind not in FLIGHT and m.kind != "detail" and self.ends_on_door(m):
                vr += DOOR_END_W                     # (item 17: not a door to end on)
                m.why["ends_on_door"] = True
            if pr or vr:
                m.score += pr - vr
                m.why["prior"] = round(pr - vr, 2)
        # one of each kind a room: a room's later shots differ in kind (user 2026-11-16: a tour of
        # pans) - a kind again only when nothing else will do
        used = self.__dict__.setdefault("_kinds", {}).get(room, [])
        if used:                                   # (any kind: a room is not two trucks either)
            again = [m for t in tiers for m in t if m.kind in used]
            tiers = [[m for m in t if m.kind not in used] for t in tiers] + [again]
        # the tour's caps are hard (v10: three rises past a cap of two): a kind at its cap only when
        # nothing else will do
        first = covered is None and piece_rank == 0
        cap_of = lambda m: self.at_cap(m.kind, further=covered is not None,          # noqa: E731
                                       own_piece=first and self.on_own_piece(room, m))
        capped = [m for t in tiers for m in t if cap_of(m)]
        if capped:                                 # a kind at its cap only when nothing else will do -
            # for a further angle too: a room shown whole beats variety (user 2026-11-26: the study
            # kept a third of its walls unseen because pans were at their cap)
            c_ids = {id(m) for m in capped}
            tiers = [[m for m in t if id(m) not in c_ids] for t in tiers] + [capped]
        if self.continuous:
            # continuous cuts (client review 2026-11-20, items 5-6): a move that would turn the camera
            # more than 45 deg, or reverse it, across the cut from the previous shot only when no
            # other move will do - whatever its tier (after an exit the next room is entered on the
            # same heading)
            breaks = [m for t in tiers for m in t if (m.why.get("cut") or {}).get("cost", 0.0) > 0]
            if breaks:
                b_ids = {id(m) for m in breaks}
                tiers = [[m for m in t if id(m) not in b_ids] for t in tiers] + [breaks]
            if join_from is not None:
                # the room's first scene carried on from the walk-in (join_from: its last pose):
                # the moves a short bridge reaches first (user 2026-11-27: walk-out, walk-in and the
                # first scene one move; chosen on their own they started 140 deg round from it)
                near_ = [m for t in tiers for m in t if self.join_s(join_from, m) <= JOIN_EST_MAX_S
                         and self.ahead_of(join_from, m)]
                if near_:
                    n_ids = {id(m) for m in near_}
                    tiers = [near_] + [[m for m in t if id(m) not in n_ids] for t in tiers]
            if then_forward and last_shot:
                # the room is left moving forward: a shot ending backward reverses at that cut, so it
                # comes only when no other move will do - whatever its tier (user 2026-11-27: the green
                # house bathroom backed out, then the bedroom was walked into; a door pan instead)
                back = [m for t in tiers for m in t if breaks_into(m, then_forward)]
                if back:
                    k_ids = {id(m) for m in back}
                    tiers = [[m for m in t if id(m) not in k_ids] for t in tiers] + [back]
        tiers = [t for t in tiers if t]
        stats["to_render_checks"] += _n(tiers)
        if not tiers:
            return None, [f"room_no_move:{room}" if kind != "detail" else f"room_no_detail:{room}"]
        for m in {id(m): m for t in tiers for m in t}.values():
            if m.kind == "detail" or "front_m" not in m.why:
                continue
            if (piece_rank == 0 and covered is None and m.kind in ("walkway", "arc") and m.piece is not None
                    and (m.piece.cls in TYPE_WHOLE or m.piece.cls is None)):     # (unlabelled: its main piece)
                # a room's first shot: the walk / arc on its bed or sofa (user 2026-11-18 liked the walk;
                # it held by a 0.01 score margin, which pacing its path to the 3D pace undid, 2026-11-27)
                m.score += ON_PIECE_W
            if opening:
                # the tour's first shot moves into the home, toward what it shows (user 2026-11-16:
                # "in the first shot we should pan in instead of zoom out")
                d0 = float(np.linalg.norm(m.tgt[0][:2] - m.eye[0]))
                d1 = float(np.linalg.norm(m.tgt[-1][:2] - m.eye[-1]))
                m.score += OPEN_IN_W * float(np.clip(d0 - d1, -1.0, 1.5))
                m.why["moves_in_m"] = round(d0 - d1, 2)
            else:                                                     # pulling back, the view widening
                f0, f1 = m.why["front_m"]
                m.score += WIDE_W * float(np.clip(f1 - f0, -1.0, 1.5))
        # one budget of rendered checks for the shot, over all its tiers: a weak scan's room has
        # dozens of tiers (preference x clearance x fallbacks), each once allowed OCC_CHECK_KIND a
        # kind - Home Scan planning went from 6 to 30+ minutes
        # a room's first shot keeps the full budget (a room skipped is worse than a slow plan: the
        # Home Scan's primary bedroom failed the void check 60 times running, a scan with no ceiling);
        # further angles get SHOT_CHECKS
        self._budget = {"kinds": {}, "left": FIRST_SHOT_CHECKS if (covered is None and piece_rank == 0) else SHOT_CHECKS}
        least, seen = None, set()
        for cands in tiers:
            for m in cands:
                if id(m) in seen:
                    continue
                seen.add(id(m))
                if m.kind != "detail":
                    b = self._feature_bonus(room, m)                      # what the view shows (labels)
                    m.score += b
                    if b:
                        m.why["features"] = round(b / FEATURE_W, 2)
            best, low = self._check(room, cands, dur)
            if low is not None and (least is None or low.occ < least.occ):
                least = low
            if best is not None:
                if typed is not None and id(best) not in typed:
                    type_warn = [f"room_type_not_in_view:{room}"]
                best.why.update(candidates=len(cands), occlusion=round(best.occ, 3))
                self._kinds.setdefault(room, []).append(best.kind)
                self.record(best, dur)
                return best, type_warn
        if least is None:
            return None, [f"room_no_move:{room}"]
        if covered is not None:                   # a further shot is a good one or none (user 2026-11-16:
            return None, [f"coverage_no_move:{room}"]     # a held view ended on a blur at the lens)
        hold = self._hold(least, room)
        # a hold passes the same checks as any shot (v12: four holds, the black band and a card at
        # the lens in them): a room whose only shot is a bad hold is skipped
        if self.occ is not None:
            hold.why.pop("near_lens", None)
            bad = self._cover(hold) >= 1.0 or self._void(hold) > VOID_MAX
            if bad:
                return None, [f"room_only_bad_hold:{room}"] + type_warn
        self.record(hold, dur)
        return hold, [f"view_partly_blocked:{room}"] + type_warn

    def front(self, e, t, past=None) -> float:
        """Metres to the nearest wall or tall thing across the middle third of the view from e to t;
        past: a piece the view is of - its own cells do not count (green house v15: walks onto the
        primary bed stopped 0.5 m out, at the bed)."""
        h = math.atan2(t[1] - e[1], t[0] - e[0])
        hs = h + np.radians(np.linspace(-HFOV / 6, HFOV / 6, 7))
        if past is None:
            return float(self.space.rays(e, hs, max_m=6.0).min())
        return float(self.space.rays(e, hs, max_m=6.0, past=past).min())

    def stage_stats(self, room: str, further: bool) -> dict:
        """How a room's candidates fared, its first shot and its further angles apart: generated,
        passed_rules (room to breathe, framing, own room), to_render_checks (after the type, variety
        and cap rules), then each render check's failures (fail_speed / near_lens / edge / blocked /
        void) - the planning report's rooms[].checks (why a room was skipped, or got one angle)."""
        from collections import Counter
        all_ = self.__dict__.setdefault("check_stats", {})
        return all_.setdefault(room, {}).setdefault("further" if further else "first", Counter())

    def on_own_piece(self, room: str, m: Move) -> bool:
        """A walk or arc onto the piece the room's type is about (a dining room's table), standing
        in the room, that frames it."""
        from tour_worker.planner3d.space import PREFER
        from tour_worker.scoring.rooms import canon
        if m.kind not in ("walkway", "arc") or m.piece is None or m.piece.cls is None:
            return False
        if canon(self.space.room_types.get(room)) not in PREFER.get(m.piece.cls, ()):
            return False
        if self.space.piece_room(m.piece) != room:
            return False
        # close to it all along: from across the living room it is a living room shot (a first
        # try at this rule picked a pull-back 6-7 m from the table)
        fp = m.piece.frame or m.piece
        if float(np.linalg.norm(m.eye - fp.centre, axis=1).max()) > OWN_PIECE_NEAR:
            return False
        return self.framed(m)

    def aim_on(self, fp) -> np.ndarray:
        """Where a walk or arc onto a piece aims: its middle; a bed's, BED_AIM toward its head (the
        headboard and pillows are what a bedroom shot is of)."""
        if fp.cls != "bed":
            return np.asarray(fp.centre, float)
        ax, sg = self.bed_head(fp)
        return fp.centre + sg * BED_AIM * (fp.half_long * fp.u if ax == 0 else fp.half_short * fp.v)

    def core_of(self, fp, eye, tgt) -> float | None:
        """How much of the piece stays in frame all along a move (the CORES step it keeps), None when
        it leaves. A bed: BED_SEEN_PATH of its top and its head in every sampled frame (from the
        aisle at a bed's foot no frame holds its four corners - green house v15), scored CORES[-1]."""
        core = next((c for c in CORES if all(in_frame(q, t, fp, c) for q, t in zip(eye[::10], tgt[::10]))), None)
        if core is not None or fp.cls != "bed":
            return core
        head = self.bed_head(fp)
        for q, t in zip(eye[::10], tgt[::10]):
            share, hd = bed_seen(q, t, fp, head)
            if share < BED_SEEN_PATH or not hd:
                return None
        return CORES[-1]

    def bed_head(self, piece) -> tuple[int, int]:
        """The side of a bed against a wall, its head: (axis, sign) - axis 0 the piece's long side,
        1 across it. A floor-plan bed often takes in its nightstands and is wider than long, so all
        four sides are tried: the one with the least clearance just beyond it."""
        sp = self.space
        # a labelled "bed" off the piece's middle is its headboard (green house v15: a 1.96 x 0.41 m
        # strip at the wall) - its side is the head
        for o in sp.room_objects(getattr(self, "_room_now", None) or "", ("bed",)):
            f = o.get("footprint")
            if not f:
                continue
            lu, lv = piece.local(np.asarray(f["centre"], np.float64))
            ru, rv = lu / max(piece.half_long, 1e-6), lv / max(piece.half_short, 1e-6)
            if max(abs(ru), abs(rv)) >= 0.5 and max(abs(ru), abs(rv)) <= 1.3:
                return (0, 1 if ru > 0 else -1) if abs(ru) >= abs(rv) else (1, 1 if rv > 0 else -1)
        best, side = None, (0, 1)
        for ax, (w, h_, d) in enumerate(((piece.u, piece.half_long, piece.v), (piece.v, piece.half_short, piece.u))):
            span = piece.half_short if ax == 0 else piece.half_long
            for sg in (1, -1):
                c = np.mean([sp.clearance(piece.centre + sg * (h_ + 0.15) * w + t * 0.6 * span * d) for t in (-1, 0, 1)])
                if best is None or c < best:
                    best, side = c, (ax, sg)
        return side

    def _roomy(self, m: Move) -> bool:
        """Room to breathe at both ends (FRONT_MIN, an enter's start FRONT_MIN_ENTER) and the end at
        least AIM_MIN from what it aims at. Close-ups and the hallway links are exempt."""
        if m.kind in ("detail", "link"):
            return True
        if "front_m" not in m.why:
            pv = (m.piece.frame or m.piece) if m.piece is not None and m.kind in ("walkway", "arc") else None
            m.why["front_m"] = (round(self.front(m.eye[0], m.tgt[0], pv), 2), round(self.front(m.eye[-1], m.tgt[-1], pv), 2))
        f0, f1 = m.why["front_m"]
        aim = float(np.linalg.norm(m.tgt[-1][:2] - m.eye[-1]))
        k = RELAX if getattr(self, "relaxed", False) else 1.0
        if m.kind == "backout":
            return f0 >= BACKOUT_FRONT[0] and f1 >= BACKOUT_FRONT[1]
        if m.kind == "door_pan":
            return min(f0, f1) >= DOOR_PAN_FRONT
        front, aim_min = self.breathe(getattr(self, "_room_now", None))
        f0_min = (AFTER_DOOR_FRONT if m.why.get("after_exit") else min(FRONT_MIN_ENTER, front) if m.kind == "enter"
                  else front)
        return f0 >= k * f0_min and f1 >= k * front and aim >= k * aim_min

    def breathe(self, room) -> tuple[float, float]:
        """(room in front, distance to the subject) a shot of `room` needs: FRONT_MIN / AIM_MIN in a
        big room, half the room's side in a small one (BREATHE_MIN at least) - a 3 m bedroom never
        has 2.2 m in front at both ends (green house v12: 7000 of its further angles failed on it,
        each room one shot)."""
        if room is None:
            return FRONT_MIN, AIM_MIN
        cache = self.__dict__.setdefault("_breathe", {})
        if room not in cache:
            sp = self.space
            foot = float((sp.room_fill == sp.room_keys.index(room)).sum()) * 0.05 * 0.05 if room in sp.room_keys else 0.0
            side = math.sqrt(max(foot, 0.0))
            cache[room] = (float(np.clip(BREATHE_SHARE * side, BREATHE_MIN, FRONT_MIN)),
                           float(np.clip(BREATHE_SHARE * side, BREATHE_MIN, AIM_MIN)))
        return cache[room]

    def _good(self, m: Move, dur: float) -> bool:
        """A room shot is seen to move, never starts or ends in a doorway (a walk-in may start in
        one), and ends with its subject whole and floor under it."""
        if m.kind in ("detail", "link"):
            return True
        sp = self.space
        ends = ((m.eye[-1],) if m.kind == "enter" else (m.eye[0],) if m.kind == "backout"
                else () if m.kind == "door_pan" else (m.eye[0], m.eye[-1]))     # (a door pan stands by its door)
        if any(sp.clearance(q) < WALL_CLEAR or self.in_doorway(q) for q in ends):
            return False
        if m.kind == "wall_pan" and not self.ends_on_something(getattr(self, "_room_now", None) or "", m.eye[-1], m.tgt[-1]):
            return False                          # (the study's pan onto a bare wall and a door frame)
        if self.pitch_room(m) < -0.5:            # never steeper than the cap (PITCH_MAX_ROOM / _RISE)
            return False
        if m.why.get("after_exit"):              # a matched walk-in: the cut at the door comes first
            if not self.moves_enough(m, dur):
                return False
            if not (self.framed(m) or getattr(self, "relaxed", False)) and not m.why.get("unframed"):
                m.why["unframed"] = True
                m.score -= UNFRAMED_W
            return True
        return self.moves_enough(m, dur) and (self.framed(m) or getattr(self, "relaxed", False))

    def _points(self, room: str, mask: np.ndarray) -> np.ndarray:
        """Where shots may start and end: spread over the whole space a shot may move through, and
        again over the room's own floor - spread over a whole open house, a small room's share of
        the points lands in its corners and doorways (none where a camera would stand)."""
        sp = self.space
        pts = sp.spread_points(mask, 36)
        own = sp.spread_points(mask & (sp.room_of == sp.room_keys.index(room)), OWN_POINTS, min_gap=0.4)
        if not len(own):
            return pts
        keep = [p for p in own if not len(pts) or np.linalg.norm(pts - p, axis=1).min() > 0.3]
        return np.concatenate([pts, np.array(keep).reshape(-1, 2)]) if keep else pts

    def in_doorway(self, q) -> bool:
        """q within DOOR_NEAR of a doorway: where two rooms meet through a gap narrower than
        DOOR_CLEAR x 2 (an open-plan seam is not a doorway)."""
        sp = self.space
        if "_doors" not in self.__dict__:
            from scipy import ndimage as ndi
            lab, dt = sp.room_of, sp.dt
            seam = np.zeros(lab.shape, bool)
            for ax in (0, 1):
                a, b = np.moveaxis(lab, ax, 0)[:-1], np.moveaxis(lab, ax, 0)[1:]
                m = (a >= 0) & (b >= 0) & (a != b)
                s0, s1 = np.moveaxis(seam, ax, 0)[:-1], np.moveaxis(seam, ax, 0)[1:]
                s0 |= m
                s1 |= m
            # a doorway is a seam narrow between WALLS (room_dt) all across: an open-plan seam is
            # narrow only where it meets a wall, and chairs round a dining table pinch the clearance
            # there too (green house v14-v16: the dining room's every walk onto its table "started
            # in a doorway" - it fell back to a held pan)
            dt = getattr(sp, "room_dt", dt)
            parts, n = ndi.label(seam, structure=np.ones((3, 3), bool))
            if n:
                widest = ndi.maximum(dt, parts, index=np.arange(1, n + 1))
                seam = np.isin(parts, 1 + np.flatnonzero(np.asarray(widest) < DOOR_CLEAR + 0.15))
            from tour_worker.planner3d.space import CELL
            self._doors = ndi.distance_transform_edt(~seam) * CELL <= DOOR_NEAR if seam.any() else seam
        i, j = sp.cell(q)
        return bool(0 <= i < self._doors.shape[0] and 0 <= j < self._doors.shape[1] and self._doors[i, j])

    def walls_seen(self, room: str, m: Move) -> np.ndarray:
        """Flat indices of the room's wall cells the move shows (five poses along it, a cell's
        tolerance for rounding)."""
        sp = self.space
        walls = sp.wall_cells(room)
        if not len(walls):
            return walls
        hit = []
        for k in np.linspace(0, len(m.eye) - 1, 5).astype(int):
            e, t = m.eye[k], m.tgt[k]
            hit.append(sp.seen_cells(e, math.atan2(t[1] - e[1], t[0] - e[0]), HFOV))
        grid = np.zeros(sp.shape, bool)
        grid.flat[np.concatenate(hit)] = True
        grid = ndi.binary_dilation(grid, iterations=1)
        return walls[grid.flat[walls]]

    def new_walls(self, room: str, m: Move, covered: np.ndarray) -> float:
        """Share of the room's walls the move shows that `covered` (flat indices) does not."""
        walls = self.space.wall_cells(room)
        if not len(walls):
            return 0.0
        seen = self.walls_seen(room, m)
        return float(len(np.setdiff1d(seen, covered, assume_unique=False))) / len(walls)

    def _check(self, room: str, cands: list[Move], dur: float) -> tuple[Move | None, Move | None]:
        """(the best candidate within the speed limits, its own room share, the lens rule and the
        void limit; the least blocked one within the speed limits) - best first, lazily, stopping
        once no later candidate can win. The void check renders, so it runs on the winners only -
        and when it turns every winner down, the candidates the early stop skipped are judged next
        (Home Scan v17: the dining room had 118 candidates, ~5 judged; all five showed the missing
        ceiling, and the room got a held pan)."""
        sp = self.space
        cands = sorted(cands, key=lambda m: -m.score)
        bud = getattr(self, "_budget", None) or {"kinds": {}, "left": SHOT_CHECKS}
        import time as _time
        # over the planning budget: further angles and flights settle for their best so far - never a
        # room's first shot (2026-11-26: four Home Scan rooms were skipped with 6 checks each)
        if _time.perf_counter() > getattr(self, "deadline", float("inf")) and getattr(self, "_late_ok", False):
            bud["left"] = min(bud["left"], LATE_CHECKS)
        least, checked, done = None, bud["kinds"], set()
        st = self.stage_stats(room, getattr(self, "_cov_now", False))
        while True:
            best, passed, stopped = None, [], False
            for m in cands:
                if id(m) in done:
                    continue
                if best is not None and m.score <= best.score - OCC_W * best.occ:
                    stopped = True
                    break
                if checked.get(m.kind, 0) >= OCC_CHECK_KIND:      # every kind gets its share of checks
                    continue
                if bud["left"] <= 0:                            # the shot's checks are spent
                    st["budget_out"] += 1
                    break
                if m.kind in OWN_SHARE and "own_share" not in m.why:    # it films its own room
                    k = len(m.eye) // 2
                    share = sp.own_share(room, m.eye[k], math.atan2(*(m.tgt[k][:2] - m.eye[k])[::-1]), HFOV)
                    m.why["own_share"] = round(share, 2)
                    # a view mostly of another room loses score (a small room in an open plan still
                    # gets its best view: there is no view of it without its neighbour)
                    m.score -= OWN_W * max(0.0, OWN_SHARE[m.kind] - share)
                    if best is not None and m.score <= best.score - OCC_W * best.occ:
                        continue
                done.add(id(m))
                if "turn_dps" not in m.why:
                    turn, move = trial_speeds(m.eye, m.tgt, dur, m.zs, "linear" if self.continuous else m.easing)
                    m.why.update(turn_dps=round(turn, 1), move_mps=round(move, 2))
                if m.why["turn_dps"] > self.turn_budget or m.why["move_mps"] > self.move_budget:
                    st["fail_speed"] += 1
                    continue
                checked[m.kind] = checked.get(m.kind, 0) + 1
                before = self.__dict__.get("_occ_renders", 0)
                m.occ = self._cover(m)
                if self.__dict__.get("_occ_renders", 0) > before:       # (served from the cache: free)
                    bud["left"] -= 1
                if least is None or m.occ < least.occ:
                    least = m
                if m.occ > MAX_OCC:
                    st["fail_near_lens" if "near_lens" in m.why else "fail_edge" if "edge_cover" in m.why else "fail_blocked"] += 1
                    continue
                passed.append(m)
                if best is None or m.score - OCC_W * m.occ > best.score - OCC_W * best.occ:
                    best = m
            if self.occ is None or best is None:
                return best, least
            # what of its view the scan never captured (rendered, so only for the winners): best
            # first, down the ranking until one passes
            for m in sorted(passed, key=lambda m: -(m.score - OCC_W * m.occ)):
                if m.kind != "detail":
                    before = self.__dict__.get("_occ_renders", 0)
                    v = self._void(m)
                    if self.__dict__.get("_occ_renders", 0) > before:   # a check served from the cache is free
                        bud["left"] -= 1
                    m.why["void"] = round(v, 3)
                    if v > VOID_MAX:
                        st["fail_void"] += 1
                        continue
                return m, least
            if not stopped:                       # every candidate judged: none
                return None, least

    # ---- what a view shows ----------------------------------------------------------------------
    def ends_on_something(self, room: str, eye2, tgt) -> bool:
        """The frame shows something beyond a door, a window or a curtain (a further angle onto a bare
        wall and a door frame - the green house study, v13 - is no shot)."""
        return self.features_seen(room, eye2, tgt, skip=("door", "windowpane", "curtain", "blind"), own=True) >= COVER_FEATURE_MIN

    def across_zone(self, room: str, m: Move) -> float:
        """Share of the move's last line of sight (eye to target, on the floor plan) over the floor of
        another room of its open plan (a dining, kitchen or living area; hallways do not count) - a
        view across it."""
        sp = self.space
        types = getattr(sp, "room_types", {}) or {}
        mine = sp.spaces.get(room, {room})
        e, t = np.asarray(m.eye[-1][:2]), np.asarray(m.tgt[-1][:2])
        rooms = [sp.room_near(e + (t - e) * a, 0.5) for a in np.linspace(0.05, 0.95, 19)]
        share = sum(1 for r in rooms if r is not None and r != room and r in mine
                    and (types.get(r) or "") not in ("hallway", "corridor", "landing")) / len(rooms)
        m.why["across_zone"] = round(share, 2)
        return share

    def ends_on_door(self, m: Move) -> bool:
        """At the move's last frame the nearest labelled thing in the middle of the view (within
        END_CONE_DEG of its centre) is a door (client review item 17: a shot ends on something worth
        showing - the green house kitchen's first shot ended on its door)."""
        sp = self.space
        e, t = np.asarray(m.eye[-1][:2]), np.asarray(m.tgt[-1][:2])
        head = math.atan2(*(t - e)[::-1])
        best = None
        for o in sp.objects:
            v = np.asarray(o["centre"][:2]) - e
            dist = float(np.linalg.norm(v))
            if not 0.4 < dist < 8.0:
                continue
            a = (math.atan2(v[1], v[0]) - head + math.pi) % (2 * math.pi) - math.pi
            if abs(a) > math.radians(END_CONE_DEG):
                continue
            if sp.ray(e, math.atan2(v[1], v[0]), max_m=dist + 0.3) < dist - 0.6:
                continue                                           # behind a wall
            if best is None or dist < best[0]:
                best = (dist, o["class"])
        return best is not None and best[1] in ("door", "screen door")

    def features_seen(self, room: str, eye2, tgt, skip=(), own=False) -> float:
        """Sum of FEATURES weights of the room's (and its open plan's) labelled features in the
        middle 70% of the frame, in front within 8 m."""
        sp = self.space
        rooms = {room} if own else sp.spaces.get(room, {room})     # own: not things seen through a door
        d = np.asarray(tgt[:2]) - np.asarray(eye2[:2])
        head = math.atan2(d[1], d[0])
        half = math.radians(HFOV) * 0.35
        w = 0.0
        for o in sp.objects:
            if o.get("room") not in rooms or o["class"] not in FEATURES or o["class"] in skip:
                continue
            v = np.asarray(o["centre"][:2]) - np.asarray(eye2[:2])
            dist = float(np.linalg.norm(v))
            if not 0.5 < dist < 8.0:
                continue
            a = (math.atan2(v[1], v[0]) - head + math.pi) % (2 * math.pi) - math.pi
            if abs(a) <= half:
                w += FEATURES[o["class"]]
        return min(w, 2.0)

    def _feature_bonus(self, room: str, m: Move) -> float:
        k = len(m.eye) // 2
        return FEATURE_W * max(self.features_seen(room, m.eye[0], m.tgt[0]), self.features_seen(room, m.eye[k], m.tgt[k]))

    def _detail(self, room, rmin, dur, prev, earlier) -> list[Move]:
        """Close-ups of the room's details (scene labels): from DETAIL_DIST m off, the eye a little
        above the object, a slow DETAIL_PUSH m push-in; the object well inside the frame."""
        sp = self.space
        out = []
        objs = [o for o in sp.room_objects(room, DETAILS)
                if DETAIL_SIZE[0] <= max(o["max"][0] - o["min"][0], o["max"][1] - o["min"][1]) <= DETAIL_SIZE[1]]
        if prev is not None:                          # not what the previous shot ended on
            pe, pt = prev
            objs = [o for o in objs if np.linalg.norm(np.asarray(o["centre"][:2]) - np.asarray(pt[:2])) > DETAIL_NOT_PREV]
        for o in objs:
            c = np.asarray(o["centre"], float)
            size = max(o["max"][0] - o["min"][0], o["max"][1] - o["min"][1], o["max"][2] - o["min"][2])
            z_eye = float(np.clip(c[2] + 0.35, 0.9, 1.6))
            for dist in DETAIL_DIST:
                for ang in range(0, 360, 30):
                    a = math.radians(ang)
                    p0 = c[:2] + (dist + DETAIL_PUSH) * np.array([math.cos(a), math.sin(a)])
                    p1 = c[:2] + dist * np.array([math.cos(a), math.sin(a)])
                    if sp.clearance(p0) < 0.2 or sp.clearance(p1) < 0.2 or sp.room_at(p0) not in sp.spaces.get(room, {room}):
                        continue
                    pitch = math.degrees(math.atan2(z_eye - c[2], dist))
                    if pitch > DETAIL_PITCH_MAX:
                        continue
                    s_ = np.linspace(0, 1, SAMPLES)
                    eye = np.outer(1 - s_, p0) + np.outer(s_, p1)
                    tgt = np.repeat(c[None], SAMPLES, 0)
                    first, last = (eye[0], tgt[0]), (eye[-1], tgt[-1])
                    if jump_cut(prev, first) or repeats(first, last, earlier):
                        continue
                    fill = size / (2 * dist * math.tan(math.radians(HFOV) / 2))
                    if not 0.15 <= fill <= 0.8:
                        continue
                    score = 1.0 * DETAILS[o["class"]] - 1.5 * abs(fill - 0.4) + 0.3 * min(sp.clearance(p0), 0.6)
                    out.append(Move("detail", eye, tgt, score, pitch, None, eye_z=z_eye,
                                    why={"object": o["class"], "distance_m": dist, "fill": round(fill, 2)}))
        return out

    # ---- what a room without a main piece is about --------------------------------------------------
    def anchor(self, room: str) -> np.ndarray | None:
        """The room's working surfaces (ANCHORS by room type: a kitchen's counter run, sink, hob),
        as one aim point: their area-weighted centre at their top. Only low objects of the room
        itself (never upper cabinets, never a neighbour's); None when it has none."""
        from tour_worker.scoring.rooms import canon
        classes = ANCHORS.get(canon(self.space.room_types.get(room)) or "", ())
        objs = [o for o in self.space.room_objects(room, classes)
                if 0.3 <= o["max"][2] <= LOW_TOP and max(o["max"][0] - o["min"][0], o["max"][1] - o["min"][1]) >= 0.2]
        if not objs:
            return None
        w = np.array([(o["max"][0] - o["min"][0]) * (o["max"][1] - o["min"][1]) for o in objs]) + 1e-6
        c = np.array([o["centre"][:2] for o in objs])
        top = np.array([o["max"][2] for o in objs])
        a = np.array([*(w[:, None] * c).sum(0) / w.sum(), float((w * top).sum() / w.sum())])
        # what a push / pull ends on: the working area round the anchor (a counter run along a wall
        # is the room's structure - its middle ANCHOR_HALF m each way, down to the floor)
        box = objects_box(objs)
        if box is not None:
            box[:, 0] = np.clip(box[:, 0], a[0] - ANCHOR_HALF, a[0] + ANCHOR_HALF)
            box[:, 1] = np.clip(box[:, 1], a[1] - ANCHOR_HALF, a[1] + ANCHOR_HALF)
        self.__dict__.setdefault("_anchor_box", {})[room] = box
        return a

    def _push(self, room, anchor, pts, paths, rmin, dur, prev, earlier, own) -> list[Move]:
        """A slow push toward the room's working surfaces, aimed at them from start to end."""
        aim = np.array([anchor[0], anchor[1], min(anchor[2] + 0.15, AIM_Z_MAX)])
        out = []
        for a in sorted(own):
            for b in range(len(pts)):
                if a == b:
                    continue
                L = float(np.linalg.norm(pts[b] - pts[a]))
                if not PUSH_LEN[0] <= L <= PUSH_LEN[1]:
                    continue
                d0, d1 = float(np.linalg.norm(anchor[:2] - pts[a])), float(np.linalg.norm(anchor[:2] - pts[b]))
                if not (PUSH_END[0] <= min(d0, d1) and max(d0, d1) <= PUSH_END[1] + 0.8 and abs(d1 - d0) >= 0.3):
                    continue
                mv, to = pts[b] - pts[a], anchor[:2] - pts[a]
                toward = float(mv @ to) / (np.linalg.norm(mv) * np.linalg.norm(to))
                if abs(toward) < 0.6:                     # straight toward what it shows, or back from it
                    continue
                path, plen = self._path(paths, a, b, rmin)
                if path is None or plen > 1.3 * L or plen / dur * self.peak(1.5) > self.move_budget:
                    continue
                path = resample(path, SAMPLES)
                tgt = cap_pitch(path, np.repeat(aim[None], len(path), 0), PITCH_LADDER[0])
                first, last = (path[0], tgt[0]), (path[-1], tgt[-1])
                if jump_cut(prev, first) or repeats(first, last, earlier):
                    continue
                depth = self.space.view_depth(path[0], math.atan2(*(aim[:2] - path[0])[::-1]), HFOV)
                score = 1.0 + 0.3 * plen + 0.5 * abs(toward) - 0.4 * abs(min(d0, d1) - 2.2) + 0.05 * min(depth, 5.0)
                out.append(Move("push" if toward > 0 else "pull", path, tgt, score,
                                subject=self.__dict__.get("_anchor_box", {}).get(room),
                                why={"travel_m": round(plen, 2), "toward": round(toward, 2),
                                     "end_from_anchor_m": round(d1, 2), "anchor": np.round(anchor, 2).tolist()}))
        return out

    def _goal(self, room: str) -> np.ndarray:
        """What a reveal of `room` turns onto: its main piece (its top), else the middle of its floor."""
        pieces = self.space.main_pieces(room)
        if pieces:
            return np.array([*pieces[0].centre, pieces[0].top])
        sp = self.space
        ys, xs = np.nonzero((sp.room_of == sp.room_keys.index(room)) & (sp.dt >= 0.3))
        return np.array([sp.lo[0] + (xs.mean() + .5) * 0.05, sp.lo[1] + (ys.mean() + .5) * 0.05, EYE - 0.4])

    def _reveal(self, room, next_room, pts, paths, rmin, dur, prev, own) -> list[Move]:
        """The opening from a room with nothing to compose on (a hall): along a free path toward the
        tour's next room, the view turning evenly from down the path onto that room, which fills the
        last frame (REVEAL_SHARE of its middle)."""
        sp = self.space
        goal = self._goal(next_room)
        out = []
        for a in sorted(own):
            for b in range(len(pts)):
                if a == b:
                    continue
                L = float(np.linalg.norm(pts[b] - pts[a]))
                if not MOVE_LEN[0] + 0.4 <= L <= MOVE_LEN[1] + 1.4:
                    continue
                d0, d1 = float(np.linalg.norm(goal[:2] - pts[a])), float(np.linalg.norm(goal[:2] - pts[b]))
                if not (d1 < d0 - 0.5 and d1 >= 1.8):    # toward the next room, not into it
                    continue
                path, plen = self._path(paths, a, b, rmin)
                if path is None or plen > 1.25 * L or plen / dur * self.peak(1.5) > self.move_budget:
                    continue
                path = resample(path, SAMPLES)
                d = path[min(10, len(path) - 1)] - path[0]
                h0 = math.atan2(d[1], d[0])
                hg = math.atan2(goal[1] - path[0][1], goal[0] - path[0][0])
                dh = (h0 - hg + math.pi) % (2 * math.pi) - math.pi
                if abs(dh) > math.radians(REVEAL_SWING):   # start turned part way: an even, slow swing
                    h0 = hg + math.copysign(math.radians(REVEAL_SWING), dh)
                d = np.array([math.cos(h0), math.sin(h0)])
                far = np.array([*(path[0] + d * min(sp.ray(path[0], h0), 6.0)), EYE - LOOK_DROP])
                u = np.linspace(0, 1, len(path))
                w = (u * u * (3 - 2 * u))[:, None]
                tgt = cap_pitch(path, (1 - w) * far + w * goal, PITCH_LADDER[0])
                first = (path[0], tgt[0])
                if jump_cut(prev, first):
                    continue
                h1 = math.atan2(*(tgt[-1][:2] - path[-1])[::-1])
                share = sp.own_share(next_room, path[-1], h1, HFOV)
                if share < REVEAL_SHARE:
                    continue
                depth = sp.view_depth(path[0], h0, HFOV)
                score = 1.0 + 0.3 * plen + 1.5 * share + 0.1 * min(depth, 6.0) + 0.3 * parallax(path, tgt)
                out.append(Move("reveal", path, tgt, score,
                                why={"travel_m": round(plen, 2), "reveals": next_room, "share_next": round(share, 2),
                                     "view_depth_m": round(depth, 2)}))
        return out

    # ---- between rooms ---------------------------------------------------------------------------
    @staticmethod
    def _along(poly: np.ndarray) -> np.ndarray:
        return np.r_[0, np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))]

    def link(self, src: str, hall: str, dst: str, route, dur: float, prev=None) -> tuple[Move | None, list[str]]:
        """The link shot: the tour leaves `src` for `dst` through `hall`. From a spot of src's space
        on the walking route, LINK_BEFORE m before the hallway, a slow pan from src's side onto the
        hallway (LINK_AHEAD m into it, toward dst), drifting LINK_DRIFT m toward it."""
        sp = self.space
        self.turn_budget, self.move_budget = MAX_TURN_DPS * PRE_SYNC_SHARE["room"], MAX_MOVE_MPS * PRE_SYNC_SHARE["room"]
        poly, _ = route
        at = [sp.room_at(q) for q in poly]
        h_in = next((k for k, r in enumerate(at) if r == hall), None)
        if h_in is None:
            return None, [f"link_no_route:{hall}"]
        s_ = self._along(poly)
        mine = sp.spaces.get(src, {src})
        cands = []
        for k in range(h_in):
            back = s_[h_in] - s_[k]
            if not LINK_BEFORE[0] <= back <= LINK_BEFORE[1] or at[k] not in mine | {hall}:
                continue
            p0 = poly[k]
            if sp.clearance(p0) < RMIN_LADDER[-1]:
                continue
            k1 = int(np.searchsorted(s_, s_[k] + LINK_DRIFT))
            p1 = poly[min(k1, h_in)]
            # it ends looking down the hallway: at the farthest point of the route past the hallway's
            # start still in sight (at least LINK_AHEAD m on), toward the next room
            aim = None
            for j in range(len(poly) - 1, h_in, -1):
                if s_[j] - s_[h_in] < LINK_AHEAD:
                    break
                d = poly[j] - p1
                dist = float(np.linalg.norm(d))
                if dist > 0.5 and sp.ray(p1, math.atan2(d[1], d[0]), max_m=dist + 0.5) >= 0.95 * dist:
                    aim = np.array([*poly[j], EYE - LOOK_DROP])
                    break
            if aim is None:
                continue
            h_end = math.atan2(aim[1] - p1[1], aim[0] - p1[0])
            for sign in (1, -1):
                # start turned toward the room being left: the side whose view holds more of it
                for swing in np.arange(LINK_SWING[1], LINK_SWING[0] - 1e-6, -5.0):
                    h0 = h_end + sign * math.radians(swing)
                    u = np.linspace(0, 1, SAMPLES)
                    w = u * u * (3 - 2 * u)
                    eye = np.outer(1 - u, p0) + np.outer(u, p1)
                    hs = h0 + (h_end - h0) * w
                    rr = np.linalg.norm(aim[:2] - eye, axis=1)
                    tgt = np.c_[eye[:, 0] + rr * np.cos(hs), eye[:, 1] + rr * np.sin(hs), np.full(SAMPLES, aim[2])]
                    turn, move = trial_speeds(eye, tgt, dur)
                    if turn <= self.turn_budget and move <= self.move_budget:
                        break
                else:
                    continue
                if jump_cut(prev, (eye[0], tgt[0])):
                    continue
                src_share = sp.own_share(src, eye[0], h0, HFOV)
                hall_share = sp.own_share(hall, eye[-1], h_end, HFOV)
                score = 1.0 + 1.0 * src_share + 1.5 * hall_share + 0.3 * swing / LINK_SWING[1] \
                    + 0.1 * min(sp.view_depth(eye[-1], h_end, HFOV), 6.0)
                cands.append(Move("link", eye, tgt, score, why={
                    "from": src, "onto": hall, "toward": dst, "looks_down_m": round(float(np.linalg.norm(aim[:2] - p1)), 2), "swing_deg": float(swing), "src_share": round(src_share, 2),
                    "hall_share": round(hall_share, 2), "turn_dps": round(turn, 1), "move_mps": round(move, 2)}))
        if not cands:
            return None, [f"link_no_move:{hall}"]
        cands = self.keep_direction(cands)
        best, least = self._check(hall, cands, dur)
        if best is None:
            return None, [f"link_blocked:{hall}"]
        best.why.update(candidates=len(cands), occlusion=round(best.occ, 3))
        return best, []

    def _enter(self, room, piece, route, rmin, dur, prev, earlier, after_exit=False) -> list[Move]:
        """In along the walking route: from ENTER_BEFORE m before the doorway (in the hallway) to a
        spot ENTER_AFTER m into the room, looking at its main piece (else the middle of its floor).
        After an exit (the last room's shot flew to its doorway): it starts AFTER_EXIT m inside,
        still heading along the route, and turns onto the piece - one flight, cut at the door."""
        if after_exit:
            return self._enter_after_exit(room, piece, route, rmin, dur, earlier)
        sp = self.space
        poly, _ = route
        at = [sp.room_at(q) for q in poly]
        d_in = next((k for k, r in enumerate(at) if r == room), None)
        if d_in is None or d_in == 0:
            return []
        door = poly[d_in]
        fp = (piece.frame or piece) if piece is not None else None
        aim = np.array([*fp.centre, fp.top]) if fp is not None else self._goal(room)
        s_ = self._along(poly)
        starts = [k for k in range(d_in) if ENTER_BEFORE[0] <= s_[d_in] - s_[k] <= ENTER_BEFORE[1]
                  and sp.clearance(poly[k]) >= rmin][::3]
        mask = (sp.dt >= rmin) & (sp.room_of >= 0)
        inside = sp.spread_points(sp.space_mask(room, rmin), 36)
        ends = [q for q in inside if sp.room_at(q) == room and ENTER_AFTER[0] <= np.linalg.norm(q - door) <= ENTER_AFTER[1]]
        if not starts or not ends:
            return []
        pts = np.array([poly[k] for k in starts] + list(ends))
        paths = sp.paths_between(pts, mask)
        out = []
        for a in range(len(starts)):
            for b in range(len(starts), len(pts)):
                path, plen = self._path(paths, a, b, rmin)
                L = float(np.linalg.norm(pts[b] - pts[a]))
                if path is None or plen > 1.4 * L or plen / dur * self.peak(1.5) > self.move_budget:
                    continue
                path = resample(path, SAMPLES)
                tgt = cap_pitch(path, np.repeat(aim[None], len(path), 0), PITCH_LADDER[0])
                first, last = (path[0], tgt[0]), (path[-1], tgt[-1])
                if jump_cut(prev, first) or repeats(first, last, earlier):
                    continue
                share = sp.own_share(room, path[0], math.atan2(*(aim[:2] - path[0])[::-1]), HFOV)
                if share < ENTER_SHARE:                          # it sees into the room from the start
                    continue
                core = 0.0
                if fp is not None:
                    core = next((c for c in CORES if in_frame(path[-1], tgt[-1], fp, c)), 0.0)
                score = 1.0 + 0.3 * plen + 1.0 * share + 1.0 * core
                out.append(Move("enter", path, tgt, score, piece=piece, subject=piece_box(fp) if fp is not None else None,
                                why={"travel_m": round(plen, 2), "share_start": round(share, 2), "core_end": core,
                                     "through": [r for r in route[1] if r != room][-1:] or None}))
        return out

    # ---- small rooms: moves that turn (user 2026-11-16, Krk: "there is almost no pan") ------------
    def _arc(self, room, piece, rmin, dur, prev, earlier) -> list[Move]:
        """An arc around the room's piece: the camera slides along a circle round it (ARC_R m out,
        ARC_SWEEP deg), aimed at its top all along - movement and parallax where a straight walk has
        no room. Every point on free floor of the room's space; the piece's top in frame throughout."""
        sp = self.space
        fp = piece.frame or piece
        c = np.asarray(fp.centre, float)
        top = np.array([*self.aim_on(fp), fp.top])
        mine = sp.spaces.get(room, {room})
        out = []
        u = np.linspace(0, 1, SAMPLES)
        for r in ARC_R:
            for a0 in np.radians(np.arange(0, 360, 20)):
                for sweep in ARC_SWEEP:
                    for sign in (1, -1):
                        a = a0 + sign * math.radians(sweep) * u
                        eye = c + r * np.c_[np.cos(a), np.sin(a)]
                        if any(sp.clearance(q) < rmin or sp.room_at(q) not in mine for q in eye[::7]):
                            continue
                        tgt = cap_pitch(eye, np.repeat(top[None], SAMPLES, 0), PITCH_LADDER[0])
                        first, last = (eye[0], tgt[0]), (eye[-1], tgt[-1])
                        if jump_cut(prev, first) or repeats(first, last, earlier):
                            continue
                        core = self.core_of(fp, eye, tgt)
                        if core is None:
                            continue
                        score = 1.0 + 1.5 * core + 0.5 * sweep / ARC_SWEEP[-1] - 0.3 * abs(r - 2.7)
                        out.append(Move("arc", eye, tgt, score, PITCH_LADDER[0], piece, subject=piece_box(fp),
                                        why={"radius_m": r, "sweep_deg": sweep, "core": core}))
        return _keep_best(out)

    def _subjects(self, room: str) -> list[tuple[np.ndarray, np.ndarray, str]]:
        """What a pan of the room may show: its pieces and its low features - (aim point, box, name)."""
        sp = self.space
        out = []
        for pc in sp.main_pieces(room):
            fp = pc.frame or pc
            out.append((np.array([*fp.centre, fp.top]), piece_box(fp), pc.cls or "piece"))
        for o in sp.room_objects(room, FEATURES):
            if o["max"][2] <= LOW_TOP and o["class"] not in NO_AIM:
                out.append((np.array([*o["centre"][:2], max(o["centre"][2], EYE - 0.6)]), objects_box([o]), o["class"]))
        return out

    def _room_pan(self, room, pts, rmin, dur, prev, earlier) -> list[Move]:
        """A pan across the room's own things: from an open spot, the view turns from one subject to
        another (20-55 deg apart, both at least AIM_MIN away), drifting a little toward them."""
        sp = self.space
        subs = self._subjects(room)
        if len(subs) < 2:
            return []
        out = []
        u = np.linspace(0, 1, SAMPLES)
        w = (u * u * (3 - 2 * u))[:, None]
        spots = sorted(pts, key=lambda p: -sp.clearance(p))[:10]
        for p in spots:
            for i, (ta, _, na) in enumerate(subs):
                for j, (tb, box_b, nb) in enumerate(subs):
                    if i == j:
                        continue
                    da, db = ta[:2] - p, tb[:2] - p
                    if np.linalg.norm(da) < AIM_MIN or np.linalg.norm(db) < AIM_MIN:
                        continue
                    ang = math.degrees(abs((math.atan2(db[1], db[0]) - math.atan2(da[1], da[0]) + math.pi) % (2 * math.pi) - math.pi))
                    if not 20.0 <= ang <= 55.0:
                        continue
                    mid = (ta[:2] + tb[:2]) / 2 - p
                    e1 = p + 0.3 * mid / max(1e-9, np.linalg.norm(mid))
                    if sp.clearance(e1) < rmin or sp.room_at(e1) != room:
                        e1 = p
                    eye = np.outer(1 - u, p) + np.outer(u, e1)
                    tgt = cap_pitch(eye, (1 - w) * ta + w * tb, PITCH_LADDER[0])
                    first, last = (eye[0], tgt[0]), (eye[-1], tgt[-1])
                    if jump_cut(prev, first) or repeats(first, last, earlier):
                        continue
                    score = 0.9 + 0.4 * ang / 55.0 + 0.05 * min(sp.view_depth(p, math.atan2(mid[1], mid[0]), HFOV), 5.0)
                    out.append(Move("room_pan", eye, tgt, score, subject=box_b,
                                    why={"from": na, "to": nb, "sweep_deg": round(ang)}))
        return _keep_best(out)

    # ---- the scene picks the move ----------------------------------------------------------------
    def type_signs(self, room: str) -> list[dict]:
        from tour_worker.scoring.rooms import canon
        cls = TYPE_SIGNS.get(canon(self.space.room_types.get(room)) or "", ())
        return self.space.room_objects(room, cls) if cls else []

    def sees(self, eye2, tgt, o: dict, ez: float = EYE) -> bool:
        """o's centre inside the frame (its middle 80%) and in sight (no wall between)."""
        sp = self.space
        c = np.asarray(o["centre"], float)
        v = c[:2] - np.asarray(eye2[:2])
        dist = float(np.linalg.norm(v))
        if not 0.5 < dist < 9.0:
            return False
        V, K = look_at([eye2[0], eye2[1], ez], tgt), intrinsics(HFOV, FRAME_W, FRAME_H)
        q = V[:3, :3] @ c + V[:3, 3]
        if q[2] < 0.3:
            return False
        px, py = K[0, 0] * q[0] / q[2] + K[0, 2], K[1, 1] * q[1] / q[2] + K[1, 2]
        if not (0.1 * FRAME_W <= px <= 0.9 * FRAME_W and 0.05 * FRAME_H <= py <= 0.95 * FRAME_H):
            return False
        return sp.ray(np.asarray(eye2[:2]), math.atan2(v[1], v[0]), max_m=dist + 0.3) >= dist - 0.6

    def whole_boxes(self, room: str) -> list:
        """(8, 3) boxes of the room's beds / sofas, each its labelled pieces merged (WHOLE_GAP)."""
        cache = self.__dict__.setdefault("_whole", {})
        if room in cache:
            return cache[room]
        out = []
        for cls in TYPE_WHOLE:
            groups = [[o] for o in self.space.room_objects(room, (cls,))]
            merged = True
            while merged and len(groups) > 1:
                merged = False
                for a in range(len(groups)):
                    for b in range(a + 1, len(groups)):
                        la = np.min([o["min"] for o in groups[a]], 0); ha = np.max([o["max"] for o in groups[a]], 0)
                        lb = np.min([o["min"] for o in groups[b]], 0); hb = np.max([o["max"] for o in groups[b]], 0)
                        gap = np.maximum(0.0, np.maximum(la[:2] - hb[:2], lb[:2] - ha[:2])).max()
                        if gap <= WHOLE_GAP:
                            groups[a] += groups.pop(b)
                            merged = True
                            break
                    if merged:
                        break
            out += [objects_box(g) for g in groups]
        cache[room] = [b for b in out if b is not None]
        return cache[room]

    def shows_type(self, room: str, m: Move, whole: bool = False) -> bool:
        """The move's middle or last frame shows something that makes the room what it is; whole:
        a bed or sofa (TYPE_WHOLE) among them is in the last frame whole, with floor under it."""
        signs = self.type_signs(room)
        zs = m.zs
        if whole:
            # the whole bed / sofa: its labelled pieces merged (user 2026-11-17: the primary bedroom
            # passed on a fragment labelled "bed" while the frame showed a corner of the bed)
            hit = any(frames_fully(m.eye[-1], m.tgt[-1], box, float(zs[-1])) for box in self.whole_boxes(room))
            m.why["frames_type"] = hit
            return hit
        hit = any(self.sees(m.eye[k], m.tgt[k], o, float(zs[k])) for k in (len(m.eye) // 2, len(m.eye) - 1) for o in signs)
        m.why["shows_type"] = hit
        return hit

    # ---- the tour's sequence (the photo tours' rules, planner/cinematic.py) ---------------------
    @staticmethod
    def side(m: Move) -> int:
        """Which way the view travels across the frame: +1 / -1 (sideways travel, else the turn), 0."""
        k = len(m.eye) // 2
        h = m.tgt[k][:2] - m.eye[k]
        v = m.eye[-1] - m.eye[0]
        if np.linalg.norm(v) >= 0.3 and parallax(m.eye, m.tgt) >= 0.35:
            c = h[0] * v[1] - h[1] * v[0]
        else:
            d0, d1 = m.tgt[0][:2] - m.eye[0], m.tgt[-1][:2] - m.eye[-1]
            c = d0[0] * d1[1] - d0[1] * d1[0]
            if abs(math.degrees(math.atan2(c, float(d0 @ d1)))) < 8.0:
                return 0
        return 1 if c > 0 else -1

    def record(self, m: Move, dur: float | None = None) -> None:
        """A shot the tour plays: its kind and side, for the next shots' sequence costs, and the way
        it leaves the frame (continuous cuts: the next shot carries it on) - its direction and its
        speed (%fw/s, the next shot is planned to it: to_pace)."""
        self.__dict__.setdefault("_history", []).append((m.kind, self.side(m)))
        self._last_kind = m.kind
        self._last_exit = move_motion(m, end=True)
        dur = dur or m.why.get("dur_s")
        self._last_exit_pct = self.pace_ends(m, float(dur))[1] if dur else None

    def keep_direction(self, cands: list) -> list:
        """Continuous cuts: the candidates that carry the previous shot's move on (cut_cost <= 0),
        or all of them when none does."""
        if not self.continuous or self.__dict__.get("_last_exit") is None:
            return cands
        ok = [m for m in cands if self.cut_cost(m) <= 0]
        return ok or cands

    def cut_cost(self, m: Move) -> float:
        """Continuous cuts (client review 2026-11-20, item 5): entering m from the previous shot's
        exit - a lateral turn over 45 deg or forward into backward costs CUT_TURN_W / CUT_AXIAL_W,
        carrying the move on earns CUT_SAME_W. Measured as render/continuity.py measures a cut."""
        prev = self.__dict__.get("_last_exit")
        if prev is None:
            return 0.0
        from tour_worker.render.continuity import MAX_TURN_DEG, _axial_reversal, _turn_deg
        a, b = _unit(prev), _unit(move_motion(m, end=False))
        turn, rev = _turn_deg(a, b), _axial_reversal(a, b)
        cost = (CUT_TURN_W if turn is not None and turn > MAX_TURN_DEG else 0.0) + (CUT_AXIAL_W if rev else 0.0)
        if not cost and ((turn is not None and turn <= 20.0) or (a.f > 2.5 and b.f > 2.5)):
            cost = -CUT_SAME_W
        m.why["cut"] = {"turn_deg": None if turn is None else round(turn, 1), "forward_back": rev, "cost": cost}
        return cost

    def _reversed(self, cands: list, then_forward: bool = False) -> list:
        """The candidates of REVERSIBLE kinds that break the cut from the previous shot (cut_cost > 0)
        or, before a flight out, end moving backward - played in reverse."""
        import dataclasses
        out = []
        for m in cands:
            if m.kind not in REVERSIBLE or m.why.get("reversed"):
                continue
            breaks = self.__dict__.get("_last_exit") is not None and self.cut_cost(m) > 0
            if not (breaks or (then_forward and breaks_into(m, then_forward))):
                continue
            out.append(dataclasses.replace(m, eye=m.eye[::-1].copy(), tgt=m.tgt[::-1].copy(),
                                           z=None if m.z is None else m.z[::-1].copy(), occ=None,
                                           why={k: v for k, v in m.why.items() if k not in ("cut", "front_m", "own_share",
                                                                                          "turn_dps", "move_mps")}
                                           | {"reversed": True}))
        return out

    def sequence_cost(self, m: Move) -> float:
        """Score lost to the tour so far (2D: cinematic.py's beam search): the previous kind again,
        two shots back, the same side as the previous shot (sides alternate), each earlier use of
        the kind the same way, a kind past its second use, and moving in / out past a third of the
        tour or twice running."""
        hist = self.__dict__.get("_history", [])
        if not hist:
            return 0.0
        sd = self.side(m)
        cost = 0.0
        if hist[-1][0] == m.kind:
            cost += SAME_KIND_W
        if len(hist) >= 2 and hist[-2][0] == m.kind:
            cost += TWO_BACK_W
        if self.continuous:                       # one continuous move: direction kept across the cut
            cost += self.cut_cost(m)
        elif sd and hist[-1][1] == sd:
            cost += SAME_SIDE_W
        cost += USED_DIR_W * sum(1 for k, d in hist if k == m.kind and d == sd)
        cost += OVERUSE_W * max(0, sum(1 for k, _ in hist if k == m.kind) - 1)
        if m.kind not in FLIGHT:                  # the library mixes: caps per kind and share
            uses = sum(1 for k, _ in hist if k == m.kind)
            n_tour = max(getattr(self, "tour_shots", 9), 1)
            if uses >= KIND_CAP.get(m.kind, 2) or uses + 1 > max(1, math.ceil(KIND_SHARE * n_tour)):
                cost += CAP_W
        if m.kind in DOLLY_KINDS:
            cap = math.ceil(max(getattr(self, "tour_shots", 9), 1) / 3)
            if sum(1 for k, _ in hist if k in DOLLY_KINDS) >= cap:
                cost += DOLLY_CAP_W
            if hist[-1][0] in DOLLY_KINDS:
                cost += DOLLY_RUN_W
        m.why["sequence_cost"] = round(cost, 2)
        return cost

    def at_cap(self, kind: str, further: bool = False, own_piece: bool = False) -> bool:
        """The tour has used the kind as often as it may. further: for a room's further angle, the
        kind's last use is kept for a room's first shot (green house: the dining room's second
        angle took the tour's second rise, and the study - shown by a rise - got a glide)."""
        if kind in FLIGHT:
            return False
        hist = self.__dict__.get("_history", [])
        # (a kind the tour has once at most is not held back: it would never be a further angle)
        uses = sum(1 for k, _ in hist if k == kind) + (1 if further and KIND_CAP.get(kind, 2) >= 2 else 0)
        n_tour = max(getattr(self, "tour_shots", 9), 1)
        if own_piece:
            # a room's first shot walking onto what the room is about (its sofa, table, bed): each
            # frames a different subject - only the hard cap holds (green house: the dining room's
            # table walk took the tour's share of walks, and the primary bedroom lost its bed walk)
            return uses >= KIND_CAP.get(kind, 2)
        return uses >= KIND_CAP.get(kind, 2) or uses + 1 > max(1, math.ceil(KIND_SHARE * n_tour))

    @staticmethod
    def sinks(m: Move) -> bool:
        """The view drops past what it shows (2D: never sink below the interest): its aim ends
        SINK_M lower than it starts while the subject's top is above the end aim. A rise looks down
        on purpose; a push may tilt onto its counter."""
        if m.kind in ("rise", "descend", "push", "pull", "detail", "link") or m.subject is None:
            return False
        return bool(m.tgt[-1][2] < m.tgt[0][2] - SINK_M and float(m.subject[:, 2].max()) > m.tgt[-1][2])

    def profile(self, room: str) -> dict:
        """What the room is like, and the kinds of move that suits (prior per kind, 0..1)."""
        cache = self.__dict__.setdefault("_profiles", {})
        if room in cache:
            return cache[room]
        sp = self.space
        area = float(sp.room_area(room))
        open_plan = len(sp.spaces.get(room, {room})) > 1
        pieces = [p.cls for p in sp.main_pieces(room)]
        subjects = self._subjects(room)
        prior = {}
        if open_plan or area >= RISE_AREA:
            prior["rise"] = 1.0                             # space to open up from above
        # the photo tours' scene fit (planner/cinematic.py): a push only when one subject dominates
        # the room's features; a truck / pan when they spread wide; a bathroom pulls back
        dom, spread = self._dominance_spread(room)
        prior["push"] = float(np.clip(2.0 * (dom - 0.3), -0.5, 1.0))
        if self.anchor(room) is not None or spread >= 0.5:
            prior["truck"] = max(1.0 if self.anchor(room) is not None else 0.0, min(1.0, spread))
        prior["room_pan"] = -0.5 + 0.5 * min(1.0, spread)  # a pan only where nothing else will do
        if any(c in ("table", "kitchen island", "bed", "coffee table", "pool table") for c in pieces):
            prior["arc"] = 1.0                              # a piece with floor round it: circle it
        prior["corner_sweep"] = 0.6                         # shows two or three walls in one move
        from tour_worker.scoring.rooms import canon
        if "bath" in (canon(sp.room_types.get(room)) or "") or area < 8.0:
            prior["backout"] = 1.0                          # a small room: back out of it
            if any(k in (canon(sp.room_types.get(room)) or "") for k in DOOR_PAN_ROOMS):
                prior["door_pan"] = 0.8                     # ... or look in from its door
        if area < 14.0 or spread >= 0.6:
            prior["panorama"] = 0.5                         # a small room, or things all round it
        if (open_plan or area >= RISE_AREA) and any(c in ("table", "bed", "kitchen island", "sofa") for c in pieces):
            prior["descend"] = 0.8                          # come down onto the table / bed
        from tour_worker.scoring.rooms import canon
        if canon(sp.room_types.get(room)) in ("bathroom", "primary_bathroom", "ensuite", "powder_room"):
            prior["pull"], prior["push"] = 1.0, min(prior["push"], 0.0)
        elif area < 12.0:
            prior["pull"] = 0.5                             # small: move out, not across
        cache[room] = {"area": round(area, 1), "open_plan": open_plan, "pieces": pieces,
                       "subjects": len(subjects), "dominance": round(dom, 2), "spread": round(spread, 2),
                       "prior": prior}
        return cache[room]

    def _dominance_spread(self, room: str) -> tuple[float, float]:
        """(the largest labelled feature's share of the room's feature weight x size; how widely
        the features spread - their extent across the room's longer side, 0..1). The photo tours
        read these off the photo (scene_context.py dominance / h_extent); here off the labels."""
        sp = self.space
        objs = [o for o in sp.room_objects(room, FEATURES)]
        if not objs:
            return 0.0, 0.0
        w = np.array([FEATURES[o["class"]] * max(0.05, float(np.prod(np.asarray(o["max"][:2]) - np.asarray(o["min"][:2]))))
                      for o in objs])
        dom = float(w.max() / w.sum())
        c = np.array([o["centre"][:2] for o in objs], float)
        mask = sp.room_of == sp.room_keys.index(room)
        ys, xs = np.nonzero(mask)
        side = max(np.ptp(xs), np.ptp(ys), 1) * 0.05
        ext = float(max(np.ptp(c[:, 0]), np.ptp(c[:, 1]))) if len(c) > 1 else 0.0
        return dom, min(1.0, ext / side)

    def _truck(self, room, rmin, dur, prev, earlier) -> list[Move]:
        """A truck: the camera slides sideways past one of the room's subjects (a counter run, a
        bed, shelves), TRUCK_DIST m out, its aim sliding TRUCK_AIM of the way with it - parallax
        across the whole frame, the subject held."""
        sp = self.space
        subs = list(self._subjects(room))
        a = self.anchor(room)
        if a is not None:
            subs.append((np.array([a[0], a[1], min(a[2] + 0.15, AIM_Z_MAX)]),
                         self.__dict__.get("_anchor_box", {}).get(room), "working surface"))
        mine = sp.spaces.get(room, {room})
        u = np.linspace(0, 1, SAMPLES)
        w = (u * u * (3 - 2 * u))[:, None]
        out = []
        for ta, box, name in subs:
            for dist in TRUCK_DIST:
                for a0 in np.radians(np.arange(0, 360, 20)):
                    n = np.array([math.cos(a0), math.sin(a0)])
                    c = ta[:2] + dist * n
                    if sp.clearance(c) < rmin or sp.room_at(c) not in mine:
                        continue
                    if sp.ray(c, math.atan2(-n[1], -n[0]), max_m=dist + 0.3) < dist - 0.6:
                        continue                              # it is behind a wall from here
                    side = np.array([-n[1], n[0]])
                    for L in TRUCK_LEN:
                        for sg in (1, -1):
                            p0, p1 = c - sg * side * L / 2, c + sg * side * L / 2
                            eye = (1 - w) * p0 + w * p1
                            if any(sp.clearance(q) < rmin or sp.room_at(q) not in mine for q in eye[::7]):
                                continue
                            shift = (w - 0.5) * L * TRUCK_AIM * sg
                            tgt = np.c_[ta[0] + shift[:, 0] * side[0], ta[1] + shift[:, 0] * side[1], np.full(SAMPLES, ta[2])]
                            tgt = cap_pitch(eye, tgt, PITCH_LADDER[0])
                            first, last = (eye[0], tgt[0]), (eye[-1], tgt[-1])
                            if jump_cut(prev, first) or repeats(first, last, earlier):
                                continue
                            score = 1.2 + 0.4 * L - 0.3 * abs(dist - 2.8)
                            out.append(Move("truck", eye, tgt, score, subject=box,
                                            why={"of": name, "slide_m": L, "out_m": dist}))
        return _keep_best(out)

    def _rise(self, room, pts, rmin, dur, prev, earlier) -> list[Move]:
        """A rise: in an open or large room, from an open spot the camera climbs from eye height to
        RISE_Z (under the ceiling), drifting toward the room's main things and tilting down to
        RISE_PITCH - the space opens up below, as from a drone."""
        sp = self.space
        prof = self.profile(room)
        if "rise" not in prof["prior"]:
            return []
        top = RISE_Z
        if "scene_ceiling_missing" not in getattr(sp, "warnings", []):
            top = min(RISE_Z, float(sp.room_height) - RISE_HEADROOM)
        if top < EYE + 0.35:
            return []
        subs = self._subjects(room)
        if not subs:
            g = self._goal(room)
            subs = [(g, None, "floor")]
        u = np.linspace(0, 1, SAMPLES)
        w = u * u * (3 - 2 * u)
        z = EYE + (top - EYE) * w
        out = []
        for p in sorted(pts, key=lambda q: -sp.clearance(q))[:10]:
            for ta, box, name in subs:
                v = ta[:2] - p
                d = float(np.linalg.norm(v))
                if not RISE_DIST[0] <= d <= RISE_DIST[1]:
                    continue
                if sp.ray(p, math.atan2(v[1], v[0]), max_m=d + 0.3) < d - 0.6:
                    continue
                for drift in RISE_DRIFT:
                    p1 = p + drift * v / d
                    if sp.clearance(p1) < rmin or sp.room_at(p1) not in sp.spaces.get(room, {room}):
                        continue
                    eye = np.outer(1 - w, p) + np.outer(w, p1)
                    h = np.linalg.norm(ta[:2] - eye, axis=1)
                    # level-ish at the start (the usual look), RISE_PITCH down at the top
                    pitch = np.radians(PITCH_LADDER[0] * 0.6 + (RISE_PITCH - PITCH_LADDER[0] * 0.6) * w)
                    look = np.minimum(np.tan(pitch), (z - 0.2) / np.maximum(h, 0.1))
                    tgt = np.c_[np.repeat(ta[None, :2], SAMPLES, 0), z - h * look]
                    first, last = (eye[0], tgt[0]), (eye[-1], tgt[-1])
                    if jump_cut(prev, first) or repeats(first, last, earlier):
                        continue
                    depth = sp.view_depth(p, math.atan2(v[1], v[0]), HFOV)
                    score = 1.1 + 0.1 * min(depth, 6.0) + 0.2 * drift
                    out.append(Move("rise", eye, tgt, score, subject=box, z=z,
                                    why={"to_m": round(top, 2), "of": name, "drift_m": drift, "pitch_end": RISE_PITCH}))
        return _keep_best(out, 20)

    def exit(self, room: str, nxt: str, route, dur: float, prev=None, earlier=()) -> tuple[Move | None, list[str]]:
        """The room's last shot heads for the next room: along the walking route to EXIT_STOP m before
        where the room ends (its doorway, or the open-plan neighbour's floor), looking ahead where it
        is going - the next shot picks the flight up inside the next room."""
        sp = self.space
        self.turn_budget, self.move_budget = MAX_TURN_DPS * PRE_SYNC_SHARE["room"], MAX_MOVE_MPS * PRE_SYNC_SHARE["room"]
        poly, _ = route
        mine = sp.spaces.get(room, {room}) if nxt not in sp.spaces.get(room, {room}) else {room}
        at = [sp.room_at(q) for q in poly]
        k_out = next((k for k, r in enumerate(at) if k > 0 and r not in mine), None)
        if k_out is None:
            return None, [f"exit_no_route:{room}"]
        s_ = self._along(poly)
        ends = [k for k in range(k_out) if EXIT_STOP <= s_[k_out] - s_[k] <= EXIT_STOP + 0.6]
        cands = []
        for rmin in RMIN_LADDER:
            for ke in ends[::2]:
                if sp.clearance(poly[ke]) < rmin or self.in_doorway(poly[ke]):
                    continue
                for ks in range(ke):
                    L = s_[ke] - s_[ks]
                    if not EXIT_LEN[0] <= L <= EXIT_LEN[1] or (ks % 3) or at[ks] not in mine:
                        continue
                    if L / dur * self.peak(2.1) > self.move_budget:   # eased in: its peak ~2x its mean speed
                        continue
                    seg = poly[ks:ke + 1]
                    if min(sp.clearance(q) for q in seg) < rmin * 0.9:
                        continue
                    path = resample(smooth(sp, seg, rmin), SAMPLES)
                    sp_ = self._along(path)
                    # look ahead along the route, EXIT_AHEAD m on (into the next room at the end)
                    s0 = s_[ks]
                    look = [np.interp(min(s0 + d + EXIT_AHEAD, s_[-1]), s_, poly[:, j]) for d in sp_ for j in (0, 1)]
                    look = np.array(look).reshape(-1, 2)
                    tgt = np.c_[look, np.full(SAMPLES, EYE - LOOK_DROP)]
                    first, last = (path[0], tgt[0]), (path[-1], tgt[-1])
                    if jump_cut(prev, first) or repeats(first, last, earlier):
                        continue
                    score = 1.0 + 0.3 * L + 0.1 * min(sp.view_depth(path[-1], math.atan2(*(tgt[-1][:2] - path[-1])[::-1]), HFOV), 6.0)
                    cands.append(Move("exit", path, tgt, score, easing="ease_in", why={"toward": nxt, "travel_m": round(L, 2),
                                                                      "stops_before_m": round(float(s_[k_out] - s_[ke]), 2)}))
            if cands:
                break
        cands = [m for m in cands if self.moves_enough(m, dur) and sp.clearance(m.eye[0]) >= WALL_CLEAR]
        if not cands:
            return None, [f"exit_no_move:{room}"]
        cands = self.keep_direction(cands)
        best, least = self._check(room, cands, dur)
        if best is None:
            # why (2026-11-17: every exit "blocked", no reason): too fast, clutter at the lens, the
            # frame's edge, the scan's black, the band under a missing ceiling
            if least is None:
                why = "speed"
            else:
                why = (f"near{least.why['near_lens']}" if "near_lens" in least.why
                       else f"edge{least.why['edge_cover']}" if "edge_cover" in least.why
                       else f"band{least.why['band_deficit_deg']}" if "band_deficit_deg" in least.why
                       else f"void{least.why['void']}" if least.why.get("void", 0) > VOID_MAX
                       else f"occ{round(least.occ, 2)}")
            return None, [f"exit_blocked:{room}:{why}:{len(cands)}"]
        best.why.update(candidates=len(cands), occlusion=round(best.occ, 3))
        self.record(best)
        return best, []

    def crossings(self, route) -> list[dict]:
        """Where the walking route passes from one room into the next: {k (first route point past
        it), s (m along the route), from, to, width_m, open (one open-plan space), narrow (a cut
        there: not open and narrower than THROUGH_MIN_W)}. The width is twice the clearance to the
        walls where the route crosses (it keeps to a doorway's middle)."""
        sp = self.space
        poly, _ = route
        s_ = self._along(poly)
        at = [sp.room_at(q) for q in poly]
        rdt = getattr(sp, "room_dt", sp.dt)
        out = []
        for k in range(1, len(poly)):
            if at[k] is None or at[k - 1] is None or at[k] == at[k - 1]:
                continue
            r = 0.0
            for q, s in zip(poly, s_):
                if abs(s - s_[k]) <= 0.25:
                    i, j = sp.cell(q)
                    if 0 <= i < rdt.shape[0] and 0 <= j < rdt.shape[1]:
                        r = max(r, float(rdt[i, j]))
            open_ = at[k] in sp.spaces.get(at[k - 1], {at[k - 1]})
            out.append({"k": k, "s": float(s_[k]), "from": at[k - 1], "to": at[k], "width_m": round(2 * r, 2),
                        "open": open_, "narrow": not open_ and 2 * r < THROUGH_MIN_W})
        return out

    def flight(self, kind: str, room: str, route, start: tuple[float, float], end: tuple[float, float],
               pace: float, extra_s: float = 0.0, prev=None, earlier=(), aim=None, piece=None,
               end_room: str | None = None, matched: bool = False,
               look_max_s: float | None = None, geometry_only: bool = False,
               start_room: str | None = None, look_fixed_s: float | None = None, min_s: float = 0.0) -> tuple[Move | None, list[str]]:
        """One leg of a room change along the walking route: it starts between start[0] and start[1]
        m along the route and ends between end[0] and end[1] m, at `pace` m/s (FLIGHT_S long, never
        under 3 s), looking ahead along the route - and, given `aim` (a through flight into the next
        room), turning onto it over its second half. matched: it picks up a cut at a door, so it
        starts on the route's heading (no jump-cut test: the door is the cut). end_room: where it
        must end; start_room: where it must start (a room change's first leg starts in the room being
        left - never across the house from it, user 2026-11-26). extra_s: the cut handle its path is planned longer by. look_max_s: the look ahead
        stops there (an exit / a walk keeps looking at the door it heads for, not round the bend past
        it); a through flight turns onto `aim` only as far as the turn limit allows. geometry_only:
        a dry run (no render checks, nothing recorded) - its best candidate by score. Returns the move
        (why.dur_s: its length) or None with why."""
        sp = self.space
        self.turn_budget, self.move_budget = MAX_TURN_DPS * PRE_SYNC_SHARE["room"], MAX_MOVE_MPS * PRE_SYNC_SHARE["room"]
        poly, _ = route
        s_ = self._along(poly)
        at = [sp.room_at(q) for q in poly]
        pace = float(np.clip(pace, *FLIGHT_PACE))
        through = aim is not None
        if through:
            pace = THROUGH_PACE
        u = np.linspace(0, 1, SAMPLES)
        fp = (piece.frame or piece) if piece is not None else None
        cands = []
        st = dict.fromkeys(("starts", "ends", "short", "speed", "clear", "cut", "heading", "still"), 0)
        for rmin in RMIN_LADDER:
            ok = [sp.clearance(q) >= rmin and not self.in_doorway(q) for q in poly]
            ks_ = [k for k in range(len(poly)) if start[0] <= s_[k] <= start[1] and ok[k]
                   and (start_room is None or at[k] == start_room)]
            ke_ = [k for k in range(len(poly)) if end[0] <= s_[k] <= end[1] and ok[k]
                   and (end_room is None or at[k] == end_room)]
            st["starts"], st["ends"] = max(st["starts"], len(ks_)), max(st["ends"], len(ke_))
            # at most FLIGHT_TRIES starts x ends, spread along the route (the Home Scan's long open
            # routes gave hundreds of pairs a leg: planning went from 6 to 19 minutes)
            pick = lambda ks: [ks[int(i)] for i in np.unique(np.linspace(0, len(ks) - 1, min(len(ks), FLIGHT_TRIES)).round())] if ks else []   # noqa: E731
            found = False
            for ks in pick(ks_):
                if found:
                    break
                for ke in pick(ke_):
                    L = s_[ke] - s_[ks]
                    if L < 0.6:
                        st["short"] += 1
                        continue
                    dur = float(np.clip(L / pace, FLIGHT_S[0], THROUGH_S_MAX if through else FLIGHT_S[1]))
                    if through:                    # it is the room's opening shot: its time (min_s)
                        dur = max(dur, min_s)
                    if L / (dur + extra_s) * self.peak(2.1) > self.move_budget:
                        st["speed"] += 1
                        continue
                    seg = poly[ks:ke + 1]
                    # (a flight through a door passes its frame closer than a room shot may come to a
                    # wall: in the doorway DOOR_PASS_CLEAR is enough - the lens check decides the rest)
                    need = [DOOR_PASS_CLEAR if kind == "through" and self.in_doorway(q) else rmin * 0.9 for q in seg]
                    if any(sp.clearance(q) < n for q, n in zip(seg, need)):
                        st["clear"] += 1
                        continue
                    path = resample(smooth(sp, seg, rmin), SAMPLES)
                    d = self._along(path)
                    s_look = s_[-1] if look_max_s is None else min(s_[-1], look_max_s)
                    if look_fixed_s is not None:
                        # one point to look at, beyond where it is going: the view turns as the eye
                        # moves, slowly - never swinging round a bend in the route
                        look = np.repeat(_route_point(poly, s_, look_fixed_s)[None], len(d), 0)
                    else:
                        look = np.array([_route_point(poly, s_, min(s_[ks] + x + EXIT_AHEAD, s_look)) for x in d])
                    close = np.linalg.norm(look - path, axis=1) < 1.0     # the route ends at the eye: straight on
                    if close.any() and look_max_s is None and look_fixed_s is None:
                        look[close] = [_route_point(poly, s_, s_[ks] + x + EXIT_AHEAD) for x in d[close]]
                    tgt = np.c_[look, np.full(SAMPLES, EYE - LOOK_DROP)]
                    if float(np.min(np.linalg.norm(look - path, axis=1))) < 0.8:   # the route ends at the eye
                        st["short"] += 1
                        continue
                    core = 0.0
                    turn = move = None
                    if aim is not None:
                        v = np.clip((u - THROUGH_TURN_FROM) / (1 - THROUGH_TURN_FROM), 0, 1)
                        w = (v * v * (3 - 2 * v))[:, None]
                        ahead_t = tgt
                        for wmax in (1.0, 0.7, 0.45, 0.25, 0.0):     # as far onto the room as the turn limit allows
                            tgt = cap_pitch(path, (1 - wmax * w) * ahead_t + wmax * w * np.asarray(aim, float)[None], PITCH_LADDER[0])
                            if float(np.min(np.linalg.norm(tgt[:, :2] - path, axis=1))) < 0.5:
                                turn = None
                                break
                            turn, move = trial_speeds(path, tgt, dur + extra_s, EYE, "linear" if self.continuous else "ease_in_out")
                            if turn <= min(self.turn_budget, THROUGH_TURN_DPS):
                                break
                        if turn is None:
                            st["short"] += 1
                            continue
                        if fp is not None:
                            core = next((c for c in CORES if in_frame(path[-1], tgt[-1], fp, c)), 0.0)
                    first, last = (path[0], tgt[0]), (path[-1], tgt[-1])
                    if (not matched and jump_cut(prev, first)) or repeats(first, last, earlier):
                        st["cut"] += 1
                        continue
                    if matched:                     # it keeps the heading through the door it follows
                        h = path[min(8, SAMPLES - 1)] - path[0]
                        ahead = look[0] - path[0]
                        c = float(h @ ahead) / max(float(np.linalg.norm(h) * np.linalg.norm(ahead)), 1e-9)
                        if c < math.cos(math.radians(MATCH_TURN)):
                            st["heading"] += 1
                            continue
                    # the previous shot's pace kept (item 4), then a deep view ahead, then the room framed
                    score = 1.0 - abs(L / dur - pace) / pace + 0.1 * min(
                        sp.view_depth(path[-1], math.atan2(*(tgt[-1][:2] - path[-1])[::-1]), HFOV), 6.0) + 1.0 * core
                    if through:                      # its first and last views not walled up beside the lens
                        score -= NEAR_WALL_W * self.wall_share(path, tgt)
                    if through:                    # the room's opening shot: never looking up (AIM_Z_MAX)
                        tgt = tgt.copy()
                        tgt[:, 2] = np.minimum(tgt[:, 2], AIM_Z_MAX)
                    m = Move(kind, path, tgt, score, piece=piece if aim is not None else None,
                             subject=piece_box(fp) if (aim is not None and fp is not None) else None,
                             easing="linear", why={"travel_m": round(L, 2), "dur_s": round(dur, 2),
                                                   "matched": matched, "core_end": core})
                    if turn is None:
                        turn, move = trial_speeds(path, tgt, dur + extra_s, EYE, "linear" if self.continuous else "ease_in_out")
                    if turn > self.turn_budget or move > self.move_budget:
                        st["speed"] += 1
                        continue
                    m.why.update(turn_dps=round(turn, 1), move_mps=round(move, 2))
                    if self.moves_enough(m, dur):
                        cands.append(m)
                        if geometry_only:            # a dry run only asks whether one exists
                            found = True
                            break
                    else:
                        st["still"] += 1
            if cands:
                break
        if not cands:
            return None, [f"{kind}_no_move:{room}:" + "/".join(f"{k}{v}" for k, v in st.items() if v)]
        for m in cands:                         # at the 3D pace, into the cut before (to_pace) ...
            least = PACE_MIN_KEEP
            if through and end_room is not None:  # ... never past DOOR_LEAD before its room's door
                at_ = [sp.room_at(q) for q in m.eye]
                k_ = next((j for j in range(len(at_)) if at_[j] == end_room), None)
                arc_ = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(m.eye, axis=0), axis=1))]
                if k_ and arc_[-1] > 1e-6:
                    least = max(least, 1.0 - max(0.0, arc_[k_] - DOOR_LEAD) / arc_[-1])
            self.to_pace(m, m.why["dur_s"] + extra_s, least=least)
        if geometry_only:
            return max(cands, key=lambda m: m.score), []
        # _check's speed test takes one length: the longest leg's (each passed its own above); the
        # flight's own render budget
        self._budget = {"kinds": {}, "left": FLIGHT_CHECKS}
        self._late_ok = True
        try:
            best, least = self._check(room, cands, max(m.why["dur_s"] for m in cands) + extra_s)
        finally:
            self._late_ok = False
        if best is None:
            why = "speed" if least is None else (f"near{least.why['near_lens']}" if "near_lens" in least.why
                                                 else f"edge{least.why['edge_cover']}" if "edge_cover" in least.why
                                                 else f"void{least.why['void']}" if least.why.get("void", 0) > VOID_MAX
                                                 else f"occ{round(least.occ, 2)}")
            return None, [f"{kind}_blocked:{room}:{why}:{len(cands)}"]
        best.why.update(candidates=len(cands), occlusion=round(best.occ, 3))
        self.record(best)
        return best, []

    def _enter_after_exit(self, room, piece, route, rmin, dur, earlier) -> list[Move]:
        sp = self.space
        poly, _ = route
        at = [sp.room_at(q) for q in poly]
        d_in = next((k for k, r in enumerate(at) if r == room), None)
        if d_in is None or d_in == 0:
            return []
        s_ = self._along(poly)
        fp = (piece.frame or piece) if piece is not None else None
        aim = np.array([*fp.centre, fp.top]) if fp is not None else self._goal(room)
        inside = [k for k in range(d_in, len(poly)) if AFTER_EXIT[0] <= s_[k] - s_[d_in] <= AFTER_EXIT[1]
                  and at[k] == room and sp.clearance(poly[k]) >= rmin and not self.in_doorway(poly[k])]
        ends = [q for q in sp.spread_points(sp.space_mask(room, rmin), 36)
                if sp.room_at(q) == room and ENTER_AFTER[0] + 1.0 <= np.linalg.norm(q - poly[d_in]) <= ENTER_AFTER[1] + 1.2]
        if not inside or not ends:
            return []
        mask = (sp.dt >= rmin) & (sp.room_of >= 0)
        starts = inside[::2]
        pts = np.array([poly[k] for k in starts] + list(ends))
        paths = sp.paths_between(pts, mask)
        u = np.linspace(0, 1, SAMPLES)
        w = (u * u * (3 - 2 * u))[:, None]
        out = []
        for a, ks in enumerate(starts):
            ahead = np.array([np.interp(min(s_[ks] + EXIT_AHEAD, s_[-1]), s_, poly[:, j]) for j in (0, 1)])
            t0 = np.array([*ahead, EYE - LOOK_DROP])
            for b in range(len(starts), len(pts)):
                path, plen = self._path(paths, a, b, rmin)
                L = float(np.linalg.norm(pts[b] - pts[a]))
                if path is None or plen > 1.4 * L or plen < 0.8 or plen / dur * self.peak(1.5) > self.move_budget:
                    continue
                path = resample(path, SAMPLES)
                h0 = path[min(8, SAMPLES - 1)] - path[0]
                a0 = ahead - poly[ks]                            # it keeps flying the same way
                if float(h0 @ a0) < math.cos(math.radians(MATCH_TURN)) * float(np.linalg.norm(h0) * np.linalg.norm(a0)):
                    continue
                tgt = cap_pitch(path, (1 - w) * t0 + w * aim, PITCH_LADDER[0])
                if repeats((path[0], tgt[0]), (path[-1], tgt[-1]), earlier):
                    continue
                core = next((c for c in CORES if fp is not None and in_frame(path[-1], tgt[-1], fp, c)), 0.0)
                score = 1.5 + 0.3 * plen + 1.0 * core
                out.append(Move("enter", path, tgt, score, piece=piece, subject=piece_box(fp) if fp is not None else None,
                                easing="ease_out",
                                why={"travel_m": round(plen, 2), "after_exit": True, "core_end": core}))
        return out

    def _room_centre(self, room: str) -> np.ndarray:
        sp = self.space
        ys, xs = np.nonzero(sp.room_of == sp.room_keys.index(room))
        return np.array([sp.lo[0] + (xs.mean() + .5) * 0.05, sp.lo[1] + (ys.mean() + .5) * 0.05])

    def _look(self, eye: np.ndarray, heads: np.ndarray) -> np.ndarray:
        """Targets for headings from eye (one per sample): as far as the room goes (2-6 m), a level look."""
        sp = self.space
        out = np.zeros((len(eye), 3))
        for k, (e, h) in enumerate(zip(eye, heads)):
            r = float(np.clip(sp.ray(e, h, max_m=6.0), 2.0, 6.0))
            out[k] = (e[0] + r * math.cos(h), e[1] + r * math.sin(h), EYE - LOOK_DROP)
        return out

    def _room_subject(self, room: str):
        """(point, box) a shot of the room turns onto: its bed / sofa whole (TYPE_WHOLE), else its
        main piece, else its middle (no box)."""
        boxes = self.whole_boxes(room)
        if boxes:
            b = max(boxes, key=lambda b: float(np.ptp(b[:, 0]) * np.ptp(b[:, 1])))
            return b[:, :2].mean(0), b
        pieces = self.space.main_pieces(room)
        if pieces:
            fp = pieces[0].frame or pieces[0]
            return np.asarray(fp.centre, float), piece_box(fp)
        return self._room_centre(room), None

    def _room_door(self, room: str):
        """Where the room is left for the rest of the home: on the walking route to its nearest
        other room, the last point inside it."""
        cache = self.__dict__.setdefault("_doors_of", {})
        if room in cache:
            return cache[room]
        sp = self.space
        best = None
        for other in sp.room_keys:
            if other == room:
                continue
            r = sp.route(room, other)
            if not r:
                continue
            poly = r[0]
            L = float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum())
            at = [sp.room_at(q) for q in poly]
            k = next((n for n in range(1, len(at)) if at[n] != room), None)
            if k is None:
                continue
            if best is None or L < best[0]:
                best = (L, poly[k - 1])
        cache[room] = None if best is None else np.asarray(best[1], float)
        return cache[room]

    def _backout(self, room, pts, rmin, dur, prev, earlier) -> list[Move]:
        """Back out of a small room: from BACKOUT_IN in from its deepest point (farthest from its
        door), looking into the room, the camera backs out toward the door - the room opening up as
        it goes (bathrooms, small bedrooms, a study). It may end in the doorway."""
        sp = self.space
        from tour_worker.scoring.rooms import canon
        foot = float((sp.room_fill == sp.room_keys.index(room)).sum()) * 0.05 * 0.05 if room in sp.room_keys else 0.0
        kind_ = canon(sp.room_types.get(room)) or ""
        if foot > BACKOUT_SMALL_M2 and "bath" not in kind_ and "bedroom" not in kind_:
            return []
        door = self._room_door(room)
        if door is None:
            return []
        # the deepest points: any free floor of the room with a body's clearance (a bathroom's floor
        # is a strip between its fittings - the grid of standing points had none, green house v13)
        ys, xs = np.nonzero((sp.room_of == sp.room_keys.index(room)) & (sp.dt >= BACKOUT_CLEAR))
        if not len(ys):
            return []
        cells = np.c_[sp.lo[0] + (xs + .5) * 0.05, sp.lo[1] + (ys + .5) * 0.05]
        far = cells[np.argsort(-np.linalg.norm(cells - door, axis=1))[:200:50]]
        u = np.linspace(0, 1, SAMPLES)
        w = (u * u * (3 - 2 * u))[:, None]
        out = []
        for deep in far:
            v = deep - door
            L = float(np.linalg.norm(v))
            if L < 0.5 + 0.3:
                continue
            into = v / L
            p0 = deep - min(BACKOUT_IN, L - 0.6) * into     # a step in from the deepest point
            p1 = door + 0.15 * into                  # just inside the door
            if float(np.linalg.norm(p1 - p0)) < 0.5:
                continue
            # out along the floor the room has (round its fittings - a straight line ran through them)
            ends = np.array([sp.snap(p0, BACKOUT_CLEAR, 0.4) if sp.snap(p0, BACKOUT_CLEAR, 0.4) is not None else p0,
                             sp.snap(p1, BACKOUT_CLEAR, 0.4) if sp.snap(p1, BACKOUT_CLEAR, 0.4) is not None else p1])
            route = sp.paths_between(ends, sp.dt >= BACKOUT_CLEAR)(0, 1)
            if route is None or len(route) < 2:
                continue
            path = resample(smooth(sp, np.asarray(route, float), BACKOUT_CLEAR), SAMPLES)
            plen = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
            if plen > 1.6 * float(np.linalg.norm(ends[1] - ends[0])) + 0.3:
                continue                               # not a back-out, a tour of the fittings
            eye = path
            for swing in (0.0, 0.6, -0.6):           # looking at the deep end all the way out (a little to one side)
                aim = deep + swing * np.array([-into[1], into[0]])
                d = aim - eye[-1]                        # one direction all the way: from the door end
                d = d / max(float(np.linalg.norm(d)), 1e-6)       # toward the deep end (no turning)
                # it stops where the way out still has room in front (a wardrobe or a wall jog by the
                # door - the green house's primary bedroom ended 0.25 m from one)
                # (and where it turns onto the room's subject, that view is open too - the green
                # house's bedroom door sits in a corner, a wall stub 0.2-0.5 m off toward the bed)
                subj_pt, subj_box = self._room_subject(room)
                ok_k = [k for k in range(len(eye)) if self.front(eye[k], eye[k] + 3.0 * d) >= BACKOUT_FRONT[1]
                        and self.front(eye[k], np.array([*subj_pt, EYE - 0.25])) >= BACKOUT_FRONT[1]]
                if not ok_k:
                    continue
                kk = ok_k[-1]
                e_ = resample(eye[:kk + 1], SAMPLES) if kk >= 2 else None
                if e_ is None or float(np.linalg.norm(np.diff(e_, axis=0), axis=1).sum()) < 0.5:
                    continue
                # ... and on the way the view turns from the deep end onto the room's subject - its bed
                # (whole) or its middle (user 2026-11-17: "the camera hugged the left wall and went
                # back, when all it had to do was pan in between that motion to see the room")
                h0 = math.atan2(d[1], d[0])
                h1 = math.atan2(subj_pt[1] - e_[-1][1], subj_pt[0] - e_[-1][0])
                dh = (h1 - h0 + math.pi) % (2 * math.pi) - math.pi
                lim = math.radians(0.55 * self.turn_budget * dur)
                dh = float(np.clip(dh, -lim, lim))
                ww = np.linspace(0, 1, SAMPLES)
                ww = ww * ww * (3 - 2 * ww)
                heads = h0 + dh * ww
                tgt = np.c_[e_ + 3.0 * np.c_[np.cos(heads), np.sin(heads)], np.full(SAMPLES, EYE - 0.25)]
                first, last = (e_[0], tgt[0]), (e_[-1], tgt[-1])
                if jump_cut(prev, first) or repeats(first, last, earlier):
                    continue
                travel = float(np.linalg.norm(np.diff(e_, axis=0), axis=1).sum())
                mid = float(np.mean([sp.clearance(q) for q in e_[::5]]))       # the open middle, not a wall
                whole = subj_box is not None and frames_fully(e_[-1], tgt[-1], subj_box, EYE)
                piece = next(iter(self.space.main_pieces(room)), None)       # what it turns onto is no blockage
                out.append(Move("backout", e_, tgt, 1.2 + 0.3 * travel + 1.5 * mid + (1.5 if whole else 0.0), piece=piece,
                                why={"travel_m": round(travel, 2), "swing_deg": swing, "turn_onto_deg": round(math.degrees(dh)),
                                     "clear_m": round(mid, 2), "subject_whole": whole}))
        return out

    def _door_pan(self, room, dur, prev, earlier) -> list[Move]:
        """Look into a small room from its door (the rooms _backout films): standing DOOR_PAN_IN
        inside the door, the view pans across the room's far end, either way. It leaves the room
        moving sideways - the move for a room left by walking on forward (then_forward)."""
        sp = self.space
        from tour_worker.scoring.rooms import canon
        kind_ = canon(sp.room_types.get(room)) or ""
        if not any(k in kind_ for k in DOOR_PAN_ROOMS):     # (a bedroom or kitchen is walked into)
            return []
        door = self._room_door(room)
        if door is None:
            return []
        ys, xs = np.nonzero((sp.room_of == sp.room_keys.index(room)) & (sp.dt >= BACKOUT_CLEAR))
        if not len(ys):
            return []
        cells = np.c_[sp.lo[0] + (xs + .5) * 0.05, sp.lo[1] + (ys + .5) * 0.05]
        mid = cells.mean(0)
        into = mid - door
        if float(np.linalg.norm(into)) < 0.6:
            return []
        into = into / float(np.linalg.norm(into))
        u = np.linspace(0, 1, SAMPLES)
        w = u * u * (3 - 2 * u)
        sweep = math.radians(float(np.clip(0.6 * self.turn_budget * dur, *DOOR_PAN_SWEEP)))
        out = []
        # what it ends on: the fittings nearest its door (a room's floor can run on past them - the
        # green house bathroom's reached a counter 4 m off in the hall, and the pan aimed between
        # the two, at a shower screen, 2026-11-27)
        objs = sp.room_objects(room, DOOR_PAN_ON)
        fit = [np.asarray(o["centre"][:2], float) for o in objs]
        # (a fitting is solid: a clear line reaches its near face, half its footprint short of its centre)
        size = [0.5 * float(np.linalg.norm(np.asarray(o["max"][:2], float) - np.asarray(o["min"][:2], float)))
                if o.get("min") is not None and o.get("max") is not None else 0.3 for o in objs]
        on, near = None, []
        if fit:
            f0 = min(fit, key=lambda f: float(np.linalg.norm(f - door)))
            near = [(f, r) for f, r in zip(fit, size) if float(np.linalg.norm(f - f0)) <= DOOR_PAN_CLUSTER_M]
            on = np.mean([f for f, _ in near], axis=0)
        # where it stands: just in from the door, toward the room or toward its fittings - only
        # where the floor plan has a clear line to them (the deeper point stood behind the screen)
        spots = [door + d_in * into for d_in in DOOR_PAN_IN]
        if on is not None and float(np.linalg.norm(on - door)) > 0.6:
            toward = (on - door) / float(np.linalg.norm(on - door))
            spots += [door + d_in * toward for d_in in DOOR_PAN_IN]
        for d_in, spot in zip(list(DOOR_PAN_IN) * 2, spots):
            at = sp.snap(spot, BACKOUT_CLEAR, 0.3)
            if at is None or sp.room_at(at) != room:
                continue
            if near:                                  # at least one of them in clear sight
                seen_ = [float(sp.rays(at, np.array([math.atan2(f[1] - at[1], f[0] - at[0])]))[0])
                         >= float(np.linalg.norm(f - at)) - r - DOOR_PAN_SIGHT_TOL for f, r in near]
                if not any(seen_):
                    continue
            h_mid = math.atan2(*(mid - at)[::-1])
            # onto its fittings when it has any: the pan ends on them, from either side; else across
            # the middle of the room
            h_on = math.atan2(*(on - at)[::-1]) if on is not None else None
            for sgn in (1.0, -1.0):                  # across the room, either way
                heads = (h_on + sgn * sweep * (w - 1.0)) if h_on is not None else (h_mid + sgn * sweep * (w - 0.5))
                eye = np.repeat(np.asarray(at, float)[None], SAMPLES, 0)
                tgt = np.c_[eye + 3.0 * np.c_[np.cos(heads), np.sin(heads)], np.full(SAMPLES, EYE - 0.25)]
                first, last = (eye[0], tgt[0]), (eye[-1], tgt[-1])
                if jump_cut(prev, first) or repeats(first, last, earlier):
                    continue
                piece = next(iter(sp.main_pieces(room)), None)
                out.append(Move("door_pan", eye, tgt, 1.2 + 1.5 * float(sp.clearance(at)), piece=piece,
                                why={"travel_m": 0.0, "sweep_deg": round(math.degrees(sweep)), "in_m": d_in}))
        return out

    def _wall_views(self, room, pts, rmin, dur, prev, earlier, covered) -> list[Move]:
        """Coverage's own move (user 2026-11-17: "the shots are just one view of the room ... have
        we abandoned our coverage policies?" - every other move aims at the room's furniture, which
        its first shot framed): the biggest stretches of wall not seen yet (by direction from the
        room's middle); from across the room, with a clear line and room to breathe, a slow
        WALL_PAN deg pan across each, drifting 0.3 m toward it."""
        sp = self.space
        walls = sp.wall_cells(room)
        unseen = np.setdiff1d(walls, covered) if len(covered) else walls
        if len(unseen) < 20:
            return []
        ii, jj = np.unravel_index(unseen, sp.shape)
        xy = np.c_[sp.lo[0] + (jj + .5) * 0.05, sp.lo[1] + (ii + .5) * 0.05]
        c = self._room_centre(room)
        ang = np.arctan2(xy[:, 1] - c[1], xy[:, 0] - c[0])
        bins = ((ang + math.pi) / (2 * math.pi) * 12).astype(int) % 12
        counts = np.bincount(bins, minlength=12)
        front, aim_min = self.breathe(room)
        sweep = min(WALL_PAN, 0.55 * self.turn_budget * dur)    # as wide as the turn limit allows in the time
        if sweep < 15.0:
            return []
        u = np.linspace(0, 1, SAMPLES)
        w = u * u * (3 - 2 * u)
        out = []
        for b in np.argsort(-counts)[:3]:
            if counts[b] < 15:
                continue
            T = xy[bins == b].mean(0)
            for p in pts:
                v = T - p
                d = float(np.linalg.norm(v))
                if not aim_min + 0.3 <= d <= 6.0:
                    continue
                hc = math.atan2(v[1], v[0])
                if sp.ray(p, hc, max_m=d + 0.3) < d - 0.5:      # it sees that wall from here
                    continue
                p1 = p + 0.3 * v / d
                eye = np.outer(1 - w, p) + np.outer(w, p1)
                for side in (1, -1):
                    heads = hc + side * math.radians(sweep) * (w - 0.5)
                    tgt = self._look(eye, heads)
                    first, last = (eye[0], tgt[0]), (eye[-1], tgt[-1])
                    if jump_cut(prev, first) or repeats(first, last, earlier):
                        continue
                    out.append(Move("wall_pan", eye, tgt, 1.0 + 0.01 * counts[b],
                                    why={"onto_unseen_wall_cells": int(counts[b])}))
        return _keep_best(out, 24)

    def _corner_sweep(self, room, pts, rmin, dur, prev, earlier) -> list[Move]:
        """A corner sweep: from near a corner the camera dollies toward the room's middle while its
        view turns across it (SWEEP_TURN) - two or three walls in one move (coverage)."""
        sp = self.space
        c = self._room_centre(room)
        mine = sp.spaces.get(room, {room})
        u = np.linspace(0, 1, SAMPLES)
        w = u * u * (3 - 2 * u)
        out = []
        for p in sorted(pts, key=lambda q: -float(np.linalg.norm(q - c)))[:12]:
            to = c - p
            dist = float(np.linalg.norm(to))
            if dist < 2.0:
                continue
            hc = math.atan2(to[1], to[0])
            for L in SWEEP_LEN:
                if L > dist - 0.8:
                    continue
                p1 = p + L * to / dist
                eye = np.outer(1 - w, p) + np.outer(w, p1)
                if any(sp.clearance(q) < rmin or sp.room_at(q) not in mine for q in eye[::7]):
                    continue
                for turn in SWEEP_TURN:
                    for side in (1, -1):
                        heads = hc + side * math.radians(turn) * (0.5 - w)
                        tgt = self._look(eye, heads)
                        first, last = (eye[0], tgt[0]), (eye[-1], tgt[-1])
                        if jump_cut(prev, first) or repeats(first, last, earlier):
                            continue
                        score = 1.0 + 0.25 * L + 0.3 * turn / SWEEP_TURN[-1]
                        out.append(Move("corner_sweep", eye, tgt, score, why={"travel_m": L, "turn_deg": turn}))
        return _keep_best(out, 20)

    def _panorama(self, room, pts, rmin, dur, prev, earlier) -> list[Move]:
        """A panorama: from an open spot near the room's edge, a slow pan across the room
        (PANO_SWEEP, as wide as the turn limit allows in the time), drifting 0.2 m in."""
        sp = self.space
        c = self._room_centre(room)
        sweep = min(PANO_SWEEP[1], 0.6 * self.turn_budget * dur)
        if sweep < PANO_SWEEP[0]:
            return []
        u = np.linspace(0, 1, SAMPLES)
        w = u * u * (3 - 2 * u)
        out = []
        for p in sorted(pts, key=lambda q: -float(np.linalg.norm(q - c)))[:8]:
            to = c - p
            dist = float(np.linalg.norm(to))
            if dist < 1.5 or sp.clearance(p) < max(rmin, 0.45):
                continue
            hc = math.atan2(to[1], to[0])
            p1 = p + 0.2 * to / dist
            eye = np.outer(1 - w, p) + np.outer(w, p1)
            for side in (1, -1):
                heads = hc + side * math.radians(sweep) * (w - 0.5)
                tgt = self._look(eye, heads)
                first, last = (eye[0], tgt[0]), (eye[-1], tgt[-1])
                if jump_cut(prev, first) or repeats(first, last, earlier):
                    continue
                out.append(Move("panorama", eye, tgt, 1.0 + 0.4 * sweep / PANO_SWEEP[1], why={"sweep_deg": round(sweep)}))
        return _keep_best(out, 12)

    def _descend(self, room, pts, rmin, dur, prev, earlier) -> list[Move]:
        """A descend: the rise played the other way - from RISE_Z down to eye height while drifting
        toward a low subject (a dining table, a bed), the tilt easing from RISE_PITCH to level. It
        comes down ONTO what it shows, never past it."""
        sp = self.space
        if "descend" not in self.profile(room)["prior"]:
            return []
        top = RISE_Z
        if "scene_ceiling_missing" not in getattr(sp, "warnings", []):
            top = min(RISE_Z, float(sp.room_height) - RISE_HEADROOM)
        if top < EYE + 0.35:
            return []
        low = [(np.array([*(pc.frame or pc).centre, (pc.frame or pc).top]), piece_box(pc.frame or pc), pc.cls)
               for pc in sp.main_pieces(room) if (pc.frame or pc).top <= 1.0]
        if not low:
            return []
        u = np.linspace(0, 1, SAMPLES)
        w = u * u * (3 - 2 * u)
        z = top + (EYE - top) * w
        out = []
        for p in sorted(pts, key=lambda q: -sp.clearance(q))[:10]:
            for ta, box, name in low:
                v = ta[:2] - p
                d = float(np.linalg.norm(v))
                if not RISE_DIST[0] <= d <= RISE_DIST[1]:
                    continue
                for drift in RISE_DRIFT:
                    p1 = p + drift * v / d
                    if sp.clearance(p1) < rmin or sp.room_at(p1) not in sp.spaces.get(room, {room}):
                        continue
                    eye = np.outer(1 - w, p) + np.outer(w, p1)
                    h = np.linalg.norm(ta[:2] - eye, axis=1)
                    pitch = np.radians(RISE_PITCH + (PITCH_LADDER[0] - RISE_PITCH) * w)
                    look = np.minimum(np.tan(pitch), (z - 0.2) / np.maximum(h, 0.1))
                    tgt = np.c_[np.repeat(ta[None, :2], SAMPLES, 0), z - h * look]
                    first, last = (eye[0], tgt[0]), (eye[-1], tgt[-1])
                    if jump_cut(prev, first) or repeats(first, last, earlier):
                        continue
                    out.append(Move("descend", eye, tgt, 1.1 + 0.2 * drift, subject=box, z=z,
                                    why={"from_m": round(top, 2), "onto": name, "drift_m": drift}))
        return _keep_best(out, 12)

    # continuous cuts (client review 2026-11-20, items 2-4): every shot between the tour's first and
    # last plays at a steady pace (render/camera_path.ramp_profile), so a candidate is judged at its
    # mean speed plus PEAK_STEADY for the arc-length mix of travel and turn - not at the eased peaks
    continuous = False

    def peak(self, eased: float) -> float:
        """A move's peak speed over its mean: the eased profile's, or a steady pace's."""
        return PEAK_STEADY if self.continuous else eased

    def moves_enough(self, m: Move, dur: float) -> bool:
        """A room shot is seen to move: it travels MIN_TRAVEL_PS m/s, or turns MIN_TURN_PS deg/s."""
        travel = float(np.linalg.norm(np.diff(m.eye3, axis=0), axis=1).sum())
        d0, d1 = _dir(m.eye[0], m.tgt[0]), _dir(m.eye[-1], m.tgt[-1])
        turn = math.degrees(math.acos(max(-1.0, min(1.0, float(d0 @ d1)))))
        m.why.setdefault("travel_m", round(travel, 2))
        m.why["turn_deg"] = round(turn, 1)
        # travel and turn add up: a slow slide with a slow turn moves too (green house v15: the
        # primary bedroom's walks onto the bed - 0.8 m in a 1 m aisle, turning - read as stills)
        if travel / (MIN_TRAVEL_PS * dur) + turn / (MIN_TURN_PS * dur) >= 1.0:
            return True
        if m.kind in ("push", "pull"):             # what a push shows moving: its subject growing
            a0 = float(np.linalg.norm(m.tgt[0][:2] - m.eye[0]))
            a1 = float(np.linalg.norm(m.tgt[-1][:2] - m.eye[-1]))
            scale = abs(math.log(max(a0, 0.1) / max(a1, 0.1)))
            m.why["scale"] = round(math.exp(scale), 2)
            return scale >= MIN_SCALE_PS * dur
        return False

    def framed(self, m: Move) -> bool:
        """The subject a room shot ends on is whole in its last frame, with floor under it. A push
        or pull that would cut the floor off tilts down as it goes (up to TILT_MAX at its end) -
        a push in that settles onto the counter - rather than being dropped."""
        if m.subject is None:
            return True
        ze = float(m.zs[-1])
        if frames_fully(m.eye[-1], m.tgt[-1], m.subject, ze):
            return True
        if m.piece is not None and m.piece.cls in FAR_FLOOR and frames_on_far_floor(m.eye[-1], m.tgt[-1], m.subject, ze):
            m.why["far_floor"] = True
            return True
        if m.piece is not None and m.piece.cls == "bed":
            # a bed the room's free floor cannot fit whole: most of it and its headboard end
            fp = m.piece.frame or m.piece
            share, head = bed_seen(m.eye[-1], m.tgt[-1], fp, self.bed_head(fp), ze)
            m.why["bed_seen"] = round(share, 2)
            if share >= BED_SEEN and head:
                return True
        if m.kind not in ("push", "pull"):
            return False
        h = float(np.linalg.norm(m.tgt[-1][:2] - m.eye[-1][:2]))
        p0 = math.degrees(math.atan2(ze - m.tgt[-1][2], h))
        w = np.linspace(0.0, 1.0, len(m.tgt))
        w = w * w * (3 - 2 * w)                            # eased: the tilt grows as the move closes
        for p in np.arange(math.ceil(p0) + 1.0, TILT_MAX + 0.5, 1.0):
            dz = float(m.tgt[-1][2] - (ze - h * math.tan(math.radians(p))))
            end = m.tgt[-1].copy()
            end[2] -= dz
            if frames_fully(m.eye[-1], end, m.subject, ze):
                m.tgt = m.tgt.copy()
                m.tgt[:, 2] -= dz * w
                m.why["tilt_deg"] = round(float(p), 1)
                return True
        return False

    # ---- move kinds ------------------------------------------------------------------------------
    def _path(self, paths, i, j, rmin):
        """The smoothed walking path from point i to j and its length - cached on the path finder
        (each kind of move, and each step of the pitch ladder, asks for the same pairs)."""
        cache = paths.__dict__.setdefault("smoothed", {})
        key = (i, j, rmin)
        if key not in cache:
            poly = paths(i, j)
            if poly is None or len(poly) < 2:
                cache[key] = (None, 0.0)
            else:
                q = smooth(self.space, poly, rmin)
                cache[key] = (q, float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum()))
        q, plen = cache[key]
        return (None if q is None else q.copy()), plen

    def _walkway(self, room, piece, pts, paths, rmin, dur, pitch, prev, earlier, own) -> list[Move]:
        fp = piece.frame or piece                     # what is framed: the labelled object, else the piece
        top = np.array([*self.aim_on(fp), fp.top])
        out = []
        keep = [k for k, p in enumerate(pts) if piece.dist(p) > 0.05]
        for a_i, a in enumerate(keep):
            for b in keep[a_i + 1:]:
                L = float(np.linalg.norm(pts[b] - pts[a]))
                if not MOVE_LEN[0] <= L <= MOVE_LEN[1]:
                    continue
                for i, j in ((a, b), (b, a)):
                    # a walkway's subject is the room's own piece: it may start anywhere in the open
                    # plan (a dining table filmed from the living side is a dining shot)
                    path, plen = self._path(paths, i, j, rmin)
                    if path is None or plen > 1.3 * L or plen / dur * self.peak(1.5) > self.move_budget:
                        continue
                    path = resample(path, SAMPLES)
                    tgt = cap_pitch(path, np.repeat(top[None], len(path), 0), pitch)
                    first, last = (path[0], tgt[0]), (path[-1], tgt[-1])
                    if jump_cut(prev, first) or repeats(first, last, earlier):
                        continue
                    core = self.core_of(fp, path, tgt)
                    if core is None:                          # the piece leaves the frame somewhere
                        continue
                    par = parallax(path, tgt)
                    d0 = float(np.linalg.norm(path[0] - fp.centre))
                    k = len(path) // 2
                    depth = self.space.view_depth(path[k], math.atan2(*(tgt[k][:2] - path[k])[::-1]), HFOV)
                    score = (1.0 + 1.0 * par + 0.3 * plen + 1.5 * core - 1.5 * abs(d0 - SIT_BACK) / SIT_BACK
                             + 0.1 * min(depth, 5.0) - 0.15 * (pitch - PITCH_LADDER[0]))
                    out.append(Move("walkway", path, tgt, score, pitch, piece, subject=piece_box(fp),
                                    why={"parallax": round(par, 2), "travel_m": round(plen, 2), "core": core,
                                         "start_from_piece_m": round(d0, 2), "pitch_cap": pitch}))
        return out

    def _glide(self, room, pts, paths, rmin, dur, prev, earlier, opening, own, closing=False) -> list[Move]:
        out = []
        for a in sorted(own):
            for b in range(len(pts)):
                if a == b:
                    continue
                L = float(np.linalg.norm(pts[b] - pts[a]))
                if not MOVE_LEN[0] + 0.4 <= L <= MOVE_LEN[1] + 1.4:
                    continue
                path, plen = self._path(paths, a, b, rmin)
                if path is None or plen > 1.25 * L or plen / dur * self.peak(1.5) > self.move_budget:
                    continue
                path = resample(path, SAMPLES)
                d = path[-1] - path[0]
                d /= np.linalg.norm(d)
                head = math.atan2(d[1], d[0])
                reach = self.space.ray(path[-1], head)
                if reach < 1.0:                                  # walking into a wall
                    continue
                far = path[-1] + d * min(reach, 6.0)
                dist = float(np.linalg.norm(far - path[0]))
                aims = [("depth", np.array([*far, EYE - min(0.45, 0.12 * dist)]), None)]
                for o in self.space.room_objects(room, FEATURES):      # or push toward a feature ahead
                    if o["max"][2] > LOW_TOP or o["class"] in NO_AIM:  # never upper cabinets, a door, a curtain
                        continue
                    v = np.asarray(o["centre"][:2]) - path[-1]
                    if np.linalg.norm(v) > 0.8 and float(v @ d) / np.linalg.norm(v) > 0.5:
                        aims.append((o["class"], np.array([*o["centre"][:2], max(o["centre"][2], EYE - 0.6)]),
                                     objects_box([o])))
                for why_aim, aim, box in aims:
                    tgt = cap_pitch(path, np.repeat(aim[None], len(path), 0), PITCH_LADDER[0])
                    first, last = (path[0], tgt[0]), (path[-1], tgt[-1])
                    if jump_cut(prev, first) or repeats(first, last, earlier):
                        continue
                    depth = self.space.view_depth(path[0], math.atan2(*(aim[:2] - path[0])[::-1]), HFOV)
                    score = (0.3 * plen + 0.25 * min(depth, 6.0) + 0.3 * parallax(path, tgt)
                             + (OPENING_GLIDE if opening else 0.0) + (0.25 * min(depth, 6.0) if opening or closing else 0.0))
                    out.append(Move("glide", path, tgt, score, subject=box,
                                    why={"travel_m": round(plen, 2), "view_depth_m": round(depth, 2), "aim": why_aim}))
        return out

    def _pan(self, room, pts, rmin, dur, prev, earlier, closing, opening=False) -> list[Move]:
        sp = self.space
        out = []
        spots = sorted(pts, key=lambda p: -sp.clearance(p))[:12]
        heads = np.radians(np.arange(0, 360, 15))
        for p in spots:
            depths = [sp.view_depth(p, h, HFOV) for h in heads]
            hc = float(heads[int(np.argmax(depths))])
            # drift a little towards the view, where there is room (parallax)
            fwd = np.array([math.cos(hc), math.sin(hc)])
            e1 = p + 0.35 * fwd
            if sp.clearance(e1) < rmin or sp.room_at(e1) not in sp.spaces.get(room, {room}):
                e1 = p + 0.05 * fwd
            for sign in (1, -1):
                sweep = PAN_SWEEP
                for _ in range(12):                     # too fast for its length: sweep less
                    s = np.linspace(0, 1, SAMPLES)
                    eye = np.outer(1 - s, p) + np.outer(s, e1)
                    hs = hc + sign * math.radians(sweep) * (s - 0.5)
                    tgt = np.c_[eye[:, 0] + 2.5 * np.cos(hs), eye[:, 1] + 2.5 * np.sin(hs), np.full(SAMPLES, EYE - LOOK_DROP)]
                    turn, move = trial_speeds(eye, tgt, dur)
                    if turn <= self.turn_budget and move <= self.move_budget:
                        break
                    sweep *= 0.85
                else:
                    continue
                first, last = (eye[0], tgt[0]), (eye[-1], tgt[-1])
                if jump_cut(prev, first) or repeats(first, last, earlier):
                    continue
                depth = float(np.mean([sp.view_depth(e, h, HFOV) for e, h in zip(eye[::10], hs[::10])]))
                score = (0.25 * min(depth, 6.0) + 0.4 * sweep / PAN_SWEEP + (0.6 if closing else 0.0) - 0.3
                         + (0.25 * min(depth, 6.0) if opening or closing else 0.0))
                out.append(Move("pan", eye, tgt, score,
                                why={"sweep_deg": round(sweep), "view_depth_m": round(depth, 2),
                                     "turn_dps": round(turn, 1), "move_mps": round(move, 2)}))
        return out

    def _hold(self, least: Move, room: str) -> Move:
        """Every move runs into something near the lens: hold the least blocked start, with a breath
        of pull-back (never move into the blockage)."""
        # not a still (user 2026-11-16, Krk: a 9 s hold read as a frozen frame): from the held spot
        # the view turns HOLD_SWEEP deg, toward the side that shows more of the room
        p, t = least.first
        sp = self.space
        h0 = math.atan2(t[1] - p[1], t[0] - p[0])
        r = float(np.linalg.norm(t[:2] - p))
        side = max((1, -1), key=lambda sg: sp.own_share(room, p, h0 + sg * math.radians(HOLD_SWEEP), HFOV))
        s = np.linspace(0, 1, SAMPLES)
        hs = h0 + side * math.radians(HOLD_SWEEP) * (s * s * (3 - 2 * s))
        eye = np.repeat(np.asarray(p)[None], SAMPLES, 0)
        tgt = np.c_[p[0] + r * np.cos(hs), p[1] + r * np.sin(hs), np.full(SAMPLES, t[2])]
        m = Move("hold_pan", eye, tgt, least.score, least.pitch, least.piece, least.occ,
                 why={"held_because": "every move is blocked near the lens", "occlusion": round(least.occ, 3),
                      "sweep_deg": HOLD_SWEEP})
        return m

    def to_pace(self, m: Move, dur: float, least: float = PACE_MIN_KEEP) -> None:
        """In place: a candidate at the tour's 3D pace (pace_target, %fw/s; speed_band.BAND_3D) before
        anything judges it - the speed band would play only the last share of a faster move, and the
        rules (framing, the lens and void checks, the walls it shows, the cut) judged the whole of it
        (user 2026-11-27: pans planned at 5-8 deg/s played at ~1, a walk through a door lost its door).
        Faster: its start goes, the end it frames kept (as the band trims). Slower than the band by
        more than its cut ratio: it loses score - one slow shot holds every later cut down."""
        target = getattr(self, "pace_target", None)
        if not target or m.kind in ("detail", "link") or len(m.eye) < 8:
            return
        prev = self.__dict__.get("_last_exit_pct")       # the cut into it: within PACE_CUT_RATIO
        s0, s1 = self.pace_ends(m, dur)
        rep = (s0 + s1) / 2
        m.why["pace_pct"] = round(rep, 2)
        keep = 1.0
        if rep > target * 1.02:
            keep = target / rep
        if prev and s0 * keep > prev * PACE_CUT_RATIO:
            keep = min(keep, prev * PACE_CUT_RATIO / s0)
        # ... and its peak turn / travel well under the caps (the fitted curve runs a little over the
        # samples: a walkway planned under 9 deg/s played 10.1, 2026-11-27)
        turn_pk, move_pk = self.peaks(m, dur)
        if turn_pk * keep > PACE_PEAK_SHARE * MAX_TURN_DPS:
            keep = min(keep, PACE_PEAK_SHARE * MAX_TURN_DPS / turn_pk)
        if move_pk * keep > PACE_PEAK_SHARE * MAX_MOVE_MPS:
            keep = min(keep, PACE_PEAK_SHARE * MAX_MOVE_MPS / move_pk)
        if keep < 0.98:
            # never below being seen to move (moves_enough: travel and turn add up) - the pace and
            # that floor are one number (2.5 deg/s), and a slow shot before pulled pans under it
            travel = float(np.linalg.norm(np.diff(m.eye3, axis=0), axis=1).sum())
            d0, d1 = _dir(m.eye[0], m.tgt[0]), _dir(m.eye[-1], m.tgt[-1])
            turn = math.degrees(math.acos(max(-1.0, min(1.0, float(d0 @ d1)))))
            seen = travel / (MIN_TRAVEL_PS * dur) + turn / (MIN_TURN_PS * dur)
            if seen > 1e-6 and m.kind in PACE_SEEN_KINDS:
                least = max(least, min(1.0, PACE_SEEN_MARGIN / seen))
            keep = max(least, keep)
            trim_move_start(m, keep)
            m.why["paced_keep"] = round(keep, 2)
            m.why.pop("front_m", None)               # (measured again on what plays)
            s0, s1 = self.pace_ends(m, dur)
            rep = (s0 + s1) / 2
        if rep < target / PACE_SLOW_RATIO:
            m.score -= PACE_SLOW_W * (1.0 - rep / target)
        if prev and s0 < prev / PACE_CUT_RATIO:      # slower in than the cut allows: the band would slow
            m.score -= PACE_SLOW_W * (1.0 - s0 * PACE_CUT_RATIO / prev)    # the shot before it

    def peaks(self, m: Move, dur: float) -> tuple[float, float]:
        """(peak turn deg/s, peak travel m/s) of a move as the renderer plays it (steady along its
        arc), from its samples - over 3 samples, so one kink does not read as a peak. (trial_speeds
        fits the curve first: on every candidate that was 38 s of a green-house plan, 2026-11-27.)"""
        from tour_worker.scene.camera import TARGET_WEIGHT
        eye3, tgt = m.eye3, np.asarray(m.tgt, float)
        de = np.linalg.norm(np.diff(eye3, axis=0), axis=1)
        step = de + TARGET_WEIGHT * np.linalg.norm(np.diff(tgt, axis=0), axis=1)
        total = float(step.sum())
        if total < 1e-9 or len(step) < 3:
            return 0.0, 0.0
        d = tgt - eye3
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        ang = np.degrees(np.arccos(np.clip(np.sum(d[1:] * d[:-1], axis=1), -1.0, 1.0)))
        k = np.ones(3)
        dt = np.convolve(dur * step / total, k, "valid")
        return (float(np.max(np.convolve(ang, k, "valid") / np.maximum(dt, 1e-6))),
                float(np.max(np.convolve(de, k, "valid") / np.maximum(dt, 1e-6))))

    def ahead_of(self, pose, m: Move) -> bool:
        """m carries a walk-in on forward (pose: its (eye, aim, walking direction)): m starts ahead of
        where it ends (not behind it by more than JOIN_BACK_M) and does not set off backward (user
        2026-11-27: "the effect where the camera goes back, rotates and goes ahead looks horrible")."""
        e0, d = np.asarray(pose[0], float)[:2], np.asarray(pose[2], float)[:2]
        if float(np.dot(np.asarray(m.eye[0], float) - e0, d)) < -JOIN_BACK_M:
            return False
        mo = move_motion(m, end=False)
        sp_ = math.sqrt(mo.x ** 2 + mo.y ** 2 + mo.f ** 2)
        return not (sp_ > 1e-9 and mo.f < -JOIN_BACK_SHARE * sp_)

    def joins_forward(self, walk: Move, br: Move) -> bool:
        """The bridge from a walk-in never walks back along the way it came (its eye's progress along
        the walk-in's last direction never falls by more than JOIN_BACK_M)."""
        d = self.walk_dir(walk)
        proj = (np.asarray(br.eye, float) - np.asarray(br.eye[0], float)) @ d
        return float(np.min(proj - np.maximum.accumulate(proj))) >= -JOIN_BACK_M

    def walk_dir(self, walk: Move) -> np.ndarray:
        v = np.asarray(walk.eye[-1], float) - np.asarray(walk.eye[max(0, len(walk.eye) - 6)], float)
        if float(np.linalg.norm(v)) < 1e-6:
            v = np.asarray(walk.tgt[-1][:2], float) - np.asarray(walk.eye[-1], float)
        return v / max(float(np.linalg.norm(v)), 1e-9)

    def join_s(self, pose, m: Move) -> float:
        """The time (s) a bridge (RoomMoves.bridge) from pose (eye, aim) to m's first pose would
        take, estimated: its walk at BRIDGE_MPS, its turn at BRIDGE_DPS."""
        e0, t0 = np.asarray(pose[0], float)[:2], np.asarray(pose[1], float)
        e1, t1 = np.asarray(m.eye[0], float), np.asarray(m.tgt[0], float)
        h0 = math.atan2(t0[1] - e0[1], t0[0] - e0[0])
        h1 = math.atan2(t1[1] - e1[1], t1[0] - e1[0])
        turn = abs(math.degrees((h1 - h0 + math.pi) % (2 * math.pi) - math.pi))
        v = m.why.setdefault("join_s", round(max(float(np.linalg.norm(e1 - e0)) / BRIDGE_MPS, turn / BRIDGE_DPS), 2))
        return float(v)

    def bridge(self, a: Move, b: Move) -> Move | None:
        """One take through the home (user 2026-11-27: "enter the room, circle it, then see its exit
        and enter the other room and do the panning and other camera moves all in one shot"): the
        camera from where move a ends to where move b starts - across the room, or along the walking
        route through the doors between them - from a's last aim to b's first, looking ahead along
        the way when it walks; at the 3D pace (BRIDGE_MPS walking, at most BRIDGE_DPS turning).
        None when no walking route joins them, or only a long one (the tour cuts there)."""
        sp = self.space
        p0, p1 = np.asarray(a.eye[-1], float), np.asarray(b.eye[0], float)
        l0, l1 = np.asarray(a.tgt[-1], float), np.asarray(b.tgt[0], float)
        h0 = math.atan2(l0[1] - p0[1], l0[0] - p0[0])
        h1 = math.atan2(l1[1] - p1[1], l1[0] - p1[0])
        r0, r1 = float(np.linalg.norm(l0[:2] - p0)), float(np.linalg.norm(l1[:2] - p1))
        wrap = lambda x: (x + math.pi) % (2 * math.pi) - math.pi   # noqa: E731
        dist = float(np.linalg.norm(p1 - p0))
        if dist < BRIDGE_STILL_M:
            path = p0 + np.linspace(0.0, 1.0, SAMPLES)[:, None] * (p1 - p0)[None]
        else:
            route = sp.paths_between(np.array([p0, p1]), sp.dt >= BRIDGE_CLEAR)(0, 1)
            if route is None or len(route) < 2:
                return None
            if float(np.linalg.norm(np.diff(np.asarray(route, float), axis=0), axis=1).sum()) > BRIDGE_MAX_M:
                return None
            poly = np.r_[[p0], np.asarray(route, float)[1:-1], [p1]] if len(route) > 2 else np.array([p0, p1])
            path = resample(smooth(sp, poly, BRIDGE_CLEAR), SAMPLES)
            path[0], path[-1] = p0, p1
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        L = float(arc[-1])
        u = arc / L if L > 1e-6 else np.linspace(0.0, 1.0, SAMPLES)
        ease = lambda x: x * x * (3 - 2 * x)   # noqa: E731
        if L >= BRIDGE_AHEAD_FROM_M:          # a walk: it looks where it is going in between
            ahead = []
            for k, s_ in enumerate(arc):
                q = path[min(len(path) - 1, int(np.searchsorted(arc, s_ + EXIT_AHEAD)))] - path[k]
                ahead.append(math.atan2(q[1], q[0]) if float(np.linalg.norm(q)) > 0.2 else None)
            last = h1
            for k in range(len(ahead) - 1, -1, -1):  # (near the end there is no "ahead": b's look)
                ahead[k] = last if ahead[k] is None else ahead[k]
                last = ahead[k]
            w_in = ease(np.clip(u / BRIDGE_TURN_SHARE, 0, 1))
            w_out = ease(np.clip((u - (1 - BRIDGE_TURN_SHARE)) / BRIDGE_TURN_SHARE, 0, 1))
            hd = np.array([h0 + wi * wrap(ah - h0) for wi, ah in zip(w_in, ahead)])
            hd = np.array([h + wo * wrap(h1 - h) for h, wo in zip(hd, w_out)])
        else:                                  # a step and a turn: from a's look to b's
            hd = h0 + ease(u) * wrap(h1 - h0)
        hd = np.unwrap(hd)
        reach = r0 + (r1 - r0) * ease(u)
        z = l0[2] + (l1[2] - l0[2]) * ease(u)
        tgt = np.c_[path + reach[:, None] * np.c_[np.cos(hd), np.sin(hd)], z]
        tgt[0], tgt[-1] = l0, l1              # (exactly a's last aim and b's first: no seam)
        turn = float(np.degrees(np.abs(np.diff(hd)).sum()))
        dur = max(L / BRIDGE_MPS, turn / BRIDGE_DPS, BRIDGE_MIN_S)
        return Move("bridge", path, tgt, 0.0, easing="linear",
                    why={"travel_m": round(L, 2), "turn_deg": round(turn, 1), "dur_s": round(dur, 2)})

    def wall_share(self, path, tgt) -> float:
        """The share of a move's first and last views (across the frame) a wall or tall thing
        fills within NEAR_WALL_M of the lens (user 2026-11-27: the green house bedroom walk-in started
        on a plain wall and kept one over the right half of the frame all the way in)."""
        half = math.radians(HFOV) / 2
        out = []
        for e, t in ((path[0], tgt[0]), (path[-1], tgt[-1])):
            h = math.atan2(t[1] - e[1], t[0] - e[0])
            out.append(float(np.mean(self.space.rays(e, h + np.linspace(-half, half, 11)) < NEAR_WALL_M)))
        return float(np.mean(out))

    def pace_ends(self, m: Move, dur: float) -> tuple[float, float]:
        """A move's on-screen speed (%fw/s) entering and leaving its shot - the speed band's measure
        (render/continuity.py units), its samples taken as evenly spread in time."""
        from tour_worker.scene.camera import TARGET_WEIGHT
        n = len(m.eye)
        k = min(4, n - 1)
        # the renderer plays a path at an even pace along its arc (the eye's travel + TARGET_WEIGHT x
        # the aim's), not its samples: an eased pan's ends play as fast as its middle (spread evenly
        # in sample time they read 3-4x slow - wall pans estimated 1.5 played 7-12 %fw/s, 2026-11-27)
        step = (np.linalg.norm(np.diff(m.eye3, axis=0), axis=1)
                + TARGET_WEIGHT * np.linalg.norm(np.diff(np.asarray(m.tgt, float), axis=0), axis=1))
        total = float(step.sum())
        if total < 1e-9:
            return 0.0, 0.0
        t0 = max(dur * float(step[:k].sum()) / total, 1e-6)
        t1 = max(dur * float(step[-k:].sum()) / total, 1e-6)
        a, b = (math.sqrt(mo.x ** 2 + mo.y ** 2 + mo.f ** 2) * 100.0 / t
                for mo, t in ((move_motion(m, end=False), t0), (move_motion(m, end=True), t1)))
        return float(a), float(b)

    def pitch_room(self, m: Move) -> float:
        """Degrees the move may still tilt down before PITCH_MAX_ROOM (PITCH_MAX_RISE for a rise /
        descend): its steepest look so far, against the cap."""
        e3 = m.eye3
        d = m.tgt - e3
        h = np.linalg.norm(d[:, :2], axis=1)
        steepest = float(np.degrees(np.arctan2(-d[:, 2], np.maximum(h, 1e-6))).max())
        cap = PITCH_MAX_RISE if m.kind in ("rise", "descend") else PITCH_MAX_ROOM
        if cap == PITCH_MAX_ROOM and "scene_ceiling_missing" in getattr(self.space, "warnings", []):
            cap = PITCH_MAX_NO_CEILING
        return cap - steepest

    def band_deficit(self, m: Move) -> float:
        """Degrees the view's top edge rises past where the scan's walls end (Space.wall_top), at
        its start, middle and end, across the frame - 0 when the scene has its ceiling."""
        sp = self.space
        top_map = getattr(sp, "wall_top", None)
        if top_map is None or "scene_ceiling_missing" not in getattr(sp, "warnings", []):
            return 0.0
        half_v = math.atan(math.tan(math.radians(HFOV) / 2) * FRAME_H / FRAME_W)
        offs = np.radians(np.array([-0.35, 0.0, 0.35]) * HFOV)
        e3 = m.eye3
        worst = 0.0
        for k in (0, len(e3) // 2, len(e3) - 1):
            e, t = e3[k], m.tgt[k]
            d = t - e
            hd = float(np.linalg.norm(d[:2]))
            head, pitch = math.atan2(d[1], d[0]), math.atan2(-d[2], max(hd, 1e-6))
            dist = sp.rays(e[:2], head + offs)
            for a, D in zip(offs, dist):
                hit = e[:2] + (D + 0.05) * np.array([math.cos(head + a), math.sin(head + a)])
                i, j = sp.cell(hit)
                if not (0 <= i < top_map.shape[0] and 0 <= j < top_map.shape[1]) or top_map[i, j] <= 0:
                    continue
                elev = math.atan(math.tan(half_v - pitch) * math.cos(a))
                need = math.atan2(float(top_map[i, j]) - BAND_MARGIN - e[2], max(float(D), 0.3))
                worst = max(worst, math.degrees(elev - need))
        return worst

    def _occ_call(self, what: str, eye, tgt, piece, near: float, region: str):
        """occ.cover / occ.void through a cache keyed by the pose (eye and target to OCC_CACHE_M), the
        near distance, the region and the framed piece: a pose rendered once is never rendered again
        (reversed twins and near-duplicate candidates share poses - the Home Scan's primary bedroom
        spent its whole budget on them, 2026-11-26). Counts the renders made (_occ_renders)."""
        q = lambda a: tuple(int(round(float(x) / OCC_CACHE_M)) for x in np.ravel(a)[:3])   # noqa: E731
        key = (what, q(eye), q(tgt), id(piece) if piece is not None else None, round(near, 3), region)
        cache = self.__dict__.setdefault("_occ_cache", {})
        if key not in cache:
            self._occ_renders = self.__dict__.get("_occ_renders", 0) + 1
            cache[key] = (self.occ.void(eye, tgt, HFOV, top=True) if what == "void"
                          else self.occ.cover(eye, tgt, HFOV, piece, near, region=region))
        return cache[key]

    def _void(self, m: Move) -> float:
        """The worst share of the view the scan never captured (start, middle, end). A black band
        across the top (a missing ceiling, TOP_VOID_MAX of its top fifth) tilts the move down
        TOP_TILT deg; if that does not clear it (or loses the subject's floor) the move fails."""
        e3 = m.eye3
        ks = (0, len(e3) // 2, len(e3) - 1)
        # no ceiling: the frame's top edge stays under where the scan's walls end (geometric, the
        # thinned void render misses a thin band); a few degrees down, else the move goes
        deficit = self.band_deficit(m)
        if deficit <= BAND_TOLERANCE:
            deficit = 0.0
        if deficit > 0:
            if deficit + 0.5 > min(BAND_TILT_MAX, self.pitch_room(m)) or m.kind in ("detail", "link"):
                m.why["band_deficit_deg"] = round(deficit, 1)
                return 1.0
            old = m.tgt
            h = np.linalg.norm(m.tgt[:, :2] - m.eye, axis=1)
            m.tgt = m.tgt.copy()
            m.tgt[:, 2] -= h * math.tan(math.radians(deficit + 0.5))
            if not self.framed(m) or self.band_deficit(m) > 0:
                m.tgt = old
                m.why["band_deficit_deg"] = round(deficit, 1)
                return 1.0
            m.why["tilt_for_band"] = round(deficit + 0.5, 1)

        def worst(tgt):
            r = [self._occ_call("void", e3[k], tgt[k], None, 0.0, "") for k in ks]
            return max(a for a, _ in r), max(b for _, b in r)
        whole, top = worst(m.tgt)
        m.why["void_top"] = round(top, 3)
        if top <= TOP_VOID_MAX or m.kind in ("detail", "link"):
            return whole
        h = np.linalg.norm(m.tgt[:, :2] - m.eye, axis=1)
        room = self.pitch_room(m)
        for deg in TOP_TILT:
            if deg > room:                       # never past the tilt cap: the move fails instead
                break
            tgt = m.tgt.copy()
            tgt[:, 2] -= h * math.tan(math.radians(deg))
            w2, t2 = worst(tgt)
            if t2 <= TOP_VOID_MAX:
                old = m.tgt
                m.tgt = tgt
                if self.framed(m):
                    m.why.update(void_top=round(t2, 3), tilt_for_void=deg)
                    return w2
                m.tgt = old
        return 1.0

    def _cover(self, m: Move) -> float:
        if self.occ is None:
            return 0.0
        e3 = m.eye3
        worst = 0.0
        for k in (len(e3) // 2, 0, len(e3) - 1):
            e, t = e3[k], m.tgt[k]
            worst = max(worst, self._occ_call("cover", e, t, m.piece, OCC_NEAR_KIND.get(m.kind, OCC_NEAR), "centre"))
            if worst > MAX_OCC:                       # already out: the other poses cannot bring it back
                break
            if m.kind not in ("link", "detail", "exit", "walk", "backout", "through", "door_pan", "loop", "bridge"):     # (they pass door frames)
                # a door frame or a wall at the frame's edge, right at the lens (user 2026-11-16, Krk:
                # the bedroom shot ended with the door frame over a quarter of the frame)
                edge = self._occ_call("cover", e, t, m.piece, EDGE_NEAR, "full")
                if edge > EDGE_MAX:
                    m.why["edge_cover"] = round(edge, 3)
                    return 1.0
        # anything big right at the lens, anywhere in the frame, all through the move - every kind,
        # exits included (green house v11: a fridge over a third of a rise, a TV over half an exit,
        # a sculpture in a pan)
        if m.kind != "detail":
            # (a flight through a door: its frame sweeps past the edges as the camera goes through -
            # only what comes within THROUGH_NEAR, a frame it would scrape, blocks it)
            r_near = (BACKOUT_NEAR if m.kind in ("backout", "door_pan")
                      else THROUGH_NEAR if m.kind in ("through", "loop", "bridge") else NEAR_ANY)
            for k in np.linspace(0, len(e3) - 1, NEAR_ANY_POSES).astype(int):
                near = self._occ_call("cover", e3[k], m.tgt[k], None, r_near, "full")
                if near > (NEAR_ANY_MAX_THROUGH if m.kind in ("through", "loop", "bridge") else NEAR_ANY_MAX):
                    m.why["near_lens"] = round(near, 3)
                    return 1.0
        return worst

