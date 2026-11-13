"""The 3D tour planner (plan 1.6): a Gaussian splat scene and where its rooms are -> a camera plan.

The caller says only what each room is and one point inside it (tour_plan's `scene` + `rooms`).
Everything else is decided here, from the scene and the rules the tours were reviewed against:

  space.py      the scene as a floor plan: up, floor, free space, clearance, furniture, the rooms
                grown from their points, open-plan spaces, walking paths, how far a view reaches
  occlusion.py  how much of a view the splats right by the lens would hide (rendered, not guessed)
  moves.py      every move a room allows - walkway moves on its main piece, glides, reveal pans -
                checked (speed, horizon, framing, occlusion, cuts) and scored; the best one wins
  plan.py       the tour: the 2D planner's room order, screen time, roles and lengths (one source of
                truth for both planners), a move per shot, transitions, captions, branding, music

No photos are involved; a plan without a scene is the photo planner's (tour_worker/planner)."""
