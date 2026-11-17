"""tools/scoring/calibrate.py (SOW PS-07): weights scored by agreement with human hero picks."""
from tools.scoring.calibrate import agreement, grid


def test_the_grid_sums_to_one_and_agreement_follows_the_weights():
    ws = list(grid(0.25))
    assert ws and all(abs(sum(w.values()) - 1.0) < 1e-9 for w in ws)
    rows = [{"listing": "L", "photo_id": "a", "room_key": "kitchen", "criteria": {"framing": 0.9, "lighting": 0.2, "coverage": 0.5, "depth_of_field": 0.5}},
            {"listing": "L", "photo_id": "b", "room_key": "kitchen", "criteria": {"framing": 0.2, "lighting": 0.9, "coverage": 0.5, "depth_of_field": 0.5}}]
    manifest = [{"listing": "L", "heroes": {"kitchen": "b"}}]
    framing_heavy = {"framing": 0.7, "lighting": 0.1, "coverage": 0.1, "depth_of_field": 0.1}
    lighting_heavy = {"framing": 0.1, "lighting": 0.7, "coverage": 0.1, "depth_of_field": 0.1}
    assert agreement(rows, manifest, framing_heavy)[0] == 0.0
    assert agreement(rows, manifest, lighting_heavy)[0] == 1.0
