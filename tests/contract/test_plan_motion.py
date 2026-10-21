"""Plan 1.4 / 1.5 validation (motion regions, music)."""
import copy

import pytest

from tests.perceptual_fixtures import _plan, _region
from tour_worker.contract.plan import compute_plan_hash, parse_plan


def test_plan_14_validation():
    assert parse_plan(_plan([_region()])).shots[0].motion_regions[0].effect == "fabric_sway"
    with pytest.raises(ValueError, match="need plan_version 1.4"):
        parse_plan(_plan([_region()], version="1.3"))
    with pytest.raises(ValueError, match="unsupported_motion_effect"):
        parse_plan(_plan([_region(effect="explode")]))
    with pytest.raises(ValueError, match="amplitude"):
        parse_plan(_plan([_region(amp=50.0)]))
    bad = copy.deepcopy(_region())
    bad["bbox"] = [0.5, 0.1, 0.4, 0.8]
    with pytest.raises(ValueError, match="bbox"):
        parse_plan(_plan([bad]))


def test_plan_15_with_music_is_accepted():
    p = _plan([_region()], version="1.5")
    p["music"] = {"track_id": "t", "uri": "s3://m/t.mp3", "sha256": "0" * 64, "end_s": 1.0, "synced": True}
    p["provenance"]["plan_hash"] = compute_plan_hash(p)
    plan = parse_plan(p)
    assert plan.plan_version == "1.5" and plan.shots[0].motion_regions[0].effect == "fabric_sway"
