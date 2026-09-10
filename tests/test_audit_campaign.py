"""Retention must distinguish recapture, observed failure and missing future."""

import pytest

from scripts.audit_campaign import audit_frames, owned_components, parse_wire

PASS = [1, 0, 0, 0, 0]
EAST = [0, 0, 0, 3, 0]


def frame(turn, owned, action=PASS, *, external=False, hidden=False):
    wire = (f"{turn} {1 + owned} 8 1 2\n4 {0 if hidden else 1}\n"
            f"1 {int(owned)}\n6 {2 if owned else 0}\n")
    if external:
        return dict(turn=turn, public_wires=[wire, wire], replies=[dict(action=action), dict(action=PASS)])
    return dict(turn=turn, players=[dict(wire_observation=wire, applied_action=action),
                                   dict(wire_observation=wire, applied_action=PASS)])


@pytest.mark.parametrize("external", [False, True])
def test_boundary_capture_is_confirmed_at_tick_and_future_is_censored(external):
    report = audit_frames([frame(49, False, EAST, external=external),
                           frame(50, True, external=external)], 0)
    capture = report["captures"][0]
    assert capture["retention"]["next_land_tick"] == dict(turn=50, endpoint_observed=True, continuously_held=True)
    assert capture["retention"]["steps50"] == dict(turn=99, endpoint_observed=False, continuously_held=None)
    assert report["windows"][0]["plain_captures_held_to_next_tick"] == 1
    assert not report["windows"][0]["complete_action_window"]


def test_loss_then_recapture_does_not_repair_first_capture():
    frames = [frame(45, False, EAST), frame(46, True), frame(47, False, EAST),
              frame(48, True), frame(49, True), frame(50, True)]
    report = audit_frames(frames, 0)
    first, second = report["captures"]
    assert first["first_observed_loss"] == 47
    assert first["retention"]["next_land_tick"]["continuously_held"] is False
    assert second["retention"]["next_land_tick"]["continuously_held"] is True
    assert report["windows"][0]["confirmed_captures"] == 2
    assert report["windows"][0]["continuously_held_to_next_tick"] == 1


def test_observed_loss_is_false_even_if_horizon_is_not_observed():
    report = audit_frames([frame(10, False, EAST), frame(11, True), frame(12, False, hidden=True)], 0)
    retention = report["captures"][0]["retention"]["next_land_tick"]
    assert retention == dict(turn=50, endpoint_observed=False, continuously_held=False)
    assert report["windows"][0]["next_tick_endpoint_observed"] == 0


def test_terminal_attempt_is_unconfirmed_and_owned_move_is_not_capture():
    report = audit_frames([frame(48, True, EAST), frame(49, False, EAST)], 0)
    assert report["captures"] == []
    assert report["actions"][0]["kind"] == "owned_transfer"
    assert report["actions"][1]["capture_confirmed_next"] is None


def test_gaps_and_turn_mismatch_cannot_establish_continuous_ownership():
    with pytest.raises(ValueError, match="observation gaps"):
        audit_frames([frame(49, False, EAST), frame(51, True)], 0)
    bad = frame(49, False, EAST)
    bad["turn"] = 48
    with pytest.raises(ValueError, match="turn mismatch"):
        audit_frames([bad], 0)


def test_missing_wire_and_seat_do_not_silently_infer_public_state():
    with pytest.raises(ValueError, match="no recorded public observation"):
        audit_frames([dict(turn=0, replies=[dict(action=PASS)] * 2)], 0)
    with pytest.raises(ValueError, match="seat"):
        audit_frames([frame(0, False)], None)


def test_components_separate_stranded_army_from_home_and_do_not_wrap_rows():
    obs = parse_wire("200 3 22 0 0\n1 1 4\n1 1 1\n0 0 1\n1 1 0\n0 0 2\n10 10 0\n")
    assert owned_components(obs) == [
        dict(first_cell=3, land=2, army=20, contains_general=False),
        dict(first_cell=2, land=1, army=2, contains_general=True),
    ]
