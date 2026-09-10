"""Audit actual campaign actions and continuously held captures without inference.

Accept an arena game JSON with public ``players`` observations, or recorded
JSONL frames with ``public_wires`` and ``replies``. The latter needs --seat.
This measures the observed sequence; it does not certify game completion,
runtime, control equivalence, or which policy proposal produced an action.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def parse_wire(wire):
    rows = [[int(x) for x in line.split()] for line in wire.splitlines()]
    if len(rows) < 4 or len(rows[0]) != 5 or (len(rows) - 1) % 3:
        raise ValueError("expected five scores and three rectangular public planes")
    height, width = (len(rows) - 1) // 3, len(rows[1])
    if not width or any(len(row) != width for row in rows[1:]):
        raise ValueError("public planes must have one consistent nonzero width")
    planes = [sum(rows[1 + k * height:1 + (k + 1) * height], []) for k in range(3)]
    terrain, owners, armies = planes
    if any(x not in range(6) for x in terrain) or any(x not in range(4) for x in owners):
        raise ValueError("unsupported public tile/owner codes")
    if any(x < 0 for x in armies):
        raise ValueError("negative public armies")
    return dict(turn=rows[0][0], shape=(height, width), scores=rows[0][1:],
                terrain=terrain, owners=owners, armies=armies)


def normalize_frames(frames, seat):
    if type(seat) is not int or seat not in (0, 1):
        raise ValueError("seat must explicitly identify player 0 or 1")
    result = []
    for frame in frames:
        if "players" in frame:
            player = frame["players"][seat]
            wire, action = player["wire_observation"], player["applied_action"]
        elif "public_wires" in frame:
            wire = frame["public_wires"][seat]
            action = frame["replies"][seat]["action"]
        else:
            raise ValueError("frame has no recorded public observation; actions alone are insufficient")
        obs = parse_wire(wire)
        if obs["turn"] != frame["turn"]:
            raise ValueError("wire/frame turn mismatch")
        if (len(action) != 5 or any(type(x) is not int for x in action)
                or action[0] not in (0, 1, 2)):
            raise ValueError("expected five integers in the recorded applied action")
        result.append(dict(obs=obs, action=action))
    if not result:
        raise ValueError("no recorded decisions")
    first = result[0]["obs"]
    for offset, frame in enumerate(result):
        if frame["obs"]["shape"] != first["shape"] or frame["obs"]["turn"] != first["turn"] + offset:
            raise ValueError("shape changes, duplicate turns and observation gaps cannot establish retention")
    return result


def owned_components(obs):
    """Observed cardinal connectivity; aggregate army is not deployable force."""
    height, width = obs["shape"]
    unvisited = {i for i, owner in enumerate(obs["owners"]) if owner == 1}
    components = []
    while unvisited:
        start = min(unvisited)
        unvisited.remove(start)
        pending, members = [start], []
        while pending:
            cell = pending.pop()
            members.append(cell)
            row, column = divmod(cell, width)
            for dr, dc in _DIRECTIONS:
                nr, nc = row + dr, column + dc
                neighbor = nr * width + nc
                if 0 <= nr < height and 0 <= nc < width and neighbor in unvisited:
                    unvisited.remove(neighbor)
                    pending.append(neighbor)
        components.append(dict(first_cell=start, land=len(members), army=sum(obs["armies"][i] for i in members),
                               contains_general=any(obs["terrain"][i] == 4 for i in members)))
    return sorted(components, key=lambda c: (-c["land"], c["first_cell"]))


def audit_frames(frames, seat, *, land_period=50):
    if type(land_period) is not int or land_period <= 0:
        raise ValueError("land period must be a positive integer")
    sequence = normalize_frames(frames, seat)
    last_turn = sequence[-1]["obs"]["turn"]
    captures, actions, snapshots = [], [], []
    first_contact = first_general = None
    for offset, frame in enumerate(sequence):
        obs, action = frame["obs"], frame["action"]
        turn = obs["turn"]
        height, width = obs["shape"]
        if first_contact is None and 2 in obs["owners"]:
            first_contact = turn
        if first_general is None and any(o == 2 and t == 4 for o, t in zip(obs["owners"], obs["terrain"])):
            first_general = turn
        if turn % land_period == 0:
            plain_armies = sorted((army for owner, terrain, army in zip(obs["owners"], obs["terrain"], obs["armies"])
                                   if owner == 1 and terrain == 1), reverse=True)
            snapshots.append(dict(turn=turn, own_land=obs["scores"][0], own_army=obs["scores"][1],
                                  opponent_land=obs["scores"][2], opponent_army=obs["scores"][3],
                                  owned_components=owned_components(obs),
                                  largest_plain_army=max(plain_armies, default=0),
                                  top_five_plain_army=sum(plain_armies[:5])))
        event = dict(turn=turn, kind={0: "move", 1: "pass", 2: "build"}[action[0]])
        if action[0] == 0:
            _, row, column, direction, split = action
            if not (0 <= row < height and 0 <= column < width and direction in range(4) and split in (0, 1)):
                raise ValueError("applied move has invalid source, direction or split")
            dr, dc = _DIRECTIONS[direction]
            if not (0 <= row + dr < height and 0 <= column + dc < width):
                raise ValueError("applied move crosses the board boundary")
            source, destination = row * width + column, (row + dr) * width + column + dc
            owner = obs["owners"][destination]
            event.update(source=source, destination=destination,
                         kind={1: "owned_transfer", 2: "enemy_attempt", 3: "allied_transfer"}.get(
                             owner, "hidden_attempt" if obs["terrain"][destination] in (0, 5) else "neutral_attempt"))
            # An applied attack is not a capture certificate. Only the next
            # public frame can confirm the destination belongs to us afterward.
            after = sequence[offset + 1]["obs"] if offset + 1 < len(sequence) else None
            if owner not in (1, 3):
                event["capture_confirmed_next"] = after["owners"][destination] == 1 if after else None
                if event["capture_confirmed_next"]:
                    lost = next((f["obs"]["turn"] for f in sequence[offset + 2:]
                                 if f["obs"]["owners"][destination] != 1), None)
                    retention = {}
                    horizons = dict(next_land_tick=land_period * (turn // land_period + 1), steps50=turn + 50,
                                    steps100=turn + 100)
                    for label, horizon in horizons.items():
                        failed = lost is not None and lost <= horizon
                        held = False if failed else True if horizon <= last_turn else None
                        retention[label] = dict(turn=horizon, endpoint_observed=horizon <= last_turn,
                                                continuously_held=held)
                    captures.append(dict(turn=turn, destination=destination, kind=event["kind"],
                                         plain=after["terrain"][destination] == 1,
                                         first_observed_loss=lost, retention=retention))
        actions.append(event)
    windows = []
    first_turn = sequence[0]["obs"]["turn"]
    for start in range(first_turn // land_period * land_period, last_turn + 1, land_period):
        events = [e for e in actions if start <= e["turn"] < start + land_period]
        selected = [c for c in captures if start <= c["turn"] < start + land_period]
        eligible = [c for c in selected if c["retention"]["next_land_tick"]["endpoint_observed"]]
        windows.append(dict(start=start, end_exclusive=start + land_period,
                            complete_action_window=first_turn <= start and last_turn >= start + land_period - 1,
                            action_counts=dict(sorted(Counter(e["kind"] for e in events).items())),
                            confirmed_captures=len(selected), next_tick_endpoint_observed=len(eligible),
                            continuously_held_to_next_tick=sum(c["retention"]["next_land_tick"]["continuously_held"]
                                                              is True for c in eligible),
                            plain_captures_held_to_next_tick=sum(c["plain"] and c["retention"]["next_land_tick"][
                                "continuously_held"] is True for c in eligible)))
    return dict(schema_version=1, seat=seat, shape=sequence[0]["obs"]["shape"], decisions=len(sequence),
                first_turn=first_turn, last_observation_turn=last_turn, land_period=land_period,
                first_observed_contact=first_contact, first_observed_enemy_general=first_general,
                snapshots=snapshots, windows=windows, captures=captures, actions=actions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--seat", type=int, choices=(0, 1))
    parser.add_argument("--land-period", type=int, default=50, help="whole-land production interval; default50")
    parser.add_argument("--output", type=Path, required=True, help="new output file; refuses replacement")
    args = parser.parse_args()
    raw = args.input.read_bytes()
    if args.input.suffix == ".jsonl":
        frames = [json.loads(line) for line in raw.splitlines() if line.strip()]
        seat = args.seat
    else:
        record = json.loads(raw)
        frames = record["frames"]
        seat = args.seat if args.seat is not None else record["game"]["seat"]
    report = audit_frames(frames, seat, land_period=args.land_period)
    report["input"] = dict(path=str(args.input), sha256=hashlib.sha256(raw).hexdigest())
    report["limitations"] = [
        "Observed decisions only; no game completion, runtime or control-equivalence certification.",
        "Terminal attacks without a following observation are unconfirmed; unseen future retention is null.",
        "Recapture starts a new event and cannot repair an earlier event's lost continuous ownership.",
        "Ownership between adjacent observations and individual soldier identities are not observable.",
        "Held ordinary tiles are a production opportunity proxy, not a counterfactual income or win estimate.",
        "Public enemy ownership and scores are aggregate; this is not a distinct-opponent FFA census.",
    ]
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(f"Audited {report['decisions']} recorded decisions; {len(report['captures'])} confirmed capture events")


if __name__ == "__main__":
    main()
