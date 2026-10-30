"""Derive bounded opponent weights from a verified frozen Classic panel."""

import hashlib
import json
from pathlib import Path


def weights_from_panel(root, reference, *, policy_sha256, frozen_sha256, scripts, weights):
    root = Path(root).resolve()
    path = (root / reference["file"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Opponent evaluation escapes continuation input")
    blob = path.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != reference["sha256"]:
        raise ValueError("Opponent evaluation checksum differs")
    panel = json.loads(blob)
    names = ["frozen_" + sha[:12] for sha in frozen_sha256] + list(scripts)
    if (
        panel["checkpoint_sha256"] != policy_sha256
        or panel["frozen_policy_sha256"] != frozen_sha256
        or panel["opponent_weights"] != weights
        or panel["coworld_classic_rules"] is not True
        or panel["games"] != 4096
        or list(panel["by_opponent_and_seat"]) != names
    ):
        raise ValueError("Opponent evaluation does not match the resumed policy and pool")
    rates = []
    totals = dict(games=0, wins=0, losses=0, draws=0)
    for name in names:
        seats = panel["by_opponent_and_seat"][name]
        if set(seats) != {"0", "1"}:
            raise ValueError("Opponent evaluation must cover both seats")
        games = wins = 0
        for row in seats.values():
            counts = [row[key] for key in totals]
            if (
                any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in counts)
                or row["games"] <= 0
                or row["wins"] + row["losses"] + row["draws"] != row["games"]
            ):
                raise ValueError("Opponent evaluation has inconsistent outcome counts")
            games += row["games"]
            wins += row["wins"]
            for key in totals:
                totals[key] += row[key]
        rates.append(wins / games)
    if any(totals[key] != panel[key] for key in totals):
        raise ValueError("Opponent evaluation totals differ")
    difficulty = [(1 - rate) ** 2 for rate in rates]
    total = sum(difficulty)
    if total <= 0:
        raise ValueError("Perfect panel supplies no opponent difficulty signal")
    # Keep every opponent. Preserve approximately the old sampling mass while
    # bounding the largest CPU opponent group; draws count as non-wins.
    updated = [max(1, min(24, round(sum(weights) * value / total))) for value in difficulty]
    while sum(updated) > 256:
        updated[updated.index(max(updated))] -= 1
    return updated, dict(
        recipe="squared_nonwin_v1",
        evaluation_sha256=digest,
        checkpoint_sha256=policy_sha256,
        games=totals["games"],
        opponent_names=names,
        win_rates=rates,
        previous_weights=weights,
        weights=updated,
    )
