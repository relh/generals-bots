"""Report hosted or population strength qualification; never changes champions."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path

Z95 = 1.959963984540054


def wilson_lower(wins: int, games: int) -> float:
    """Two-sided 95% Wilson interval lower bound; draws count as non-wins."""
    if games == 0:
        return 0.0
    p = wins / games
    correction = Z95 * Z95 / games
    return (p + correction / 2 - Z95 * math.sqrt(p * (1 - p) / games + correction / (4 * games))) / (1 + correction)


def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _seats(summary):
    raw = summary.get("by_opponent_and_seat")
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Summary requires by_opponent_and_seat counts")
    result = {}
    for label, item in raw.items():
        if not isinstance(item, dict):
            raise ValueError("Opponent counts must be objects")
        if "games" in item:  # Hosted summary: leader-seat0, relh-seat1.
            opponent, separator, side = label.rpartition("-seat")
            if not separator or not opponent or side not in ("0", "1"):
                raise ValueError(f"Invalid hosted opponent-seat label: {label}")
            entries = {side: item}
        else:  # Population evaluation: opponent -> {"0": counts, "1": counts}.
            opponent, entries = label, item
        target = result.setdefault(opponent, {})
        for side, counts in entries.items():
            if side not in ("0", "1") or side in target or not isinstance(counts, dict):
                raise ValueError(f"Invalid or duplicate seat for {opponent}: {side}")
            games = _count(counts.get("games"), "games")
            wins = _count(counts.get("wins"), "wins")
            losses = _count(counts.get("losses"), "losses")
            draws = _count(counts.get("draws", games - wins - losses), "draws")
            if wins + losses + draws != games:
                raise ValueError(f"Outcome totals differ for {opponent} seat {side}")
            target[side] = {"games": games, "wins": wins, "losses": losses, "draws": draws}
    total = sum(row["games"] for seats in result.values() for row in seats.values())
    recorded = summary.get("completed", summary.get("games", total))
    if _count(recorded, "completed games") != total:
        raise ValueError("Summary game total differs from opponent-seat totals")
    return result


def promotion_report(
    summary: dict, required_opponents: Sequence[str], *, min_games: int = 256, min_win_rate: float = 0.65
) -> dict:
    """Qualify every named opponent using balanced fresh confirmation games.

    Counts cannot prove seed independence, frozen checkpoint identity, legality,
    or parity. Those remain separate promotion requirements.
    """
    if (
        isinstance(required_opponents, str)
        or not required_opponents
        or len(set(required_opponents)) != len(required_opponents)
    ):
        raise ValueError("Specify distinct required opponent names")
    if any(not isinstance(name, str) or not name for name in required_opponents):
        raise ValueError("Opponent names must be nonempty strings")
    if isinstance(min_games, bool) or not isinstance(min_games, int) or min_games < 2:
        raise ValueError("Minimum game count must be at least two")
    if not math.isfinite(min_win_rate) or not 0.5 < min_win_rate <= 1:
        raise ValueError("Minimum win rate must be finite and greater than 0.5")
    seats = _seats(summary)
    failed = _count(summary.get("failed", 0), "failed requests")
    reports = {}
    for name in required_opponents:
        rows = seats.get(name, {})
        games = sum(row["games"] for row in rows.values())
        wins = sum(row["wins"] for row in rows.values())
        rate = wins / games if games else 0.0
        lower = wilson_lower(wins, games)
        reasons = []
        if set(rows) != {"0", "1"} or rows.get("0", {}).get("games", 0) != rows.get("1", {}).get("games", 0):
            reasons.append("missing or unbalanced player seats")
        if games < min_games:
            reasons.append("insufficient confirmation games")
        if rate < min_win_rate:
            reasons.append("win rate below target")
        if lower <= 0.5:
            reasons.append("Wilson lower confidence bound does not exceed 0.5")
        reports[name] = {
            "games": games,
            "wins": wins,
            "win_rate": rate,
            "wilson_lower_95": lower,
            "seats": rows,
            "passes": not reasons,
            "reasons": reasons,
        }
    return {
        "schema": "generals-policy-promotion-v1",
        "strength_gate_passes": failed == 0 and all(row["passes"] for row in reports.values()),
        "failed_requests": failed,
        "required_opponents": list(required_opponents),
        "thresholds": {
            "games_per_opponent": min_games,
            "win_rate": min_win_rate,
            "wilson_lower_95_strictly_above": 0.5,
            "balanced_seats": True,
        },
        "opponents": reports,
        "scope": (
            "Strength counts only. Also require fresh independent seeds, frozen policy identity, legality, "
            "serving parity, and throughput qualification before champion promotion."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--opponents", nargs="+", required=True)
    parser.add_argument("--min-games", type=int, default=256)
    parser.add_argument("--min-win-rate", type=float, default=0.65)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = promotion_report(
        json.loads(args.summary.read_text()), args.opponents, min_games=args.min_games, min_win_rate=args.min_win_rate
    )
    data = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(data)
    else:
        print(data, end="")


if __name__ == "__main__":
    main()
