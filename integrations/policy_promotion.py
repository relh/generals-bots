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


def strength_report(
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
    pending = _count(summary.get("pending", 0), "pending episodes")
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
        "schema": "generals-policy-strength-v1",
        "strength_gate_passes": failed == 0 and pending == 0 and all(row["passes"] for row in reports.values()),
        "failed_requests": failed,
        "pending_episodes": pending,
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


def promotion_report(
    summary: dict, required_opponents: Sequence[str], *, panel: Path,
    checkpoint_sha256: str, image_digest: str, source_commit: str,
    min_games: int = 256, min_win_rate: float = 0.65,
) -> dict:
    """Qualify the current hosted schema against its retained panel evidence."""
    from integrations.hosted_policy import load_panel, summarize

    if summary.get("schema") != "generals-hosted-results-v1":
        raise ValueError("Operational promotion requires the current hosted results schema")
    intent = load_panel(panel)
    for field, expected in (("checkpoint_sha256", checkpoint_sha256),
                            ("image_digest", image_digest), ("source_commit", source_commit)):
        if not expected or intent[field] != expected or summary[field] != expected:
            raise ValueError("Frozen promotion identity differs: " + field)
    if summary["policy_id"] != intent["policy_id"]:
        raise ValueError("Frozen promotion policy version differs")
    states = {}
    request_ids = set()
    for label in intent["requests"]:
        receipt = json.loads((panel / (label + "-receipt.json")).read_text())
        state = json.loads((panel / (label + "-state.json")).read_text())
        if state["id"] != receipt["id"] or receipt["id"] in request_ids:
            raise ValueError("Promotion request receipt identity differs or duplicates")
        request_ids.add(receipt["id"])
        states[label] = state
    # Recompute from retained current states: payload hashes, exact rosters,
    # outcomes, pending/failure counts and costs must match the claimed summary.
    if summary != summarize(intent, states):
        raise ValueError("Promotion summary differs from retained panel states; collect again")
    report = strength_report(summary, required_opponents, min_games=min_games, min_win_rate=min_win_rate)
    incomplete = summary["pending"] > 0 or summary["completed"] != intent["total_episodes"]
    if any(state["status"] != "completed" for state in states.values()):
        incomplete = True
    report.update(
        schema="generals-policy-promotion-v2",
        strength_gate_passes=report["strength_gate_passes"] and not incomplete,
        evidence_gate_passes=not incomplete and summary["failed"] == 0,
        evidence_reasons=["hosted panel is incomplete"] if incomplete else [],
        frozen_identity={"policy_id": intent["policy_id"], "checkpoint_sha256": checkpoint_sha256,
                         "image_digest": image_digest, "source_commit": source_commit},
        request_ids=sorted(request_ids),
        intended_episodes=intent["total_episodes"],
        scope="Retained hosted panel identity, completeness and strength. Also require fresh seeds, "
              "legality, serving parity and qualified training throughput before champion promotion.",
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True, help="Preserved intent, payloads, receipts and states")
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--opponents", nargs="+", required=True)
    parser.add_argument("--min-games", type=int, default=256)
    parser.add_argument("--min-win-rate", type=float, default=0.65)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = promotion_report(
        json.loads(args.summary.read_text()), args.opponents, panel=args.panel,
        checkpoint_sha256=args.checkpoint_sha256, image_digest=args.image_digest,
        source_commit=args.source_commit, min_games=args.min_games, min_win_rate=args.min_win_rate
    )
    data = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(data)
    else:
        print(data, end="")


if __name__ == "__main__":
    main()
