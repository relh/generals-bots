"""Report map-clustered win intervals and the fixed local promotion gate."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def matchup_report(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["board_id"], []).append(row["result"] == "win")
    means = np.array([np.mean(values) for values in grouped.values()])
    rng = np.random.default_rng(1849)
    bootstrap = rng.choice(means, size=(5000, len(means)), replace=True).mean(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    wins = sum(row["result"] == "win" for row in rows)
    rate = wins / len(rows)
    return dict(
        maps=len(means),
        games=len(rows),
        wins=wins,
        losses=sum(row["result"] == "loss" for row in rows),
        draws=sum(row["result"] == "draw" for row in rows),
        win_rate=rate,
        win_ci95=[float(low), float(high)],
        invalid_moves=sum(int(row["own_invalid_moves"]) for row in rows),
        malformed_commands=sum(int(row.get("own_malformed_commands", 0)) for row in rows),
        builds=sum(int(row["own_builds"]) for row in rows),
        gate=bool(len(means) >= 64 and rate >= 0.85 and low > 0.70),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--require-gate", action="store_true")
    args = parser.parse_args()
    metadata = json.loads((args.run / "metadata.json").read_text())
    with (args.run / "games.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    groups = {}
    for row in rows:
        groups.setdefault(f"{row['suite']}/{row['opponent']}", []).append(row)
    expected = {f"{suite}/{opponent}" for suite in metadata["suites"] for opponent in metadata["opponents"]}
    expected_cases = {
        (str(board), str(repeat), str(swapped), str(seat))
        for board in range(metadata["boards"])
        for repeat in range(metadata["repeats"])
        for swapped in (0, 1)
        for seat in (0, 1)
    }
    complete = set(groups) == expected and all(
        len(group) == len(expected_cases)
        and {(row["board_id"], row["repeat"], row["swapped"], row["seat"]) for row in group} == expected_cases
        for group in groups.values()
    )
    reports = {key: matchup_report(group) for key, group in groups.items()}
    passed = complete and all(row["gate"] for row in reports.values())
    payload = dict(complete=complete, local_gate_passed=passed, matchups=reports)
    (args.run / "gate.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        f"# {metadata['candidate']} evaluation",
        "",
        f"Seed: {metadata['seed']}. Complete planned run: {complete}. Local gate passed: {passed}.",
        "",
        "| Matchup | Maps | W / L / D | Win rate | 95% map interval | Gate |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, result in reports.items():
        low, high = result["win_ci95"]
        lines.append(
            f"| {name} | {result['maps']} | {result['wins']} / {result['losses']} / {result['draws']} "
            f"| {result['win_rate']:.1%} | {low:.1%}–{high:.1%} | {'pass' if result['gate'] else 'fail'} |"
        )
    lines += [
        "",
        "Gate: at least 64 maps, at least 85% wins with draws in the denominator,",
        "and a lower 95% map-bootstrap win-rate bound above 70%, for every planned matchup.",
        "Both starts and player IDs are paired; uncertainty resamples whole base maps.",
        "Bootstrap intervals are descriptive: all-win samples produce a degenerate interval,",
        "which does not establish a true 100% win probability. This gate does not verify",
        "seed secrecy or external tournament strength; those require separate evidence.",
        "",
        "Source hashes, frozen checkpoint identity, exact rules, and commands are in metadata.json.",
        "Per-game diagnostics and boards are retained beside this report.",
    ]
    (args.run / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if args.require_gate and not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
