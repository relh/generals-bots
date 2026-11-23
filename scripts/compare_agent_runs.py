"""Compare complete paired arena runs, resampling independent maps.

The output JSON includes paired_maps. Each map identifies a base grid by the
SHA-256 of its C-contiguous NumPy .npy encoding. Decode its cases using
case_columns. A null action_seed means the runner did not record one. For each
matchup and seat, equal_map_seat_deltas averages candidate minus control score
(win=1, draw=0.5, loss=0) within each board hash, then across distinct hashes.
board_hashes_by_seat lists the hashes contributing paired cases (the score
denominator) and hashes with unmatched cases in each arm. A partially matched
hash can appear in both lists. Unpaired comparisons fail before scoring.
"""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np


def indexed_rows(directory):
    rows = list(csv.DictReader((directory / "games.csv").open()))
    indexed = {}
    for row in rows:
        if row["result"] not in ("win", "loss", "draw"):
            raise ValueError("unfinished/unknown outcomes cannot enter strength comparisons")
        key = tuple(row.get(k, "") for k in ("suite", "opponent", "board_id", "repeat", "swapped", "seat"))
        if not (row.get("board_id") or "").strip():
            raise ValueError(f"missing map identity in {directory}: blank board_id")
        if key in indexed:
            raise ValueError(f"duplicate case in {directory}: {key}")
        indexed[key] = row
    if not indexed:
        raise ValueError(f"empty run in {directory}: no paired records")
    return indexed


def validate_segments(directory, metadata, rows, *, require_segments=False):
    """Require every external execution segment to retain the original identities.

    Completed legacy runs predate resume support. New formats explicitly require
    segment evidence; any recorded segments are checked, including on old runs.
    """
    directory = Path(directory)
    files = sorted((directory / "segments").glob("[0-9][0-9][0-9][0-9].json"))
    recorded = list((directory / "segments").glob("*.json"))
    if {p for p in recorded if not p.name.endswith(".result.json")} != set(files):
        raise ValueError("unexpected immutable runner segment filenames")
    if recorded and not files:
        raise ValueError("missing immutable runner segment metadata")
    new_format = type(metadata.get("schema_version")) is int and metadata["schema_version"] >= 2
    if not files and not require_segments and not metadata.get("segments_required", False) and not new_format:
        return {"segments": 0, "legacy_nonresumable": True, "metadata_sha256": {}}
    if not files or [p.stem for p in files] != [f"{i:04d}" for i in range(len(files))]:
        raise ValueError("missing/noncontiguous immutable runner segments")
    original_hash = hashlib.sha256((directory / "metadata.json").read_bytes()).hexdigest()
    try:
        game_ids = [int(row["game_id"]) for row in rows]
    except (KeyError, ValueError) as exc:
        raise ValueError("missing/invalid external game IDs for segment validation") from exc
    if len(game_ids) != len(set(game_ids)):
        raise ValueError("duplicate external game IDs")
    previous_completed = set()
    hashes = {}
    for path in files:
        data = path.read_bytes()
        hashes[str(path.resolve())] = hashlib.sha256(data).hexdigest()
        segment = json.loads(data)
        if "source_equivalence" not in segment or segment["source_equivalence"] is not None:
            raise ValueError("source-equivalence exceptions cannot enter strict comparisons")
        if segment.get("original_metadata_sha256") != original_hash:
            raise ValueError("segment original metadata hash mismatch")
        for key, value in metadata.items():
            if key not in segment or segment[key] != value:
                raise ValueError(f"segment execution identity changed: {key}")
        snapshot = path.with_name(path.stem + "-runner.py")
        if snapshot.exists() or new_format:
            if not snapshot.is_file():
                raise ValueError("missing segment runner source snapshot")
            snapshot_hash = hashlib.sha256(snapshot.read_bytes()).hexdigest()
            sources = segment.get("source_hashes", segment.get("source_sha256", {}))
            if sources.get("scripts/stdio_arena.py") != snapshot_hash:
                raise ValueError("segment runner snapshot identity mismatch")
            hashes[str(snapshot.resolve())] = snapshot_hash
        completed, pending = segment.get("completed_game_ids"), segment.get("pending_game_ids")
        if not isinstance(completed, list) or not isinstance(pending, list):
            raise ValueError("missing segment game IDs")
        if any(type(i) is not int for i in completed + pending):
            raise ValueError("invalid segment game IDs")
        if (
            len(completed) != len(set(completed))
            or len(pending) != len(set(pending))
            or set(completed) & set(pending)
            or set(completed) | set(pending) != set(game_ids)
        ):
            raise ValueError("incomplete/overlapping segment game IDs")
        if not previous_completed <= set(completed):
            raise ValueError("segment completion went backwards")
        previous_completed = set(completed)
    return {"segments": len(files), "original_metadata_sha256": original_hash, "metadata_sha256": hashes}


def validate_planned_cases(metadata, rows, external):
    count, repeats = metadata.get("boards"), metadata.get("repeats", 1 if external else None)
    if type(count) is not int or count < 1 or type(repeats) is not int or repeats < 1:
        raise ValueError("missing/invalid planned board or repeat budget")
    suites = [""] if external else metadata.get("suites")
    opponents = [""] if external else metadata.get("opponents")
    if not isinstance(suites, list) or not suites or not isinstance(opponents, list) or not opponents:
        raise ValueError("missing planned suites/opponents")
    expected = {
        (suite, opponent, str(board), str(repeat), str(swapped), str(seat))
        for suite in suites
        for opponent in opponents
        for board in range(count)
        for repeat in range(repeats)
        for swapped in (0, 1)
        for seat in (0, 1)
    }
    actual = {tuple(r.get(k, "") for k in ("suite", "opponent", "board_id", "repeat", "swapped", "seat")) for r in rows}
    if actual != expected:
        raise ValueError("incomplete map cluster or planned case set")


def board_sha256(grid):
    """Hash the C-order NPY encoding, including shape and dtype."""
    data = io.BytesIO()
    np.save(data, np.ascontiguousarray(grid), allow_pickle=False)
    return hashlib.sha256(data.getbuffer()).hexdigest()


def archive_hashes(directory):
    archives = {}
    for path in sorted(directory.glob("*boards.npz")):
        with np.load(path, allow_pickle=False) as arrays:
            archives[path.name] = {key: board_sha256(arrays[key]) for key in arrays.files}
    if not archives:
        raise ValueError(f"missing map identity: missing board archives in {directory}")
    return archives


def case_board_hashes(rows, archives):
    hashes = {}
    for suite, _, board, *_ in rows:
        if (suite, board) in hashes:
            continue
        archive = f"{suite}_boards.npz" if suite else "boards.npz"
        if archive not in archives and archives.keys() == {"boards.npz"}:
            archive = "boards.npz"
        try:
            hashes[suite, board] = archives[archive][board]
        except KeyError as exc:
            raise ValueError(f"missing map identity: missing board array: {archive}/{board}") from exc
    return hashes


def hash_coverage(left, right, left_hashes, right_hashes):
    coverage = {}
    for key in left.keys() | right.keys():
        suite, opponent, board, _, _, seat = key
        matchup = f"{suite}/{opponent}".strip("/") or "external"
        seats = coverage.setdefault(matchup, {})
        hashes = seats.setdefault(seat, {"paired": set(), "unpaired_control": set(), "unpaired_candidate": set()})
        old = left_hashes[suite, board] if key in left else None
        new = right_hashes[suite, board] if key in right else None
        if old == new and left[key].get("action_seed") == right[key].get("action_seed"):
            hashes["paired"].add(old)
        else:
            if old is not None:
                hashes["unpaired_control"].add(old)
            if new is not None:
                hashes["unpaired_candidate"].add(new)
    return {
        matchup: {
            seat: {label: sorted(values) for label, values in hashes.items()} for seat, hashes in sorted(seats.items())
        }
        for matchup, seats in sorted(coverage.items())
    }


def compare(control, candidate, resamples=100000):
    control, candidate = Path(control), Path(candidate)
    left, right = indexed_rows(control), indexed_rows(candidate)
    archives = [archive_hashes(path) for path in (control, candidate)]
    board_hashes = case_board_hashes(left, archives[0])
    coverage = hash_coverage(left, right, board_hashes, case_board_hashes(right, archives[1]))
    if left.keys() != right.keys():
        control_only = left.keys() - right.keys()
        candidate_only = right.keys() - left.keys()
        raise ValueError(
            f"unpaired cases: control-only={len(control_only)}, candidate-only={len(candidate_only)}; "
            f"example={min(control_only | candidate_only)}; board_hashes_by_seat={json.dumps(coverage)}"
        )
    metadata = [json.loads((p / "metadata.json").read_text()) for p in (control, candidate)]
    # Compare every retained board array, rather than trusting seed labels alone.
    if archives[0] != archives[1]:
        raise ValueError(f"different boards: board_hashes_by_seat={json.dumps(coverage)}")
    for field in ("suite_rules", "rules"):
        if metadata[0].get(field) != metadata[1].get(field):
            raise ValueError(f"different {field}")
    if isinstance(metadata[0].get("opponent"), dict):
        opponent_hashes = [m.get("opponent", {}).get("directory_sha256") for m in metadata]
        if not opponent_hashes[0] or opponent_hashes[0] != opponent_hashes[1]:
            raise ValueError("different or missing external opponent identity")
    sources = [m.get("source_hashes", m.get("source_sha256", {})) for m in metadata]
    required = {
        "generals/core/game.py",
        "generals/core/action.py",
        "generals/core/observation.py",
        "generals/core/grid.py",
        "generals/modifiers/build_castles.py",
        "generals/modifiers/deathtouch.py",
        "generals/evaluation/arena.py",
    }
    external = isinstance(metadata[0].get("opponent"), dict)
    required.add("scripts/stdio_arena.py" if external else "generals/evaluation/cli.py")
    if not external:
        required.add("generals/agents/agent.py")
        for name in {key[1] for key in left}:
            module = name
            for version in (3, 4, 5, 6, 7, 8, 9, 10):
                if name.startswith(f"sentinel-v{version}"):
                    module = f"sentinel_v{version}"
            required.add(f"generals/agents/{module}_agent.py")
            if module in (
                "sentinel_v3",
                "sentinel_v4",
                "sentinel_v5",
                "sentinel_v6",
                "sentinel_v7",
                "sentinel_v8",
                "sentinel_v9",
                "sentinel_v10",
            ):
                required.add("generals/agents/sentinel_agent.py")
            if module in (
                "sentinel_v4",
                "sentinel_v5",
                "sentinel_v6",
                "sentinel_v7",
                "sentinel_v8",
                "sentinel_v9",
                "sentinel_v10",
            ):
                required.add("generals/agents/sentinel_v3_agent.py")
            if module in ("sentinel_v6", "sentinel_v7", "sentinel_v8", "sentinel_v9", "sentinel_v10"):
                required.add("generals/agents/sentinel_v5_agent.py")
            if module in ("sentinel_v7", "sentinel_v8", "sentinel_v9", "sentinel_v10"):
                required.add("generals/agents/sentinel_v6_agent.py")
            if module in ("sentinel_v8", "sentinel_v9", "sentinel_v10"):
                required.add("generals/agents/sentinel_v7_agent.py")
            if module in ("sentinel_v9", "sentinel_v10"):
                required.add("generals/agents/sentinel_v8_agent.py")
            if module == "sentinel_v10":
                required.add("generals/agents/sentinel_v9_agent.py")
    if any(not required <= source.keys() or any(not source[k] for k in required) for source in sources):
        raise ValueError("missing simulator, runner, or opponent source identity")
    physics = [
        {k: v for k, v in source.items() if k.startswith(("generals/core/", "generals/modifiers/"))}
        for source in sources
    ]
    if physics[0] != physics[1] or any(sources[0][k] != sources[1][k] for k in required):
        raise ValueError("different simulator, runner, or opponent sources")
    if metadata[0].get("opponent_options", {}) != metadata[1].get("opponent_options", {}):
        raise ValueError("different opponent options")
    for m, rows in zip(metadata, (left, right)):
        validate_planned_cases(m, rows.values(), external)
    segments = (
        [validate_segments(p, m, rows.values()) for p, m, rows in zip((control, candidate), metadata, (left, right))]
        if external
        else []
    )
    groups = {}
    values = {"win": 1.0, "loss": 0.0, "draw": 0.5}
    for key, old in left.items():
        new = right[key]
        if old.get("action_seed") != new.get("action_seed"):
            raise ValueError(
                f"unpaired case with different action seed: {key}; board_hashes_by_seat={json.dumps(coverage)}"
            )
        group = groups.setdefault(key[:2], {})
        group.setdefault(key[2], []).append((old, new))
    results = {}
    paired_maps = []
    rng = np.random.default_rng(19983)
    for group_key, maps in groups.items():
        suite, opponent = group_key
        seat_deltas = {"0": {}, "1": {}}
        # Each map must contain all seat/label assignments for every repeat.
        for board, pairs in maps.items():
            cases = {(p[0]["repeat"], p[0]["swapped"], p[0]["seat"]) for p in pairs}
            repeats = {p[0]["repeat"] for p in pairs}
            if cases != {(r, s, p) for r in repeats for s in ("0", "1") for p in ("0", "1")}:
                raise ValueError(f"incomplete map cluster: {group_key}/{board}")
            board_hash = board_hashes[suite, board]
            for old, new in pairs:
                deltas = seat_deltas[old["seat"]].setdefault(board_hash, [])
                deltas.append(values[new["result"]] - values[old["result"]])
            paired_maps.append(
                {
                    "suite": suite,
                    "opponent": metadata[0]["opponent"]["directory_sha256"] if external else opponent,
                    "board_id": int(board),
                    "board_sha256": board_hash,
                    "cases": [
                        [
                            int(old["repeat"]),
                            int(old["swapped"]),
                            int(old["seat"]),
                            int(old["action_seed"]) if old.get("action_seed") else None,
                            old["result"],
                            new["result"],
                        ]
                        for old, new in pairs
                    ],
                }
            )
        map_scores = np.array(
            [[np.mean([values[p[i]["result"]] for p in pairs]) for i in (0, 1)] for pairs in maps.values()]
        )
        delta = map_scores[:, 1] - map_scores[:, 0]
        bootstrap = rng.choice(delta, (resamples, len(delta)), replace=True).mean(axis=1)
        all_pairs = [pair for pairs in maps.values() for pair in pairs]
        matchup = "/".join(group_key).strip("/") or "external"
        results[matchup] = {
            "board_hashes_by_seat": coverage[matchup],
            "maps": len(maps),
            "games_per_candidate": len(all_pairs),
            "control_score": float(map_scores[:, 0].mean()),
            "candidate_score": float(map_scores[:, 1].mean()),
            "paired_score_difference": float(delta.mean()),
            "paired_score_difference_ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "counts": {
                role: {outcome: sum(p[i]["result"] == outcome for p in all_pairs) for outcome in values}
                for i, role in enumerate(("control", "candidate"))
            },
            "equal_map_seat_deltas": {
                seat: {
                    "maps": len(by_hash),
                    "paired_cases": sum(len(deltas) for deltas in by_hash.values()),
                    "score_delta": float(np.mean([np.mean(deltas) for deltas in by_hash.values()])),
                }
                for seat, by_hash in seat_deltas.items()
            },
            "board_score_differences": dict(zip(maps, delta.tolist())),
        }
    return {
        "control": str(control),
        "candidate": str(candidate),
        "resamples": resamples,
        "method": "paired map-cluster percentile bootstrap; development comparisons are exploratory",
        "games_sha256": [hashlib.sha256((p / "games.csv").read_bytes()).hexdigest() for p in (control, candidate)],
        "metadata_sha256": [
            hashlib.sha256((p / "metadata.json").read_bytes()).hexdigest() for p in (control, candidate)
        ],
        "paired_maps": {
            "control_seed": metadata[0].get("seed"),
            "candidate_seed": metadata[1].get("seed"),
            "case_columns": ["repeat", "swapped", "seat", "action_seed", "control_result", "candidate_result"],
            "maps": paired_maps,
        },
        "matchups": results,
        "execution_segments": segments,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare(args.control, args.candidate)
    except ValueError as exc:
        parser.exit(1, f"{exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["matchups"], indent=2))


if __name__ == "__main__":
    main()
