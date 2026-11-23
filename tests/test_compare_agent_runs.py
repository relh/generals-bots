import csv
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from scripts.compare_agent_runs import compare


def write_run(path, result, local=False):
    path.mkdir()
    sources = {
        name: "fixture-sha"
        for name in (
            "generals/core/game.py",
            "generals/core/action.py",
            "generals/core/observation.py",
            "generals/core/grid.py",
            "generals/modifiers/build_castles.py",
            "generals/modifiers/deathtouch.py",
            "generals/evaluation/arena.py",
            "scripts/stdio_arena.py",
            "generals/evaluation/cli.py",
            "generals/agents/agent.py",
            "generals/agents/hunter_agent.py",
        )
    }
    metadata = {"rules": {"max_turns": 1200}, "source_hashes": sources, "boards": 1, "repeats": 1}
    if not local:
        metadata["opponent"] = {"directory_sha256": "fixture-opponent"}
        metadata["segments_required"] = True
    else:
        metadata.update(suites=[""], opponents=["hunter"])
    (path / "metadata.json").write_text(json.dumps(metadata))
    if not local:
        (path / "segments").mkdir()
        segment = metadata | {
            "source_equivalence": None,
            "original_metadata_sha256": hashlib.sha256((path / "metadata.json").read_bytes()).hexdigest(),
            "completed_game_ids": [],
            "pending_game_ids": list(range(4)),
        }
        (path / "segments/0000.json").write_text(json.dumps(segment))
    np.savez(path / "boards.npz", **{"0": np.array([[1, 0], [0, 2]])})
    with (path / "games.csv").open("w") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["opponent", "board_id", "repeat", "swapped", "seat", "result", "game_id"]
        )
        writer.writeheader()
        for swapped in (0, 1):
            for seat in (0, 1):
                writer.writerow(
                    dict(
                        opponent="hunter" if local else "",
                        board_id=0,
                        repeat=0,
                        swapped=swapped,
                        seat=seat,
                        result=result,
                        game_id=swapped * 2 + seat,
                    )
                )


def test_pairs_map_clusters_and_counts_draws(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_run(a, "draw")
    write_run(b, "win")
    result = compare(a, b, resamples=100)["matchups"]["external"]
    assert result["maps"] == 1 and result["games_per_candidate"] == 4
    assert result["paired_score_difference"] == 0.5
    assert result["paired_score_difference_ci95"] == [0.5, 0.5]


def test_rejects_equal_seed_labels_with_different_actual_boards(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_run(a, "draw")
    write_run(b, "win")
    np.savez(b / "boards.npz", **{"0": np.array([[2, 0], [0, 1]])})
    with pytest.raises(ValueError, match="different boards"):
        compare(a, b)


def test_rejects_incomplete_identical_case_sets(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for path in (a, b):
        write_run(path, "win")
        csv_path = path / "games.csv"
        csv_path.write_text("\n".join(csv_path.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="incomplete map cluster"):
        compare(a, b)


def test_rejects_different_external_checkpoint_identity(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for path, identity in ((a, "original"), (b, "replacement")):
        write_run(path, "win")
        (path / "metadata.json").write_text(json.dumps({"opponent": {"directory_sha256": identity}}))
    with pytest.raises(ValueError, match="opponent identity"):
        compare(a, b)


@pytest.mark.parametrize(
    "field",
    [
        "generals/agents/hunter_agent.py",
        "generals/evaluation/arena.py",
        "generals/evaluation/cli.py",
        "opponent_options",
        "missing",
    ],
)
def test_rejects_changed_or_missing_local_identity(tmp_path, field):
    a, b = tmp_path / "a", tmp_path / "b"
    write_run(a, "win", local=True)
    write_run(b, "win", local=True)
    path = b / "metadata.json"
    metadata = json.loads(path.read_text())
    if field == "missing":
        metadata.pop("source_hashes")
    elif field == "opponent_options":
        metadata[field] = {"hunter": {"changed": True}}
    else:
        metadata["source_hashes"][field] = "different"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="missing|different"):
        compare(a, b)


@pytest.mark.parametrize(
    "version,dependency",
    [
        (5, "sentinel_agent.py"),
        (5, "sentinel_v3_agent.py"),
        (6, "sentinel_agent.py"),
        (6, "sentinel_v3_agent.py"),
        (6, "sentinel_v5_agent.py"),
        (7, "sentinel_agent.py"),
        (7, "sentinel_v3_agent.py"),
        (7, "sentinel_v5_agent.py"),
        (7, "sentinel_v6_agent.py"),
        (8, "sentinel_agent.py"),
        (8, "sentinel_v3_agent.py"),
        (8, "sentinel_v5_agent.py"),
        (8, "sentinel_v6_agent.py"),
        (8, "sentinel_v7_agent.py"),
        (9, "sentinel_agent.py"),
        (9, "sentinel_v3_agent.py"),
        (9, "sentinel_v5_agent.py"),
        (9, "sentinel_v6_agent.py"),
        (9, "sentinel_v7_agent.py"),
        (9, "sentinel_v8_agent.py"),
    ],
)
def test_opponent_requires_frozen_scoring_dependencies(tmp_path, version, dependency):
    a, b = tmp_path / "a", tmp_path / "b"
    for path in (a, b):
        write_run(path, "win", local=True)
        metadata_path = path / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["opponents"] = [f"sentinel-v{version}-disabled"]
        for name in (
            "sentinel_agent.py",
            "sentinel_v3_agent.py",
            "sentinel_v5_agent.py",
            "sentinel_v6_agent.py",
            "sentinel_v7_agent.py",
            "sentinel_v8_agent.py",
            "sentinel_v9_agent.py",
        ):
            metadata["source_hashes"][f"generals/agents/{name}"] = "fixture-sha"
        metadata_path.write_text(json.dumps(metadata))
        csv_path = path / "games.csv"
        csv_path.write_text(csv_path.read_text().replace("hunter", f"sentinel-v{version}-disabled"))
    assert compare(a, b, resamples=10)["matchups"]
    path = b / "metadata.json"
    metadata = json.loads(path.read_text())
    del metadata["source_hashes"][f"generals/agents/{dependency}"]
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="missing"):
        compare(a, b, resamples=10)


@pytest.mark.parametrize("mutation", ["missing", "source", "exception"])
def test_external_segment_cannot_hide_different_execution(tmp_path, mutation):
    a, b = tmp_path / "a", tmp_path / "b"
    write_run(a, "draw")
    write_run(b, "win")
    path = b / "segments/0000.json"
    if mutation == "missing":
        path.unlink()
    else:
        segment = json.loads(path.read_text())
        if mutation == "source":
            segment["source_hashes"]["scripts/stdio_arena.py"] = "new-runner"
        else:
            segment["source_equivalence"] = {"approved": True}
        path.write_text(json.dumps(segment))
    with pytest.raises(ValueError, match="segment|source-equivalence"):
        compare(a, b, resamples=100)


def test_same_identity_resume_segments_pass(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_run(a, "draw")
    write_run(b, "win")
    segment = json.loads((b / "segments/0000.json").read_text())
    segment.update(completed_game_ids=[0, 1], pending_game_ids=[2, 3])
    (b / "segments/0001.json").write_text(json.dumps(segment))
    result = compare(a, b, resamples=100)
    assert result["execution_segments"][1]["segments"] == 2


def test_completed_legacy_nonresumable_runs_remain_comparable(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for path in (a, b):
        write_run(path, "draw")
        (path / "segments/0000.json").unlink()
        metadata = json.loads((path / "metadata.json").read_text())
        metadata.pop("segments_required")
        (path / "metadata.json").write_text(json.dumps(metadata))
    result = compare(a, b, resamples=100)
    assert result["execution_segments"][0]["legacy_nonresumable"]
    board = result["paired_maps"]["maps"][0]
    assert board["opponent"] == "fixture-opponent"
    seed_column = result["paired_maps"]["case_columns"].index("action_seed")
    assert all(case[seed_column] is None for case in board["cases"])


def test_identically_missing_planned_board_does_not_count_as_complete(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for path in (a, b):
        write_run(path, "win")
        original = list(csv.DictReader((path / "games.csv").open()))
        rows = []
        for board in range(7):
            rows.extend([r | {"board_id": str(board), "game_id": str(board * 4 + i)} for i, r in enumerate(original)])
        with (path / "games.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        metadata = json.loads((path / "metadata.json").read_text())
        metadata["boards"] = 8
        (path / "metadata.json").write_text(json.dumps(metadata))
        np.savez(path / "boards.npz", **{str(i): np.array([[1, 0], [0, 2]]) for i in range(7)})
    with pytest.raises(ValueError, match="planned case set"):
        compare(a, b, resamples=100)


def test_export_preserves_matched_outcomes_and_board_identity(tmp_path):
    a, b = tmp_path / "control", tmp_path / "candidate"
    for path in (a, b):
        write_run(path, "draw", local=True)
        metadata = json.loads((path / "metadata.json").read_text())
        metadata.update(boards=3, suites=["tiny"], seed=19)
        (path / "metadata.json").write_text(json.dumps(metadata))
        (path / "boards.npz").unlink()
        grids = {str(i): np.array([[1, i // 2], [0, 2]]) for i in range(3)}
        if path == a:
            np.savez_compressed(path / "tiny_boards.npz", **grids)
        else:
            np.savez(path / "tiny_boards.npz", **{k: np.asfortranarray(v) for k, v in grids.items()})
        with (path / "games.csv").open() as stream:
            template = list(csv.DictReader(stream))
        rows = []
        for board in range(3):
            for row in template:
                seat = int(row["seat"])
                candidate = ("win", "loss") if board < 2 else ("draw", "win")
                result = "draw" if path == a else candidate[seat]
                rows.append(dict(row, suite="tiny", board_id=board, action_seed=19 + board, result=result))
        with (path / "games.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows if path == a else reversed(rows))
    output = tmp_path / "comparison.json"
    script = Path(__file__).resolve().parents[1] / "scripts/compare_agent_runs.py"
    stdout = subprocess.check_output([sys.executable, script, a, b, "--output", output], text=True)
    report = json.loads(output.read_text())
    assert json.loads(stdout) == report["matchups"]
    matchup = report["matchups"]["tiny/hunter"]
    assert matchup["board_score_differences"] == {"0": 0, "1": 0, "2": 0.25}
    assert matchup["equal_map_seat_deltas"] == {
        "0": {"maps": 2, "paired_cases": 6, "score_delta": 0.25},
        "1": {"maps": 2, "paired_cases": 6, "score_delta": 0.0},
    }
    paired = report["paired_maps"]
    assert paired["control_seed"] == paired["candidate_seed"] == 19
    assert len(paired["maps"]) == 3
    for board in paired["maps"]:
        i = board["board_id"]
        assert (board["suite"], board["opponent"]) == ("tiny", "hunter")
        with zipfile.ZipFile(a / "tiny_boards.npz") as archive:
            assert board["board_sha256"] == hashlib.sha256(archive.read(f"{i}.npy")).hexdigest()
        cases = [dict(zip(paired["case_columns"], row, strict=True)) for row in board["cases"]]
        assert len(cases) == 4
        for case in cases:
            assert case["action_seed"] == 19 + i and case["repeat"] == 0
            assert case["control_result"] == "draw"
            outcomes = ("win", "loss") if i < 2 else ("draw", "win")
            assert case["candidate_result"] == outcomes[case["seat"]]


@pytest.mark.parametrize(
    "defect,diagnostic",
    [
        ("duplicate", "duplicate case"),
        ("unpaired", "unpaired cases: control-only=1, candidate-only=0"),
        ("blank_board", "missing map identity"),
        ("missing_array", "missing map identity"),
    ],
)
def test_invalid_pairs_have_distinct_diagnostics(tmp_path, defect, diagnostic):
    control, candidate = tmp_path / "control", tmp_path / "candidate"
    for path in (control, candidate):
        write_run(path, "draw")
    path = candidate / "games.csv"
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    if defect == "duplicate":
        rows.append(rows[0])
    elif defect == "unpaired":
        rows.pop()
    elif defect == "blank_board":
        rows[0]["board_id"] = ""
    else:
        np.savez(candidate / "boards.npz")
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match=diagnostic):
        compare(control, candidate, resamples=10)
