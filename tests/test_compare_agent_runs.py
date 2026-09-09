import csv
import hashlib
import json

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
