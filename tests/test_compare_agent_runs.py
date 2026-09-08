import csv
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
    metadata = {"rules": {"max_turns": 1200}, "source_hashes": sources}
    if not local:
        metadata["opponent"] = {"directory_sha256": "fixture-opponent"}
    (path / "metadata.json").write_text(json.dumps(metadata))
    np.savez(path / "boards.npz", **{"0": np.array([[1, 0], [0, 2]])})
    with (path / "games.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["opponent", "board_id", "repeat", "swapped", "seat", "result"])
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
