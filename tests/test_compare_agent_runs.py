import csv
import json

import numpy as np
import pytest

from scripts.compare_agent_runs import compare


def write_run(path, result):
    path.mkdir()
    (path / "metadata.json").write_text(json.dumps({"rules": {"max_turns": 1200}}))
    np.savez(path / "boards.npz", **{"0": np.array([[1, 0], [0, 2]])})
    with (path / "games.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["board_id", "repeat", "swapped", "seat", "result"])
        writer.writeheader()
        for swapped in (0, 1):
            for seat in (0, 1):
                writer.writerow(dict(board_id=0, repeat=0, swapped=swapped, seat=seat, result=result))


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
