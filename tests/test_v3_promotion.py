import csv
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from scripts import check_v3_promotion as gate


def write_json(path, value):
    Path(path).write_text(json.dumps(value))


def mutate(path, function):
    value = json.loads(path.read_text())
    function(value)
    write_json(path, value)


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    bodies = {
        gate.V2_PATH: b"frozen v2",
        gate.V3_PATH: b"frozen v3",
        "generals/core/action.py": b"action",
        "generals/core/observation.py": b"obs",
        "generals/agents/agent.py": b"agent",
        "run.sh": b"run",
        "main.py": b"main",
    }
    sources = {k: gate.digest(v) for k, v in bodies.items()}
    monkeypatch.setattr(gate, "V2", sources[gate.V2_PATH])
    monkeypatch.setattr(gate, "V3", sources[gate.V3_PATH])
    identities = {}
    archive_hashes = {}
    for variant in ["v2", "v3"]:
        payload = {k: v for k, v in bodies.items() if variant == "v3" or k != gate.V3_PATH}
        manifest = {
            "variant": variant,
            "rules": "competition",
            "source_sha256": {k: gate.digest(v) for k, v in payload.items()},
        }
        payload["manifest.json"] = json.dumps(manifest).encode()
        archive = tmp_path / (variant + ".zip")
        with zipfile.ZipFile(archive, "w") as z:
            for k, v in payload.items():
                z.writestr(k, v)
        archive_hashes[variant] = gate.digest(archive.read_bytes())
        hashes = {k: gate.digest(v) for k, v in payload.items()}
        identities[variant] = {
            "archive": str(archive),
            "archive_sha256": archive_hashes[variant],
            "manifest": manifest,
            "file_sha256": hashes,
            "directory_sha256": gate.digest(json.dumps(hashes, sort_keys=True).encode()),
        }
    monkeypatch.setattr(gate, "ARCHIVES", archive_hashes)
    opponent_archive = tmp_path / "opponent.zip"
    with zipfile.ZipFile(opponent_archive, "w") as z:
        z.writestr("main.py", b"external")
    opponent_files = {"main.py": gate.digest(b"external")}
    monkeypatch.setattr(gate, "OPPONENT", gate.digest(opponent_archive.read_bytes()))
    for k in [
        "generals/core/game.py",
        "generals/core/grid.py",
        "generals/modifiers/build_castles.py",
        "generals/modifiers/deathtouch.py",
        "generals/evaluation/arena.py",
        "generals/evaluation/cli.py",
        "scripts/stdio_arena.py",
        "generals/agents/expander_agent.py",
        "generals/agents/hunter_agent.py",
    ]:
        sources[k] = hashlib.sha256(k.encode()).hexdigest()
    paths = []
    for external, variant in [(False, "v2"), (False, "v3"), (True, "v2"), (True, "v3")]:
        directory = tmp_path / ("external-" if external else "local-") / variant
        directory.mkdir(parents=True)
        paths.append(directory)
        count = 32 if external else 64
        m = {"seed": 113000, "boards": count, "source_hashes": sources, "repeats": 1}
        if external:
            m.update(
                rules=gate.RULES,
                segments_required=True,
                schema_version=2,
                candidate=identities[variant],
                runtime={"python": "3.12.10", "packages": gate.PACKAGES},
                limits={"first_seconds": 10, "turn_seconds": 0.15, "fault_budget": 50},
                opponent={
                    "archive": str(opponent_archive),
                    "archive_sha256": gate.OPPONENT,
                    "directory_sha256": gate.digest(json.dumps(opponent_files, sort_keys=True).encode()),
                    "file_sha256": opponent_files,
                },
            )
        else:
            m.update(suites=["competition"], opponents=["expander", "hunter"])
            m.update(
                suite_rules={"competition": gate.RULES}, candidate="sentinel" if variant == "v2" else "sentinel-v3"
            )
            if variant == "v3":
                m["candidate_options"] = gate.OPTIONS
        write_json(directory / "metadata.json", m)
        if external:
            (directory / "segments").mkdir()
            (directory / "segments/0000-runner.py").write_bytes(b"scripts/stdio_arena.py")
            write_json(
                directory / "segments/0000.json",
                m
                | {
                    "original_metadata_sha256": gate.digest((directory / "metadata.json").read_bytes()),
                    "source_equivalence": None,
                    "completed_game_ids": [],
                    "pending_game_ids": list(range(128)),
                },
            )
        write_json(directory / "summary.json", {"promote": True, "all_deadlines_met": True})
        boards = {}
        rows = []
        for board_id in range(count):
            board = np.zeros((18, 18), dtype=np.int32)
            board[0, 0] = 1
            board[-1, -1] = 2
            board[1, 1] = -board_id - 2
            boards[str(board_id)] = board
            for opponent in [""] if external else ["expander", "hunter"]:
                for swapped in [0, 1]:
                    for seat in [0, 1]:
                        r = {
                            "board_id": board_id,
                            "repeat": 0,
                            "swapped": swapped,
                            "seat": seat,
                            "height": 18,
                            "width": 18,
                            "result": "draw" if external and variant == "v2" else "win",
                        }
                        if external:
                            r.update(
                                game_id=len(rows),
                                turns=1,
                                reason="terminal",
                                candidate_faults=0,
                                opponent_faults=0,
                                candidate_invalid_actions=0,
                                opponent_invalid_actions=0,
                            )
                            reply = {"fault": None, "response_seconds": 1.0, "action": [1, 0, 0, 0, 0]}
                            process = {"faults": 0, "invalid_actions": 0, "returncode": 0, "forfeit_reason": None}
                            write_json(
                                directory / f"game-{len(rows):04d}.json",
                                {
                                    "game": r,
                                    "frames": [{"turn": 0, "replies": [reply, reply], "faults": [0, 0]}],
                                    "processes": [process, process],
                                    "runner_error": None,
                                    "cleanup": {
                                        "status": "complete",
                                        "processes": [
                                            {
                                                "reaped": True,
                                                "returncode": 0,
                                                "final_reap_timeout": False,
                                                "errors": [],
                                                "waits": [{"stage": "eof", "timed_out": False}],
                                            }
                                            for _ in range(2)
                                        ],
                                    },
                                },
                            )
                        else:
                            r.update(suite="competition", opponent=opponent, candidate=m["candidate"])
                        rows.append(r)
        with (directory / "games.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        np.savez(directory / "boards.npz", **boards)
    qualification = tmp_path / "qualification"
    qualification.mkdir()
    paths.append(qualification)
    for h in range(18, 22):
        for w in range(18, 22):
            write_json(
                qualification / f"reused-{h}x{w}.json",
                {
                    "shape": [h, w],
                    "cache_mode": "reused",
                    "bundle_sha256": archive_hashes["v3"],
                    "manifest": identities["v3"]["manifest"],
                    "runtime": {"python": "3.12.10", "packages": gate.PACKAGES},
                    "limits": {"first_seconds": 10, "turn_seconds": 0.15, "fault_budget": 50},
                    "faults": 0,
                    "exit_code": 0,
                    "peak_rss_bytes": 250000000,
                    "all_deadlines_met": True,
                    "frames": [
                        {
                            "index": i,
                            "fault": False,
                            "reply_seconds": 1.0 if i == 0 else 0.01,
                            "deadline_seconds": 10 if i == 0 else 0.15,
                            "action": [1, 0, 0, 0, 0],
                        }
                        for i in range(30)
                    ],
                },
            )
    return paths


def test_complete_raw_evidence_promotes(evidence):
    result = gate.check(*evidence)
    assert result["promote"] and result["evidence_valid"], result["errors"]
    assert result["runtime"]["responses"] == 480
    assert len(result["input_sha256"]) > 270
    assert result["gates"]["external_improvement"]["score_difference_ci95"] == [0.5, 0.5]


@pytest.mark.parametrize(
    "corruption",
    [
        "source",
        "missing_shape",
        "missing_frame",
        "late",
        "runtime",
        "missing_game",
        "wrong_seed",
        "wrong_flags",
        "archive",
    ],
)
def test_green_summaries_do_not_override_bad_evidence(evidence, corruption):
    local2, local3, external2, external3, qualification = evidence
    if corruption == "source":
        mutate(local3 / "metadata.json", lambda m: m["source_hashes"].update({gate.V3_PATH: "wrong"}))
    elif corruption == "missing_shape":
        (qualification / "reused-18x18.json").unlink()
    elif corruption == "missing_frame":
        mutate(qualification / "reused-18x18.json", lambda m: m["frames"].pop())
    elif corruption == "late":
        mutate(qualification / "reused-18x18.json", lambda m: m["frames"][1].update(reply_seconds=0.15))
    elif corruption == "runtime":
        mutate(qualification / "reused-18x18.json", lambda m: m["runtime"]["packages"].update(jax="0.11.1"))
    elif corruption == "missing_game":
        (external3 / "game-0000.json").unlink()
    elif corruption == "wrong_seed":
        mutate(external3 / "metadata.json", lambda m: m.update(seed=83000))
    elif corruption == "wrong_flags":
        mutate(local3 / "metadata.json", lambda m: m["candidate_options"].update(remember_threats=False))
    elif corruption == "archive":
        mutate(external3 / "metadata.json", lambda m: m["candidate"].update(archive_sha256="changed"))
    result = gate.check(*evidence)
    assert not result["promote"] and not result["evidence_valid"] and result["errors"]


def change_local_losses(path, count):
    rows = list(csv.DictReader((path / "games.csv").open()))
    for r in rows:
        if int(r["board_id"]) < count:
            r["result"] = "loss"
    with (path / "games.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_realistic_cluster_thresholds_and_regression(evidence):
    # 55/64 winning map clusters clears 85% and lower>70%, but -2/64 score fails -3pp.
    change_local_losses(evidence[0], 7)
    change_local_losses(evidence[1], 9)
    result = gate.check(*evidence)
    assert result["evidence_valid"] and not result["promote"]
    g = result["gates"]["hunter"]
    assert g["win_rate"] == 55 / 64 and g["win_rate_ci95"][0] > 0.70
    assert g["score_difference"] == -2 / 64 and not g["pass"]


def test_win_rate_threshold_is_not_score_threshold(evidence):
    change_local_losses(evidence[0], 10)
    change_local_losses(evidence[1], 10)
    result = gate.check(*evidence)
    assert result["evidence_valid"] and not result["promote"]
    assert result["gates"]["hunter"]["win_rate"] == 54 / 64


def test_actual_board_mismatch_is_rejected(evidence):
    path = evidence[1] / "boards.npz"
    with np.load(path) as z:
        boards = {k: z[k] for k in z.files}
    boards["0"][4, 4] = -99
    np.savez(path, **boards)
    result = gate.check(*evidence)
    assert not result["promote"] and any("different boards" in e for e in result["errors"])


@pytest.mark.parametrize("change", ["missing", "source", "exception", "budget"])
def test_immutable_segments_must_agree_with_original(evidence, change):
    path = evidence[3] / "segments/0000.json"
    if change == "missing":
        path.unlink()
    elif change == "source":
        mutate(path, lambda m: m["source_hashes"].update({"scripts/stdio_arena.py": "different"}))
    elif change == "exception":
        mutate(path, lambda m: m.update(source_equivalence={"approved": True}))
    else:
        mutate(path, lambda m: m["pending_game_ids"].pop())
    result = gate.check(*evidence)
    assert not result["evidence_valid"] and not result["promote"]
    assert any("segment" in error or "source-equivalence" in error for error in result["errors"])


def test_zero_external_lower_bound_does_not_promote(evidence):
    path = evidence[3]
    rows = list(csv.DictReader((path / "games.csv").open()))
    for row in rows:
        row["result"] = "draw"
        mutate(path / f"game-{int(row['game_id']):04d}.json", lambda m: m["game"].update(result="draw"))
    with (path / "games.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = gate.check(*evidence)
    assert result["evidence_valid"] and not result["promote"]
    assert result["gates"]["external_improvement"]["score_difference_ci95"] == [0, 0]


def test_fault_counters_survive_confounded_rejection(evidence):
    path = evidence[3]
    rows = list(csv.DictReader((path / "games.csv").open()))
    rows[0]["candidate_faults"] = "1"
    with (path / "games.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def fault(raw):
        raw["game"]["candidate_faults"] = 1
        raw["frames"][0]["replies"][0]["fault"] = "deadline"
        raw["processes"][0]["faults"] = 1

    mutate(path / "game-0000.json", fault)
    result = gate.check(*evidence)
    assert not result["evidence_valid"] and not result["promote"]
    assert result["external_csv_counters"]["v3"]["candidate_faults"] == 1
    assert any("confound" in e for e in result["errors"])


def test_qualification_identity_must_match_actual_bundle(evidence):
    mutate(evidence[4] / "reused-18x18.json", lambda m: m.update(bundle_sha256="other"))
    result = gate.check(*evidence)
    assert not result["promote"] and any("identity mismatch" in e for e in result["errors"])


def test_changed_immutable_runner_source_is_rejected(evidence):
    (evidence[3] / "segments/0000-runner.py").write_bytes(b"changed implementation")
    result = gate.check(*evidence)
    assert not result["promote"] and any("runner snapshot identity" in e for e in result["errors"])


@pytest.mark.parametrize("rss,expected", [(2**31, True), (2**31 + 1, False)])
def test_sampled_memory_limit_is_enforced_despite_green_summary(evidence, rss, expected):
    mutate(evidence[4] / "reused-18x18.json", lambda m: m.update(peak_rss_bytes=rss))
    result = gate.check(*evidence)
    assert result["promote"] is expected
    if not expected:
        assert any("sampled RSS exceeds 2GiB" in e for e in result["errors"])


@pytest.mark.parametrize("returncode", [-15, -9])
def test_forced_cleanup_after_finished_game_is_not_a_forfeit(evidence, returncode):
    def cleanup(raw):
        raw["processes"][1]["returncode"] = returncode
        raw["cleanup"]["processes"][1].update(
            returncode=returncode,
            waits=[
                {"stage": "eof", "timed_out": True},
                {"stage": "term", "timed_out": returncode == -9},
                *([{"stage": "kill", "timed_out": False}] if returncode == -9 else []),
            ],
        )

    mutate(evidence[3] / "game-0000.json", cleanup)
    result = gate.check(*evidence)
    assert result["promote"], result["errors"]
    assert result["external_cleanup"]["v3"][0]["returncodes"] == [0, returncode]
    assert result["external_counters"]["v3"]["opponent"]["fatal_reply_forfeits"] == 0


def test_failed_cleanup_is_preserved_and_rejected(evidence):
    def cleanup(raw):
        raw["cleanup"]["status"] = "failed"
        raw["cleanup"]["processes"][1].update(reaped=False, final_reap_timeout=True, returncode=None)
        raw["processes"][1]["returncode"] = None

    mutate(evidence[3] / "game-0000.json", cleanup)
    result = gate.check(*evidence)
    assert not result["promote"] and any("cleanup failed" in e for e in result["errors"])
    assert result["external_cleanup"]["v3"][0]["cleanup"]["processes"][1]["final_reap_timeout"]


@pytest.mark.parametrize("reason", ["process_exit", "stdout_overflow"])
def test_during_game_fatal_forfeit_is_not_a_budget_fault(evidence, reason):
    path = evidence[3]
    rows = list(csv.DictReader((path / "games.csv").open()))
    rows[0].update(turns="0", reason="process_forfeit")
    with (path / "games.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def forfeit(raw):
        raw["game"].update(turns=0, reason="process_forfeit")
        raw["frames"][0]["replies"][0]["fault"] = reason
        raw["processes"][0].update(forfeit_reason=reason, returncode=1)
        raw["cleanup"]["processes"][0]["returncode"] = 1

    mutate(path / "game-0000.json", forfeit)
    result = gate.check(*evidence)
    assert not result["promote"] and any("during-game" in e for e in result["errors"])
    counters = result["external_counters"]["v3"]["candidate"]
    assert counters["reply_faults"] == 0 and counters["fatal_reply_forfeits"] == 1 and counters["process_forfeits"] == 1
    assert not result["external_counters_complete"]
