"""Fail-closed checker for the frozen v3-cycle.md promotion experiment (no execution)."""

import argparse
import hashlib
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.compare_agent_runs import compare, indexed_rows, validate_segments

V2 = "be909e6fa3d5b46a3dd2454eaff8a030a8e7d7158f90d484044c36fa06e4650e"
V3 = "01efd251c426f1c928b9416a26ebc493fde4529219c5980aa155604ca40dc896"
ARCHIVES = {
    "v2": "21b5a4125ee91e6230efe1a7764fd7126dbe7c50fec7003d3c349ab53e71e4fd",
    "v3": "805d4884153a25cb7ca03da79959fd4eae12d48100dab5ec410da2007745df06",
}
OPPONENT = "d69cc5d28e4805faf629c5c072a6d23011c727ca4a3b3da9f5e890ea82b78755"
V2_PATH = "generals/agents/sentinel_agent.py"
V3_PATH = "generals/agents/sentinel_v3_agent.py"
RULES = {"max_turns": 1200, "build_castles": True, "deathtouch_turn": 800}
OPTIONS = {"remember_threats": True, "sustained_defense": True}
PACKAGES = {"jax": "0.11.0", "jaxlib": "0.11.0", "numpy": "2.4.6", "scipy": "1.18.0"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def runtime(value):
    require(value["python"].split()[0] == "3.12.10", "wrong measured Python version")
    require(value["packages"] == PACKAGES, "wrong measured runtime package versions")


def seconds(value, deadline):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value < deadline


def win_interval(rows, resamples=100000):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["board_id"], []).append(row["result"] == "win")
    means = np.array([np.mean(grouped[k]) for k in sorted(grouped)])
    rng = np.random.default_rng(113001)
    samples = rng.choice(means, (resamples, len(means)), replace=True).mean(axis=1)
    return float(means.mean()), np.quantile(samples, [0.025, 0.975]).tolist()


class Evidence:
    def __init__(self):
        self.hashes = {}

    def read(self, path):
        path = Path(path)
        data = path.read_bytes()
        self.hashes[str(path.resolve())] = digest(data)
        return data

    def json(self, path):
        return json.loads(self.read(path))

    def archive(self, identity, variant):
        archive = Path(identity["archive"])
        require(
            digest(self.read(archive)) == identity["archive_sha256"] == ARCHIVES[variant],
            f"wrong {variant} frozen archive identity",
        )
        with zipfile.ZipFile(archive) as bundle:
            names = bundle.namelist()
            require(len(names) == len(set(names)), "duplicate archive members")
            manifest = json.loads(bundle.read("manifest.json"))
            require(manifest == identity["manifest"], "archive/metadata manifest mismatch")
            hashes = manifest["source_sha256"]
            require(set(names) == set(hashes) | {"manifest.json"}, "incomplete archive manifest")
            for name, expected in hashes.items():
                require(
                    digest(bundle.read(name)) == expected == identity["file_sha256"].get(name),
                    f"archive/directory source mismatch: {name}",
                )
            require(
                digest(bundle.read("manifest.json")) == identity["file_sha256"].get("manifest.json"),
                "directory manifest bytes mismatch",
            )
        require(
            manifest.get("variant", "v2") == variant and manifest["rules"] == "competition",
            "wrong bundle variant or rules",
        )
        require(hashes[V2_PATH] == V2, "wrong frozen v2 dependency")
        if variant == "v3":
            require(hashes[V3_PATH] == V3, "wrong frozen v3 source")
        require(
            digest(json.dumps(identity["file_sha256"], sort_keys=True).encode()) == identity["directory_sha256"],
            "inconsistent external directory hash",
        )
        return manifest

    def run(self, directory, external, variant):
        directory = Path(directory)
        m = self.json(directory / "metadata.json")
        self.read(directory / "games.csv")
        rows = list(indexed_rows(directory).values())
        count = 32 if external else 64
        require(m["seed"] == 113000 and m["boards"] == count, "wrong reserved seed or map budget")
        require(m.get("diagnostic_turns") is None, "diagnostic games cannot qualify")
        require({r["repeat"] for r in rows} == {"0"}, "exactly one repeat is required")
        require({r["board_id"] for r in rows} == {str(i) for i in range(count)}, "wrong board IDs")
        board_files = list(directory.glob("*boards.npz"))
        require(len(board_files) == 1, "exactly one competition board archive is required")
        self.read(board_files[0])
        with np.load(board_files[0], allow_pickle=False) as boards:
            require(set(boards.files) == {str(i) for i in range(count)}, "incomplete board archive")
            for row in rows:
                board = boards[row["board_id"]]
                require(board.ndim == 2 and all(18 <= n <= 21 for n in board.shape), "noncompetition board shape")
                require(board.shape == (int(row["height"]), int(row["width"])), "CSV/board shape mismatch")
        if external:
            require(
                m.get("schema_version") == 2 and m.get("segments_required") is True,
                "fresh promotion requires immutable-segment runner format",
            )
            for path in (directory / "segments").glob("[0-9][0-9][0-9][0-9].json"):
                self.read(path)
            require(m["rules"] == RULES and len(rows) == 128, "wrong external rules or case count")
            require(all(not r.get("suite") and not r.get("opponent") for r in rows), "unexpected external groups")
            runtime(m["runtime"])
            require(
                m["limits"]["first_seconds"] == 10
                and m["limits"]["turn_seconds"] == 0.15
                and m["limits"]["fault_budget"] == 50,
                "wrong external deadlines/fault budget",
            )
            require(m["opponent"]["archive_sha256"] == OPPONENT, "wrong pinned external opponent")
            require(digest(self.read(m["opponent"]["archive"])) == OPPONENT, "external opponent archive bytes changed")
            with zipfile.ZipFile(m["opponent"]["archive"]) as archive:
                hashes = {name: digest(archive.read(name)) for name in archive.namelist()}
            require(hashes == m["opponent"]["file_sha256"], "external opponent differs from pinned archive")
            require(
                digest(json.dumps(hashes, sort_keys=True).encode()) == m["opponent"]["directory_sha256"],
                "inconsistent opponent directory hash",
            )
            manifest = self.archive(m["candidate"], variant)
            return m, rows, manifest
        require(m["suite_rules"] == {"competition": RULES} and len(rows) == 512, "wrong local rules or case count")
        require(
            {(r["suite"], r["opponent"]) for r in rows} == {("competition", "expander"), ("competition", "hunter")},
            "wrong local matchups",
        )
        require(m["candidate"] == ("sentinel" if variant == "v2" else "sentinel-v3"), "wrong local policy variant")
        require({r["candidate"] for r in rows} == {m["candidate"]}, "CSV/metadata candidate mismatch")
        sources = m["source_hashes"]
        require(sources[V2_PATH] == V2, "wrong local frozen v2 dependency")
        if variant == "v3":
            require(
                sources[V3_PATH] == V3 and m["candidate_options"] == OPTIONS, "wrong local full-v3 identity/options"
            )
        else:
            require(not m.get("candidate_options"), "unexpected v2 policy options")
        return m, rows, None

    def external_frames(self, directory, rows, *, counters=None, cleanup_records=None):
        totals = counters if counters is not None else {}
        totals.update(
            {
                role: {"reply_faults": 0, "invalid_actions": 0, "fatal_reply_forfeits": 0, "process_forfeits": 0}
                for role in ("candidate", "opponent")
            }
        )
        cleanup_records = cleanup_records if cleanup_records is not None else []
        files = list(Path(directory).glob("game-*.json"))
        require(len(files) == len(rows), "missing or extra external game evidence")
        require({int(r["game_id"]) for r in rows} == set(range(len(rows))), "invalid external game IDs")
        budget_reasons = {"deadline", "malformed", "missing_reply", "pending_late_reply"}
        fatal_reasons = {"process_exit", "stdout_overflow"}
        for row in rows:
            raw = self.json(Path(directory) / f"game-{int(row['game_id']):04d}.json")
            cleanup_records.append(
                {
                    "game_id": int(row["game_id"]),
                    "candidate_seat": int(row["seat"]),
                    "cleanup": raw.get("cleanup"),
                    "runner_error": raw.get("runner_error"),
                    "returncodes": [p.get("returncode") for p in raw.get("processes", [])],
                }
            )
            for name in (
                "board_id",
                "repeat",
                "swapped",
                "seat",
                "game_id",
                "height",
                "width",
                "turns",
                "result",
                "reason",
            ):
                require(str(raw["game"][name]) == row[name], f"external raw game/CSV mismatch: {name}")
            frames = raw["frames"]
            require(
                row["reason"] in {"terminal", "turn_cap_draw", "fault_forfeit", "process_forfeit", "both_forfeit"},
                "game has no completed gameplay outcome",
            )
            forfeited = row["reason"] in ("fault_forfeit", "process_forfeit", "both_forfeit")
            expected_frames = int(row["turns"]) + int(forfeited)
            require(len(frames) == expected_frames > 0, "incomplete external frame evidence")
            require([f["turn"] for f in frames] == list(range(len(frames))), "noncontiguous external frames")
            require(len(raw["processes"]) == 2, "missing external process evidence")
            require(raw["runner_error"] is None, "external runner_error invalidates completion")
            cleanup = raw["cleanup"]
            require(
                cleanup["status"] == "complete" and len(cleanup["processes"]) == 2,
                "external cleanup failed or remains incomplete",
            )
            for role, seat in (("candidate", int(row["seat"])), ("opponent", 1 - int(row["seat"]))):
                replies = [f["replies"][seat] for f in frames]
                require(
                    all(r["fault"] in budget_reasons | fatal_reasons | {None} for r in replies),
                    "unknown raw reply fault reason",
                )
                faults = sum(r["fault"] in budget_reasons for r in replies)
                fatal = sum(r["fault"] in fatal_reasons for r in replies)
                invalid = sum(bool(r.get("invalid_action", False)) for r in replies)
                process = raw["processes"][seat]
                require(faults == int(row[f"{role}_faults"]) == process["faults"], "reply fault counters disagree")
                require(
                    invalid == int(row[f"{role}_invalid_actions"]) == process["invalid_actions"],
                    "invalid-action counters disagree",
                )
                totals[role]["reply_faults"] += faults
                totals[role]["invalid_actions"] += invalid
                totals[role]["fatal_reply_forfeits"] += fatal
                totals[role]["process_forfeits"] += int(process["forfeit_reason"] is not None)
                require(
                    process["forfeit_reason"] is None and fatal == 0 and not forfeited,
                    "during-game process/reply forfeit confound",
                )
                closed = cleanup["processes"][seat]
                require(
                    closed["reaped"] is True and closed["final_reap_timeout"] is False and closed["errors"] == [],
                    "external bounded cleanup failed to reap cleanly",
                )
                require(
                    type(closed["returncode"]) is int and closed["returncode"] == process["returncode"],
                    "postgame returncode evidence disagrees",
                )
                for i, reply in enumerate(replies):
                    require(len(frames[i]["replies"]) == 2, "missing paired replies")
                    require(
                        reply["fault"] is None
                        and not reply.get("skipped_observation")
                        and seconds(reply["response_seconds"], 10 if i == 0 else 0.15),
                        "external reply fault/deadline confound",
                    )
        return totals

    def segments(self, directory, metadata, rows):
        result = validate_segments(directory, metadata, rows, require_segments=True)
        for path, expected in result["metadata_sha256"].items():
            require(digest(self.read(path)) == expected, "segment changed while checking")
        return result

    def qualification(self, directory, manifest, archive_sha):
        first, warm, rss = [], [], []
        expected = {f"reused-{h}x{w}.json" for h in range(18, 22) for w in range(18, 22)}
        require(
            {p.name for p in Path(directory).glob("reused-*.json")} == expected, "missing/extra shape qualification"
        )
        for h in range(18, 22):
            for w in range(18, 22):
                report = self.json(Path(directory) / f"reused-{h}x{w}.json")
                require(report["shape"] == [h, w] and report["cache_mode"] == "reused", "wrong probe shape/cache mode")
                require(
                    report["bundle_sha256"] == archive_sha and report["manifest"] == manifest,
                    "qualification bundle/manifest identity mismatch",
                )
                runtime(report["runtime"])
                require(
                    report["limits"]["first_seconds"] == 10
                    and report["limits"]["turn_seconds"] == 0.15
                    and report["limits"]["fault_budget"] == 50,
                    "wrong qualification deadline limits",
                )
                frames = report["frames"]
                require(
                    len(frames) == 30 and [r["index"] for r in frames] == list(range(30)), "incomplete probe frames"
                )
                require(report["exit_code"] == 0 and report["faults"] == 0, "probe exit/fault failure")
                for i, frame in enumerate(frames):
                    deadline = 10 if i == 0 else 0.15
                    require(
                        frame["fault"] is False
                        and frame["deadline_seconds"] == deadline
                        and seconds(frame["reply_seconds"], deadline),
                        "raw probe reply missed deadline/faulted",
                    )
                    require(
                        len(frame["action"]) == 5 and all(type(v) is int for v in frame["action"]),
                        "malformed raw probe action",
                    )
                    (first if i == 0 else warm).append(frame["reply_seconds"])
                require(
                    type(report["peak_rss_bytes"]) is int and 0 < report["peak_rss_bytes"] <= 2**31,
                    "missing sampled RSS or sampled RSS exceeds 2GiB",
                )
                rss.append(report["peak_rss_bytes"])
        return {
            "shapes": 16,
            "responses": 480,
            "reply_faults": 0,
            "max_first_seconds": max(first),
            "max_ordinary_seconds": max(warm),
            "peak_sampled_rss_bytes": max(rss),
            "runtime": PACKAGES,
        }


def check(local_v2, local_v3, external_v2, external_v3, qualification):
    evidence = Evidence()
    result = {
        "promote": False,
        "evidence_valid": False,
        "errors": [],
        "gates": {},
        "thresholds": {
            "seed": 113000,
            "local_win_rate_min": 0.85,
            "local_win_ci_lower_strict_min": 0.70,
            "local_score_delta_min": -0.03,
            "external_delta_ci_lower_strict_min": 0,
            "sampled_rss_bytes_max": 2**31,
            "bootstrap_resamples": 100000,
            "win_bootstrap_seed": 113001,
        },
        "input_sha256": evidence.hashes,
        "limits": "Local shared-host evidence, sampled RSS, no hard memory/network sandbox. Synthetic runtime frames; "
        "bootstrap covers map sampling, not training seeds or leaderboard rank. No engine replay or signatures: "
        "the checker validates retained records and hashes, not independently attested execution.",
    }
    try:
        local = [evidence.run(p, False, v) for p, v in ((local_v2, "v2"), (local_v3, "v3"))]
        external = [evidence.run(p, True, v) for p, v in ((external_v2, "v2"), (external_v3, "v3"))]
        result["external_csv_counters"] = {
            name: {
                field: sum(int(r[field]) for r in rows)
                for field in (
                    "candidate_faults",
                    "opponent_faults",
                    "candidate_invalid_actions",
                    "opponent_invalid_actions",
                )
            }
            for name, (_, rows, _) in zip(("v2", "v3"), external)
        }
        # compare() rejects actual board, physics, runner, opponent and option mismatches.
        result["local_comparison"] = compare(local_v2, local_v3)
        result["external_comparison"] = compare(external_v2, external_v3)
        require(
            set(result["local_comparison"]["matchups"]) == {"competition/expander", "competition/hunter"},
            "incomplete local comparison",
        )
        for value in result["local_comparison"]["matchups"].values():
            require(value["maps"] == 64 and value["games_per_candidate"] == 256, "wrong local map cluster budget")
        external_match = result["external_comparison"]["matchups"]
        require(
            set(external_match) == {"external"}
            and external_match["external"]["maps"] == 32
            and external_match["external"]["games_per_candidate"] == 128,
            "wrong external map cluster budget",
        )
        for metadata, _, manifest in external:
            for name in (
                V2_PATH,
                V3_PATH,
                "generals/core/action.py",
                "generals/core/observation.py",
                "generals/agents/agent.py",
            ):
                if name in manifest["source_sha256"]:
                    require(
                        manifest["source_sha256"][name] == local[1][0]["source_hashes"][name],
                        f"local/bundle dependency mismatch: {name}",
                    )
        result["external_counters"] = {}
        result["external_counters_complete"] = False
        result["external_cleanup"] = {}
        result["external_segments"] = {}
        for name, directory, (metadata, rows, _) in zip(("v2", "v3"), (external_v2, external_v3), external):
            result["external_segments"][name] = evidence.segments(directory, metadata, rows)
            result["external_counters"][name] = {}
            result["external_cleanup"][name] = []
            evidence.external_frames(
                directory,
                rows,
                counters=result["external_counters"][name],
                cleanup_records=result["external_cleanup"][name],
            )
        result["external_counters_complete"] = True
        result["runtime"] = evidence.qualification(qualification, external[1][2], ARCHIVES["v3"])
        result["evidence_valid"] = True
        for opponent in ("expander", "hunter"):
            rows = [r for r in local[1][1] if r["opponent"] == opponent]
            win, interval = win_interval(rows)
            delta = result["local_comparison"]["matchups"][f"competition/{opponent}"]["paired_score_difference"]
            result["gates"][opponent] = {
                "pass": win >= 0.85 and interval[0] > 0.70 and delta >= -0.03,
                "win_rate": win,
                "win_rate_ci95": interval,
                "score_difference": delta,
            }
        interval = external_match["external"]["paired_score_difference_ci95"]
        result["gates"]["external_improvement"] = {"pass": interval[0] > 0, "score_difference_ci95": interval}
        result["promote"] = all(gate["pass"] for gate in result["gates"].values())
    except (ValueError, KeyError, TypeError, OSError, IndexError, zipfile.BadZipFile) as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("local-v2", "local-v3", "external-v2", "external-v3", "qualification", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    result = check(args.local_v2, args.local_v3, args.external_v2, args.external_v3, args.qualification)
    result["checker_sha256"] = digest(Path(__file__).read_bytes())
    result["comparison_checker_sha256"] = digest(Path(__file__).with_name("compare_agent_runs.py").read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("promote", "evidence_valid", "errors", "gates")}, indent=2))
    raise SystemExit(0 if result["promote"] else 1)


if __name__ == "__main__":
    main()
