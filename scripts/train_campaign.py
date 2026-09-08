"""Train a checkpoint in bounded segments with sequential frozen-policy evaluations.

Default targets1024/2048/4096/8192 give16,777,216 total transitions for a32×64 run.
Each segment preserves optimizer/RNG/environment state and evaluates before the
next segment. Restart an interrupted campaign in a fresh output directory using
its latest complete training/checkpoint.pkl as --resume.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def checkpoint_info(checkpoint, root, environment):
    command = [
        sys.executable,
        "-c",
        """
import json, sys
from generals.training.checkpoint import load_checkpoint
p = load_checkpoint(sys.argv[1])
print(json.dumps({k:p[k] for k in ('iteration','environment_steps','config','metadata')}))
""",
        str(checkpoint),
    ]
    env = {**environment, "JAX_PLATFORMS": "cpu"}
    result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def run_child(command, log, root, environment, status, status_path):
    status.update(command=command, log=str(log), child_pid=None)
    atomic_json(status_path, status)
    with log.open("w") as output:
        child = subprocess.Popen(
            command, cwd=root, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        status["child_pid"] = child.pid
        atomic_json(status_path, status)
        try:
            for line in child.stdout:
                output.write(line)
                output.flush()
                print(line, end="", flush=True)
                try:
                    metric = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(metric, dict) and "environment_steps" in metric:
                    status.update(
                        iteration=metric["iteration"],
                        environment_steps=metric["environment_steps"],
                        latest_training_metrics=metric,
                        updated_utc=datetime.now(UTC).isoformat(),
                    )
                    atomic_json(status_path, status)
            returncode = child.wait()
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait()
            status["child_pid"] = None
            atomic_json(status_path, status)
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)


def evaluation_report(directory):
    import numpy as np

    summary = json.loads((directory / "summary.json").read_text())
    groups = {}
    with (directory / "games.csv").open() as stream:
        for row in csv.DictReader(stream):
            groups.setdefault(f"{row['suite']}/{row['opponent']}", []).append(row)
    table = ["| Suite / opponent | Wins | Losses | Draws | Win rate,95%CI | Score |", "|---|---:|---:|---:|---:|---:|"]
    for matchup, result in summary.items():
        rows = groups[matchup]
        board_wins = {}
        for row in rows:
            board_wins.setdefault(row["board_id"], []).append(float(row["result"] == "win"))
        means = np.array([np.mean(values) for values in board_wins.values()])
        rng = np.random.default_rng(59171)
        bootstrap = rng.choice(means, (10000, len(means)), replace=True).mean(1)
        result["win_rate"] = result["wins"] / result["games"]
        result["draw_rate"] = result["draws"] / result["games"]
        result["win_rate_ci95"] = np.quantile(bootstrap, [0.025, 0.975]).tolist()
        low, high = result["win_rate_ci95"]
        table.append(
            f"| {matchup} | {result['wins']} | {result['losses']} | {result['draws']} | "
            f"{100 * result['win_rate']:.1f}% [{100 * low:.1f}, {100 * high:.1f}] | {100 * result['score']:.1f}% |"
        )
    atomic_json(directory / "campaign_summary.json", summary)
    return summary, table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--targets", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    parser.add_argument("--platform", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--boards", type=int, default=32)
    parser.add_argument("--evaluation-seed", type=int, default=51000)
    parser.add_argument("--suites", nargs="+", default=["classic8", "classic12"])
    parser.add_argument("--opponents", nargs="+", default=["random", "expander", "hunter", "harvester"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.boards < 1 or min(args.targets) < 1 or args.targets != sorted(set(args.targets)):
        parser.error("boards must be positive and targets must be positive, unique and increasing")
    root = Path(__file__).resolve().parents[1]
    output = args.output_root.resolve()
    resume = args.resume.resolve()
    environment = {key: value for key, value in os.environ.items() if key != "LD_LIBRARY_PATH"}
    environment.update(JAX_PLATFORMS=args.platform, XLA_PYTHON_CLIENT_PREALLOCATE="false")
    initial = checkpoint_info(resume, root, environment)
    targets = [target for target in args.targets if target > initial["iteration"]]
    if not targets:
        parser.error("all targets are already complete in the supplied checkpoint")
    steps_per_iteration = initial["config"]["num_envs"] * initial["config"]["steps"]
    manifest = dict(
        initial_checkpoint=str(resume),
        initial_checkpoint_sha256=hashlib.sha256(resume.read_bytes()).hexdigest(),
        initial=initial,
        targets=targets,
        additional_environment_steps=(targets[-1] - initial["iteration"]) * steps_per_iteration,
        target_environment_steps=initial["environment_steps"]
        + (targets[-1] - initial["iteration"]) * steps_per_iteration,
        evaluation=dict(seed=args.evaluation_seed, boards=args.boards, suites=args.suites, opponents=args.opponents),
        segments=[],
    )
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    if (output / "manifest.json").exists():
        parser.error("output-root contains a campaign; use a fresh directory and resume its latest checkpoint")
    for folder in ("training", "checkpoints", "evaluations", "logs"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    status = dict(
        state="starting",
        supervisor_pid=os.getpid(),
        child_pid=None,
        iteration=initial["iteration"],
        environment_steps=initial["environment_steps"],
        targets=targets,
        updated_utc=datetime.now(UTC).isoformat(),
    )
    atomic_json(output / "manifest.json", manifest)
    atomic_json(status_path, status)
    report = [
        "# Training campaign",
        "",
        "Fixed development evaluations; these are not the final held-out test.",
        "Wins/losses/draws include every game; score is win=1,draw=0.5,loss=0.",
        "Win intervals bootstrap board clusters and do not measure training-seed uncertainty.",
        "",
    ]

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        for target in targets:
            status.update(state="training", target_iteration=target, checkpoint=str(resume))
            command = [
                sys.executable,
                "-m",
                "generals.training.train",
                "--output",
                str(output / "training"),
                "--resume",
                str(resume),
                "--iterations",
                str(target),
            ]
            run_child(command, output / "logs" / f"train-{target:08d}.log", root, environment, status, status_path)
            frozen = output / "checkpoints" / f"iteration-{target:08d}.pkl"
            temporary = frozen.with_suffix(".tmp")
            shutil.copyfile(output / "training/checkpoint.pkl", temporary)
            temporary.replace(frozen)
            info = checkpoint_info(frozen, root, environment)
            if info["iteration"] != target:
                raise RuntimeError(f"training stopped at {info['iteration']}, expected {target}")
            evaluation = output / "evaluations" / f"iteration-{target:08d}"
            status.update(
                state="evaluating",
                iteration=target,
                environment_steps=info["environment_steps"],
                checkpoint=str(frozen),
            )
            command = [
                sys.executable,
                "-m",
                "generals.evaluation.cli",
                "--candidate",
                "learned",
                "--checkpoint",
                str(frozen),
                "--suites",
                *args.suites,
                "--opponents",
                *args.opponents,
                "--seed",
                str(args.evaluation_seed),
                "--boards",
                str(args.boards),
                "--output",
                str(evaluation),
            ]
            run_child(command, output / "logs" / f"evaluation-{target:08d}.log", root, environment, status, status_path)
            summary, table = evaluation_report(evaluation)
            segment = dict(
                iteration=target,
                environment_steps=info["environment_steps"],
                checkpoint=str(frozen),
                checkpoint_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
                training_metadata=info["metadata"],
                evaluation=str(evaluation),
                results=summary,
            )
            manifest["segments"].append(segment)
            atomic_json(output / "manifest.json", manifest)
            report += [f"## Iteration{target}: {info['environment_steps']:,} transitions", "", *table, ""]
            (output / "REPORT.md").write_text("\n".join(report))
            status["last_completed_evaluation"] = target
            resume = frozen
        status.update(state="complete", checkpoint=str(resume), updated_utc=datetime.now(UTC).isoformat())
        atomic_json(status_path, status)
    except BaseException as error:
        status.update(
            state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=str(error),
            updated_utc=datetime.now(UTC).isoformat(),
        )
        atomic_json(status_path, status)
        raise


if __name__ == "__main__":
    main()
