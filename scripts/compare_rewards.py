"""Compare terminal-only and shaped PPO from exactly matched initial snapshots.

Run sequential training and frozen-policy evaluation, retaining complete metadata.
The transition budget PER ARM is iterations * num_envs * steps.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

INITIALIZE = """
import json
from dataclasses import asdict
from pathlib import Path
import sys
from generals.training.train import Config, initialize
from generals.training.checkpoint import save_checkpoint
path = Path(sys.argv[1])
manifest = json.loads(path.read_text())
config = Config(**manifest['config_overrides'], shaping_scale=0)
snapshot = initialize(config)
root = Path(manifest['output_root'])
save_checkpoint(root / 'terminal' / 'initial_checkpoint.pkl', snapshot)
snapshot['config']['shaping_scale'] = manifest['shaping_scale']
save_checkpoint(root / 'shaped' / 'initial_checkpoint.pkl', snapshot)
manifest['resolved_terminal_config'] = asdict(config)
manifest['initialization_metadata'] = snapshot['metadata']
path.write_text(json.dumps(manifest, indent=2) + '\\n')
print('Saved matched complete initialization snapshots.', flush=True)
"""


def summarize(output_root):
    """Paired board-cluster bootstrap; scores include timeout draws as half points."""
    import numpy as np

    groups = {}
    for label, directory in (
        ("initial", "terminal/evaluation-initial"),
        ("terminal", "terminal/evaluation"),
        ("shaped", "shaped/evaluation"),
    ):
        with (output_root / directory / "games.csv").open() as stream:
            for row in csv.DictReader(stream):
                key = (row["suite"], row["opponent"])
                groups.setdefault(key, {}).setdefault(label, []).append(row)
    results = {}
    markdown = [
        "# Matched reward comparison",
        "",
        "Score: win=1, draw=0.5, loss=0. These are not win rates.",
        "",
        "| Suite / opponent | Initial | Terminal-only | Shaped | Shaped − terminal, paired95%CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, arms in sorted(groups.items()):
        entry, means = {}, {}
        identities = None
        for label in ("initial", "terminal", "shaped"):
            rows = arms[label]
            current_ids = {
                (row["board_id"], row["repeat"], row["swapped"], row["seat"], row["action_seed"]) for row in rows
            }
            if identities is not None and identities != current_ids:
                raise ValueError(f"unmatched evaluation cases for {key}")
            identities = current_ids
            by_board = {}
            for row in rows:
                by_board.setdefault(int(row["board_id"]), []).append(float(row["score"]))
            means[label] = np.array([np.mean(by_board[board]) for board in sorted(by_board)])
            entry[label] = dict(
                score=float(means[label].mean()),
                games=len(rows),
                wins=sum(row["result"] == "win" for row in rows),
                losses=sum(row["result"] == "loss" for row in rows),
                draws=sum(row["result"] == "draw" for row in rows),
            )
        for first, second in (("terminal", "shaped"), ("initial", "terminal"), ("initial", "shaped")):
            delta = means[second] - means[first]
            rng = np.random.default_rng(93471)
            boot = rng.choice(delta, (10000, len(delta)), replace=True).mean(1)
            entry[f"{second}_minus_{first}"] = dict(
                difference=float(delta.mean()), ci95=np.quantile(boot, [0.025, 0.975]).tolist()
            )
        results["/".join(key)] = entry
        scores = " | ".join(f"{100 * entry[label]['score']:.1f}%" for label in ("initial", "terminal", "shaped"))
        delta = entry["shaped_minus_terminal"]
        lo, hi = delta["ci95"]
        markdown.append(
            f"| {' / '.join(key)} | {scores} | {100 * delta['difference']:+.1f}pp [{100 * lo:+.1f}, {100 * hi:+.1f}] |"
        )
    markdown += [
        "",
        "Intervals cluster paired seats and spawn assignments on board identity.",
        "They do not measure training-seed uncertainty and are not adjusted for multiple comparisons.",
        "Use additional training seeds before claiming a reliable reward advantage.",
        "",
    ]
    (output_root / "comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    (output_root / "REPORT.md").write_text("\n".join(markdown))


def execute(command, log, cwd, environment):
    print(json.dumps(dict(event="start", command=command, log=str(log))), flush=True)
    start = time.monotonic()
    with log.open("w") as output:
        child = subprocess.Popen(
            command, cwd=cwd, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        try:
            for line in child.stdout:
                output.write(line)
                output.flush()
                print(line, end="", flush=True)
            status = child.wait()
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait()
        if status:
            raise subprocess.CalledProcessError(status, command)
    print(json.dumps(dict(event="complete", wall_seconds=time.monotonic() - start)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--platform", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--iterations", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--board-size", type=int, default=8)
    parser.add_argument("--pool-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--shaping-scale", type=float, default=0.2)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=16)
    parser.add_argument("--training-opponents", default="random,expander,hunter")
    parser.add_argument("--evaluation-opponents", nargs="+", default=["random", "expander", "hunter", "harvester"])
    parser.add_argument("--evaluation-seed", type=int, default=41000)
    parser.add_argument("--boards", type=int, default=32)
    parser.add_argument("--suites", nargs="+", default=["classic8", "terrain4"])
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="summarize existing completed evaluations")
    args = parser.parse_args()
    if args.report_only:
        summarize(args.output_root.resolve())
        return
    for name in (
        "iterations",
        "num_envs",
        "steps",
        "board_size",
        "pool_size",
        "width",
        "epochs",
        "minibatch_size",
        "save_every",
        "boards",
    ):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if not 0 < args.shaping_scale <= 1 or not 0 < args.gamma <= 1:
        parser.error("shaping-scale and gamma must be in (0, 1]")
    root = Path(__file__).resolve().parents[1]
    output_root = args.output_root.resolve()
    config = {
        name: getattr(args, name)
        for name in (
            "seed",
            "iterations",
            "num_envs",
            "steps",
            "board_size",
            "pool_size",
            "gamma",
            "width",
            "epochs",
            "minibatch_size",
            "save_every",
        )
    }
    config["opponents"] = args.training_opponents
    manifest = dict(
        output_root=str(output_root),
        config_overrides=config,
        shaping_scale=args.shaping_scale,
        treatment_difference="shaping_scale",
        environment_steps_per_arm=args.iterations * args.num_envs * args.steps,
        order=["terminal", "shaped"],
        platform=args.platform,
        evaluation=dict(
            seed=args.evaluation_seed, boards=args.boards, suites=args.suites, opponents=args.evaluation_opponents
        ),
        commands=[],
    )
    for arm in manifest["order"]:
        directory = output_root / arm
        command = [
            sys.executable,
            "-m",
            "generals.training.train",
            "--output",
            str(directory),
            "--resume",
            str(directory / "initial_checkpoint.pkl"),
            "--iterations",
            str(args.iterations),
        ]
        manifest["commands"].append(command)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    if any((output_root / arm / "initial_checkpoint.pkl").exists() for arm in manifest["order"]):
        parser.error("output-root already contains an experiment; choose a fresh directory")
    for arm in manifest["order"]:
        (output_root / arm).mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    environment = {key: value for key, value in os.environ.items() if key != "LD_LIBRARY_PATH"}
    environment.update(JAX_PLATFORMS=args.platform, XLA_PYTHON_CLIENT_PREALLOCATE="false")
    execute(
        [sys.executable, "-c", INITIALIZE, str(manifest_path)], output_root / "initialization.log", root, environment
    )
    # Initialization fills in every resolved Config default and source/device metadata.
    manifest = json.loads(manifest_path.read_text())
    for arm, command in zip(manifest["order"], manifest["commands"]):
        directory = output_root / arm
        execute(command, directory / "training.log", root, environment)
        shutil.copyfile(directory / "checkpoint.pkl", directory / "final_checkpoint.pkl")
        for kind in ("initial", "final"):
            checkpoint = directory / f"{kind}_checkpoint.pkl"
            manifest.setdefault("checkpoint_sha256", {})[f"{arm}/{kind}"] = hashlib.sha256(
                checkpoint.read_bytes()
            ).hexdigest()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if not args.skip_evaluation:
        evaluations = [("terminal", "initial_checkpoint.pkl", "evaluation-initial")]
        evaluations += [(arm, "final_checkpoint.pkl", "evaluation") for arm in manifest["order"]]
        for arm, checkpoint, result in evaluations:
            directory = output_root / arm
            command = [
                sys.executable,
                "-m",
                "generals.evaluation.cli",
                "--candidate",
                "learned",
                "--checkpoint",
                str(directory / checkpoint),
                "--suites",
                *args.suites,
                "--opponents",
                *args.evaluation_opponents,
                "--seed",
                str(args.evaluation_seed),
                "--boards",
                str(args.boards),
                "--output",
                str(directory / result),
            ]
            manifest.setdefault("evaluation_commands", []).append(command)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            execute(command, directory / (result + ".log"), root, environment)
        summarize(output_root)
    manifest["complete"] = True
    manifest["evaluation_complete"] = not args.skip_evaluation
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
